"""Numerical B300 tests; run with pytest on the GPU node (see BLACKWELL_GEMM.md)."""

import math

import pytest
import torch
import torch.nn.functional as F
import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda

from quack.cute_dsl_utils import get_max_active_clusters
from quack.gemm import build_scatter_row_addresses, gemm
from quack.gemm_config import GemmConfig, cta_tile_shape_m
from quack.gemm_interface import gemm_act

from blackwell_gemm_tables import make_table
from wave_ready import build_wave_ready_prefix


@pytest.fixture(autouse=True)
def blackwell_device():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires a Blackwell datacenter GPU (B200/B300, SM100/SM103)")
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False


def inputs(buffers, *, k=144, n=384, dtype=torch.bfloat16, cluster_m=2, tile_m=128):
    # Empty experts/buffers, nonaligned boundaries, repeated routes, partially
    # occupied two-CTA tiles, and poisoned capacity beyond valid index prefixes.
    counts = [[0, 0, 0, 0] if b == 1 else [1 + b % 3, 0, 67 + b, 131 - b]
              for b in range(buffers)]
    if buffers == 1:
        counts = [[1, 0, 67, 131]]
    xs = tuple(torch.randn(257 + 3 * b, k, device="cuda", dtype=dtype) for b in range(buffers))
    capacity = max(map(sum, counts)) + 11
    indices = []
    for b, cs in enumerate(counts):
        idx = torch.full((capacity,), -987654, dtype=torch.int32, device="cuda")
        idx[:sum(cs)] = torch.randint(xs[b].shape[0], (sum(cs),), device="cuda", dtype=torch.int32)
        indices.append(idx)
    rows, offsets = make_table(counts, n=n, tile_m=tile_m, cluster_m=cluster_m)
    table = torch.tensor(rows, dtype=torch.int32, device="cuda")
    w = torch.randn(4, n, k, device="cuda", dtype=dtype) / math.sqrt(k)
    packed = []
    reference = []
    for e in range(4):
        a = torch.cat([xs[b][indices[b][offsets[b][e]:offsets[b][e + 1]].long()]
                       for b in range(buffers)])
        packed.append(a)
        reference.append(a.float() @ w[e].float().T)
    return xs, tuple(indices), w, table, torch.cat(reference), packed, counts


def launch(xs, indices, w, table, out, *, cluster_m=2, tile_m=128, **kw):
    multi = len(xs) > 1
    gemm(xs if multi else xs[0], w, out, None, None,
         tile_M=tile_m, tile_N=128, cluster_M=cluster_m, cluster_N=1,
         max_swizzle_size=1, is_dynamic_persistent=False, pingpong=False,
         A_idx=indices if multi else indices[0], gather_work_table=table,
         multi_buffer_gather=multi, **kw)


def check(actual, reference, *, reduce=False):
    # Ground truth uses fp32 accumulation; round once at the output boundary.
    # Reduction additionally rounds each routed value before adding in output dtype.
    tol = 0.035 if actual.dtype == torch.bfloat16 else 0.006
    baseline_error = (reference.to(actual.dtype).float() - reference.float()).abs().max().item()
    torch.testing.assert_close(actual.float(), reference.float(),
                               atol=max(tol * (2 if reduce else 1), 2 * baseline_error), rtol=tol)


@pytest.mark.parametrize("cluster_m,tile_m", [(1, 128), (2, 128), (2, 256)])
@pytest.mark.parametrize("buffers", [1, 3, 16])
def test_gather_table(buffers, cluster_m, tile_m):
    xs, idx, w, table, ref, _, _ = inputs(buffers, cluster_m=cluster_m, tile_m=tile_m)
    out = torch.full((sum(i.numel() for i in idx), w.shape[1]), float("nan"),
                     dtype=xs[0].dtype, device="cuda")
    # Reverse descriptor order to expose accidental dependencies on row ordering.
    table = table.flip(0).contiguous()
    launch(xs, idx, w, table, out, cluster_m=cluster_m, tile_m=tile_m)
    check(out[:len(ref)], ref)
    assert torch.isnan(out[len(ref):]).all().item(), "wrote into unused route capacity"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_indexed_table(dtype):
    xs, idx, w, table, ref, _, _ = inputs(1, dtype=dtype)
    cluster_rows = cta_tile_shape_m(128, 2, 10) * 2
    rows = []
    # Indexed mode requires consecutive, increasing N groups per M bundle.
    for e, start, end in table[:, :3].unique(dim=0).cpu().tolist():
        tokens = idx[0][start:end].cpu().tolist()
        for ng in range(3):
            rows.append([e, ng, start, end, *tokens, *([-1] * (cluster_rows - len(tokens)))])
    table = torch.tensor(rows, dtype=torch.int32, device="cuda")
    out = torch.full((idx[0].numel(), w.shape[1]), float("nan"), dtype=dtype, device="cuda")
    unused_idx = torch.full_like(idx[0], -12345)
    launch(xs, (unused_idx,), w, table, out)
    check(out[:len(ref)], ref)
    assert torch.isnan(out[len(ref):]).all().item()


@pytest.mark.parametrize("cluster_m", [1, 2])
def test_swiglu_and_down(cluster_m):
    xs, idx, w, table, pre, packed, _ = inputs(16, n=512, cluster_m=cluster_m)
    config = GemmConfig(tile_m=128, tile_n=128, cluster_m=cluster_m, cluster_n=1,
                        max_swizzle_size=1, is_dynamic_persistent=False,
                        pingpong=False, device_capacity=10)
    _, up = gemm_act(xs, w.transpose(1, 2), activation="swiglu", A_idx=idx,
                     gather_work_table=table, multi_buffer_gather=True,
                     store_preact=False, concat_layout=("B",), tuned=False, config=config)
    reference_up = F.silu(pre[:, :256]) * pre[:, 256:]
    check(up[:len(pre)], reference_up)
    # Feed the actual up output into native down GEMM; check the whole composition
    # against a reference based on the same quantized intermediate.
    counts = [[len(a) for a in packed]]
    rows, offsets = make_table(counts, n=384, cluster_m=cluster_m)
    down_table = torch.tensor(rows, dtype=torch.int32, device="cuda")
    down_idx = torch.arange(len(pre), dtype=torch.int32, device="cuda")
    wd = torch.randn(4, 384, 256, dtype=up.dtype, device="cuda") / 16
    out = torch.empty(len(pre), 384, dtype=up.dtype, device="cuda")
    launch((up,), (down_idx,), wd, down_table, out, cluster_m=cluster_m)
    ref = torch.cat([up[offsets[0][e]:offsets[0][e + 1]].float() @ wd[e].float().T
                     for e in range(4)])
    check(out, ref)


@pytest.mark.parametrize("table_gather", [False, True])
@pytest.mark.parametrize("store_preact", [False, True])
def test_swiglu_two_cta_store_mapping(table_gather, store_preact):
    """One expert and one N tile isolate the half-width epilogue warp mapping.

    Distinct columns expose a displaced N warp; varying rows expose cross-row
    aliasing. M=129 also checks a partial cluster with a fully predicated peer.
    With preact enabled, verify the accumulator output before checking SwiGLU.
    """
    m, k, n = 129, 64, 128
    a = torch.zeros(m, k, dtype=torch.bfloat16, device="cuda")
    a[:, 0] = 1
    a[:, 1] = (torch.arange(m, device=a.device) % 7) / 8
    w = torch.zeros(1, n, k, dtype=a.dtype, device=a.device)
    columns = torch.arange(n // 2, device=a.device)
    w[0, :n // 2, 0] = 1 + (columns % 5) / 8
    w[0, n // 2:, 0] = (columns - n // 4) / 16
    w[0, n // 2:, 1] = 0.25
    config = GemmConfig(
        tile_m=128, tile_n=128, cluster_m=2, cluster_n=1, max_swizzle_size=1,
        is_dynamic_persistent=False, pingpong=False, device_capacity=10,
    )
    table_args = {}
    if table_gather:
        rows, _ = make_table([[m]], n=n, tile_m=128, cluster_m=2)
        table_args = dict(
            A_idx=torch.arange(m, device=a.device, dtype=torch.int32),
            gather_work_table=torch.tensor(rows, device=a.device, dtype=torch.int32),
        )
    preact, postact = gemm_act(
        a, w.transpose(1, 2) if table_gather else w[0].T,
        activation="swiglu", store_preact=store_preact, concat_layout=("B",),
        tuned=False, config=config, **table_args,
    )
    reference = a.float() @ w[0].float().T
    gate, up = reference.chunk(2, dim=-1)
    if store_preact:
        # B is physically [gate; up]; D without concat_layout=("D",) is interleaved.
        check(preact, torch.stack((gate, up), dim=-1).flatten(1))
    check(postact, F.silu(gate) * up)


@pytest.mark.parametrize("store", ["bulk_rows", "bulk_rows_reduce"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cluster_m", [1, 2])
@pytest.mark.parametrize("n", [272, 384])
def test_scatter(store, dtype, cluster_m, n):
    xs, idx, w, table, ref, _, _ = inputs(1, dtype=dtype, cluster_m=cluster_m, n=n)
    # Scatter expects one mapping entry per logical output row, including capacity.
    r = idx[0].numel()
    mapping = torch.randperm(r, device="cuda", dtype=torch.int32)
    reduce = store == "bulk_rows_reduce"
    if reduce:
        mapping = mapping // 2
    out = torch.zeros(r, w.shape[1], dtype=dtype, device="cuda")
    launch(xs, idx, w, table, out, cluster_m=cluster_m,
           epilogue_store=store, scatter_table=mapping)
    expected = torch.zeros_like(out, dtype=torch.float32)
    expected.index_add_(0, mapping[:len(ref)].long(), ref.to(dtype).float())
    check(out, expected, reduce=reduce)


@cute.kernel
def _observe_ready_kernel(buffers: tuple, snapshots: tuple, ready: cute.Tensor):
    """Acquire each ready group, then snapshot its columns while GEMM continues."""
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    for j in cutlass.range_constexpr(len(buffers)):
        if bid == j:
            for group in cutlass.range(buffers[j].shape[0]):
                if tid == 0:
                    value = cute.arch.load((ready.iterator + 1).llvm_ptr, Int32,
                                           sem="acquire", scope="gpu")
                    while value < group + 1:
                        value = cute.arch.load((ready.iterator + 1).llvm_ptr, Int32,
                                               sem="acquire", scope="gpu")
                cute.arch.sync_threads()
                for i in cutlass.range(tid, buffers[j].shape[1], 128):
                    snapshots[j][group, i] = buffers[j][group, i]
                cute.arch.sync_threads()


@cute.jit
def _observe_ready(buffers: tuple, snapshots: tuple, ready: cute.Tensor, stream: cuda.CUstream):
    _observe_ready_kernel(buffers, snapshots, ready).launch(
        grid=(len(buffers), 1, 1), block=(128, 1, 1), stream=stream,
    )


@pytest.mark.parametrize("cluster_m", [1, 2])
def test_multi_buffer_scatter_waves(cluster_m):
    xs, idx, w, table, ref, _, _ = inputs(1, cluster_m=cluster_m)
    r, n, tile_n = idx[0].numel(), w.shape[1], 128
    used = (17, 23, 31)  # Deliberately different logical sizes and allocation capacities.
    outputs = tuple(torch.zeros(n // tile_n, (rows + 7) * tile_n, dtype=xs[0].dtype,
                                 device="cuda") for rows in used)
    mapping = torch.tensor([[i % 3, (i // 3) % used[i % 3]] for i in range(r)],
                           dtype=torch.int32, device="cuda")
    addresses = build_scatter_row_addresses(outputs, mapping, reduce=True,
                                            tile_n=tile_n, used_rows=used)
    active = min(get_max_active_clusters(cluster_m, sm_count=10), len(table))
    prefix = build_wave_ready_prefix(table.cpu().tolist(), group_size=1, num_groups=3,
                                     wave_work_ids=active)
    prefix = torch.tensor(prefix, dtype=torch.int32, device="cuda")
    counter = torch.zeros(2, dtype=torch.int32, device="cuda")
    # Compact prefixes reinterpret allocation capacity exactly as the proxy does.
    compact = tuple(b.flatten()[:3 * rows * tile_n].view(3, rows * tile_n)
                    for b, rows in zip(outputs, used))
    snapshots = tuple(torch.empty_like(b) for b in compact)
    args = (tuple(from_dlpack(b) for b in compact), tuple(from_dlpack(b) for b in snapshots),
            from_dlpack(counter))
    observer = cute.compile(_observe_ready, *args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    def run():
        for b in outputs:
            b.zero_()
        launch(xs, idx, w, table, outputs, cluster_m=cluster_m, sm_count=10,
               epilogue_store="bulk_rows_reduce", scatter_table=addresses,
               multi_buffer_scatter=True, n_tile_major_output=True,
               wave_store_counter=counter, wave_ready_prefix=prefix)
    run()  # Compile before launching either concurrent kernel.
    torch.cuda.synchronize()
    expected = [torch.zeros(rows, n, device="cuda") for rows in used]
    for j in range(3):
        mask = mapping[:len(ref), 0] == j
        expected[j].index_add_(0, mapping[:len(ref)][mask, 1].long(), ref[mask].to(outputs[0].dtype).float())
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    for _ in range(3):
        counter.zero_()
        for b in outputs:
            b.zero_()
        for b in snapshots:
            b.fill_(float("nan"))
        producer.wait_stream(torch.cuda.current_stream())
        consumer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            run()
        observer(*args, cuda.CUstream(consumer.cuda_stream))
        torch.cuda.current_stream().wait_stream(producer)
        torch.cuda.current_stream().wait_stream(consumer)
        for j, rows in enumerate(used):
            logical = snapshots[j].view(3, rows, tile_n).permute(1, 0, 2).reshape(rows, n)
            check(logical, expected[j], reduce=True)
            assert (outputs[j].flatten()[3 * rows * tile_n:] == 0).all().item()
        assert counter.cpu().tolist() == [len(table) * cluster_m, 3]
    # Replays must reset the store counter as well as the additive destinations.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        graph.replay()
        for j, rows in enumerate(used):
            logical = compact[j].view(3, rows, tile_n).permute(1, 0, 2).reshape(rows, n)
            check(logical, expected[j], reduce=True)
        assert counter.cpu().tolist() == [len(table) * cluster_m, 3]


def test_streamed_table():
    xs, idx, w, table, ref, _, _ = inputs(16)
    ready = torch.full((1,), len(table), dtype=torch.int32, device="cuda")
    out = torch.empty(sum(i.numel() for i in idx), w.shape[1], dtype=xs[0].dtype, device="cuda")
    launch(xs, idx, w, table, out, gather_work_table_ready=ready, sm_count=8)
    torch.cuda.synchronize()
    host = table.cpu().pin_memory()
    producer = torch.cuda.Stream()
    for _ in range(3):
        out.fill_(float("nan"))
        table.fill_(-1)
        ready.zero_()
        producer.wait_stream(torch.cuda.current_stream())
        # Launch first: scheduler tile zero must wait without reading the poison.
        launch(xs, idx, w, table, out, gather_work_table_ready=ready, sm_count=8)
        with torch.cuda.stream(producer):
            for start in range(0, len(table), 7):
                torch.cuda._sleep(200_000)
                end = min(start + 7, len(table))
                table[start:end].copy_(host[start:end], non_blocking=True)
                ready.fill_(end)
        torch.cuda.current_stream().wait_stream(producer)
        check(out[:len(ref)], ref)
        assert torch.isnan(out[len(ref):]).all().item()


def test_row_address_scatter():
    xs, idx, w, table, ref, _, _ = inputs(1)
    outputs = tuple(torch.zeros(rows, w.shape[1], device="cuda", dtype=xs[0].dtype)
                    for rows in (37, 53, 29))
    mapping = torch.tensor([[i % 3, (i // 3) % outputs[i % 3].shape[0]]
                            for i in range(idx[0].numel())], device="cuda", dtype=torch.int32)
    addresses = build_scatter_row_addresses(outputs, mapping, reduce=True)
    launch(xs, idx, w, table, outputs, epilogue_store="bulk_rows_reduce",
           multi_buffer_scatter=True, scatter_table=addresses)
    for j, out in enumerate(outputs):
        expected = torch.zeros_like(out, dtype=torch.float32)
        mask = mapping[:len(ref), 0] == j
        expected.index_add_(0, mapping[:len(ref)][mask, 1].long(), ref[mask].to(out.dtype).float())
        check(out, expected, reduce=True)


def test_mapped_host_ready():
    import ctypes
    from cuda.bindings import runtime as cudart

    xs, idx, w, table, ref, _, _ = inputs(1)
    out = torch.empty(idx[0].numel(), w.shape[1], dtype=xs[0].dtype, device="cuda")
    active = min(get_max_active_clusters(2, sm_count=10), len(table))
    prefix = torch.tensor(build_wave_ready_prefix(table.cpu().tolist(), group_size=1,
                          num_groups=3, wave_work_ids=active), device="cuda", dtype=torch.int32)
    counter = torch.zeros(2, device="cuda", dtype=torch.int32)
    err, host = cudart.cudaHostAlloc(4, cudart.cudaHostAllocMapped)
    assert err == cudart.cudaError_t.cudaSuccess
    try:
        err, device_ptr = cudart.cudaHostGetDevicePointer(host, 0)
        assert err == cudart.cudaError_t.cudaSuccess
        word = ctypes.c_int32.from_address(int(host))
        address = torch.tensor([int(device_ptr)], dtype=torch.int64, device="cuda")
        for _ in range(3):
            word.value = 0
            launch(xs, idx, w, table, out, sm_count=10, epilogue_store="bulk_rows",
                   wave_store_counter=counter, wave_ready_prefix=prefix,
                   wave_ready_host_address=address)
            torch.cuda.synchronize()
            assert word.value == 3
            check(out[:len(ref)], ref)
    finally:
        torch.cuda.synchronize()
        (err,) = cudart.cudaFreeHost(host)
        assert err == cudart.cudaError_t.cudaSuccess


@pytest.mark.parametrize("dynamic", [False, True])
def test_ordinary_gemm(dynamic):
    a = torch.randn(259, 144, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(384, 144, dtype=a.dtype, device=a.device) / 12
    out = torch.empty(259, 384, dtype=a.dtype, device=a.device)
    semaphore = torch.zeros(1, dtype=torch.int32, device=a.device) if dynamic else None
    gemm(a, b, out, None, semaphore, tile_M=128, tile_N=128, cluster_M=2, cluster_N=1,
         is_dynamic_persistent=dynamic)
    check(out, a.float() @ b.float().T)
