"""Numerical tests for the shared RDMA proxy up/down launch path on B300.

Run on a CUDA node with the normal object cache enabled. These exercise the
Python helpers used by the embedded proxy; the native two-node transport still
requires RDMA_CPU_Proxy's torchrun entry points.
"""

import argparse
import ctypes
import math

import pytest
import torch
import torch.nn.functional as F
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda, runtime as cudart

from quack.gemm import gemm
from quack.gemm_config import BlackwellBulkStoreConfig, GemmConfig, cta_tile_shape_m
from quack.gemm_interface import gemm_act
from quack.gemm_sm100 import GemmSm100

import hopper_pregather_gemm as pregather
from test_blackwell_gemm_communication import check, inputs, _observe_ready


@pytest.fixture(autouse=True)
def blackwell_device():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires B200/B300")
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False


@pytest.mark.parametrize("tile_m,tile_n,down_cluster_m", [(128, 128, 1), (128, 128, 2), (256, 512, 2)])
@pytest.mark.parametrize("tuned_bulk", [False, True])
@pytest.mark.parametrize("route_weight_policy", [None, "distinct", "zeros", "ones"])
def test_proxy_up_down_composition(tile_m, tile_n, down_cluster_m, tuned_bulk, route_weight_policy):
    """Ragged multi-buffer SwiGLU -> compact scatter-add -> concurrent acquire."""
    xs, idx, wu, table, pre, packed, _ = inputs(
        3, n=512, tile_m=tile_m, tile_n=128, cluster_m=2,
    )
    rows, hidden, out_n = len(pre), 256, 3 * tile_n
    up = torch.empty(sum(i.numel() for i in idx), hidden, device="cuda", dtype=xs[0].dtype)
    ready = torch.full((1,), len(table), device="cuda", dtype=torch.int32)
    config = GemmConfig(tile_m=tile_m, tile_n=128, cluster_m=2, cluster_n=1,
                        max_swizzle_size=1, device_capacity=10, pingpong=False,
                        is_dynamic_persistent=False, use_tma_gather=False)

    def launch_up():
        gemm_act(xs, wu.transpose(1, 2), activation="swiglu", A_idx=idx,
                 gather_work_table=table, gather_work_table_ready=ready,
                 multi_buffer_gather=True, postact_out=up, store_preact=False,
                 concat_layout=("B",), tuned=False, config=config)

    parser = argparse.ArgumentParser()
    pregather.add_kernel_arguments(parser)
    args = parser.parse_args([])
    args.device_capacity = 10
    args.tokens, args.routes, args.experts = 257, rows, len(packed)
    args.hidden, args.output_dim, args.activation = hidden, out_n, None
    args.tile_m, args.tile_n, args.cluster_m = tile_m, tile_n, down_cluster_m
    args.max_swizzle_size, args.sm_count = 1, 4
    args.gather_table = args.gather_table_expert_waves = True
    args.gather_table_wave_barrier = args.gather_table_publish_ready = True
    args.scatter_table = args.multi_buffer_scatter = args.n_tile_major_output = True
    args.scatter_table_with_replacement = True
    args.scatter_output_buffers, args.epilogue_store = 3, "bulk_rows_reduce"
    args.down_projection = False
    args.iterations, args.timing_samples, args.warmup = 3, 1, 1
    if tuned_bulk:
        args.bulk_smem_padding, args.bulk_smem_cols = 16, 128
        args.bulk_smem_buffers, args.bulk_store_warps = 1, 4
    pregather.validate_args(args, prepacked=True)
    wd = torch.randn(len(packed), hidden, out_n, device="cuda", dtype=up.dtype) / math.sqrt(hidden)
    used = (17, 23, 31)
    outputs = tuple(torch.empty(3, count * tile_n, device="cuda", dtype=up.dtype)
                    for count in used)
    mapping = torch.tensor([[i % 3, (i // 3) % used[i % 3]] for i in range(rows)],
                           device="cuda", dtype=torch.int32)
    route_weights = None
    if route_weight_policy is not None:
        route_weights = (torch.zeros(rows, device="cuda", dtype=torch.float32)
                         if route_weight_policy == "zeros" else
                         torch.ones(rows, device="cuda", dtype=torch.float32)
                         if route_weight_policy == "ones" else
                         0.03125 + torch.arange(rows, device="cuda", dtype=torch.float32).remainder(17) / 16)
    down = pregather.prepare_grouped_inputs(
        args, up[:rows], wd, [len(p) for p in packed],
        output_buffers=outputs, scatter_mapping=mapping, route_weights=route_weights,
    )
    cluster_rows = cta_tile_shape_m(tile_m, down_cluster_m, 10) * down_cluster_m
    assert all(end - start <= cluster_rows
               for _, start, end, _ in down.gather_work_table.cpu().tolist())
    expected_descriptors = sum(math.ceil(len(p) / cluster_rows) for p in packed) * 3
    assert len(down.gather_work_table) == expected_descriptors

    err, host = cudart.cudaHostAlloc(4, cudart.cudaHostAllocMapped)
    assert err == cudart.cudaError_t.cudaSuccess
    try:
        err, device_ptr = cudart.cudaHostGetDevicePointer(host, 0)
        assert err == cudart.cudaError_t.cudaSuccess
        host_ready = ctypes.c_int32.from_address(int(host))
        down.wave_ready_host_address = torch.tensor([int(device_ptr)], device="cuda",
                                                   dtype=torch.int64)
        launch_down = pregather.make_launch(args, down)
        host_ready.value = 0
        launch_up()
        down.zero_up_output()
        launch_down()
        torch.cuda.synchronize()
        check(up[:rows], F.silu(pre[:, :hidden]) * pre[:, hidden:])
        # Use the quantized actual intermediate as the down reference input.
        offsets = [0]
        for p in packed:
            offsets.append(offsets[-1] + len(p))
        ref_down = torch.cat([up[offsets[e]:offsets[e + 1]].float() @ wd[e].float()
                              for e in range(len(packed))])
        if route_weights is not None:
            # Scaling precedes contribution conversion and additive reduction.
            ref_down *= route_weights[:, None]
        expected = [torch.zeros(count, out_n, device="cuda") for count in used]
        for b in range(3):
            mask = mapping[:, 0] == b
            expected[b].index_add_(0, mapping[mask, 1].long(), ref_down[mask].to(up.dtype).float())

        snapshots = tuple(torch.empty_like(out) for out in outputs)
        observer_args = (tuple(from_dlpack(out) for out in outputs),
                         tuple(from_dlpack(out) for out in snapshots),
                         from_dlpack(down.wave_store_workspace))
        observer = cute.compile(_observe_ready, *observer_args,
                                cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        producer, consumer, compute = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
        host_inputs = tuple(x.cpu().pin_memory() for x in xs)
        host_table = table.cpu().pin_memory()
        chunks = [(start, min(start + 3, len(table))) for start in range(0, len(table), 3)]
        host_ready_values = [torch.tensor([end], dtype=torch.int32, pin_memory=True)
                             for _, end in chunks]

        def publish():
            # Input copies precede table publication on the SAME producer stream.
            # RDMA visibility is supplied separately by the C++ proxy on hardware.
            with torch.cuda.stream(producer):
                for x, host_x in zip(xs, host_inputs):
                    x.copy_(host_x, non_blocking=True)
                for (start, end), ready_value in zip(chunks, host_ready_values):
                    table[start:end].copy_(host_table[start:end], non_blocking=True)
                    # Copy-engine publication needs no spare SM for a producer
                    # fill/sleep kernel while the persistent up GEMM polls.
                    ready.copy_(ready_value, non_blocking=True)

        publish()
        producer.synchronize()
        observer(*observer_args, cuda.CUstream(consumer.cuda_stream))
        consumer.synchronize()
        for _ in range(3):
            for x in xs:
                x.zero_()
            up.fill_(float("nan"))
            table.fill_(-1)
            ready.zero_()
            down.wave_store_workspace.zero_()
            down.zero_up_output()
            for snap in snapshots:
                snap.fill_(float("nan"))
            host_ready.value = 0
            for stream in (producer, consumer, compute):
                stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(compute):
                launch_up()
                launch_down()
            observer(*observer_args, cuda.CUstream(consumer.cuda_stream))
            publish()
            for stream in (producer, consumer, compute):
                torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            assert host_ready.value == 3
            assert down.wave_store_workspace.cpu().tolist() == [
                expected_descriptors * down_cluster_m, 3,
            ]
            check(up[:rows], F.silu(pre[:, :hidden]) * pre[:, hidden:])
            for b, count in enumerate(used):
                logical = snapshots[b].view(3, count, tile_n).permute(1, 0, 2).reshape(count, out_n)
                check(logical, expected[b], reduce=True)
    finally:
        # The host pointer and all scatter allocations outlive both streams.
        torch.cuda.synchronize()
        (err,) = cudart.cudaFreeHost(host)
        assert err == cudart.cudaError_t.cudaSuccess


def test_bulk_store_config_cache():
    """Different per-call settings compile separately and retain numerical values."""
    a = torch.randn(259, 144, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1, 512, 144, device="cuda", dtype=a.dtype) / 12
    out = torch.empty(259, 512, device="cuda", dtype=a.dtype)
    reference = a.float() @ b[0].float().T
    defaults = (GemmSm100.bulk_store_smem_max_cols, GemmSm100.bulk_store_num_warps)
    configs = [BlackwellBulkStoreConfig(),
               BlackwellBulkStoreConfig(max_cols=128, num_warps=1),
               BlackwellBulkStoreConfig(max_cols=128, num_warps=4)]
    plans = []
    for cfg in [*configs, configs[0]]:
        out.fill_(float("nan"))
        plan = gemm(a, b[0], out, None, None, tile_M=256, tile_N=512,
                    cluster_M=2, cluster_N=1, pingpong=False,
                    is_dynamic_persistent=False, epilogue_store="bulk_rows",
                    bulk_store_config=cfg, sm_count=4)
        check(out, reference)
        plans.append(plan)
    assert len({id(p.compiled_fn) for p in plans[:3]}) == 3
    assert plans[0] is plans[3]
    assert defaults == (GemmSm100.bulk_store_smem_max_cols, GemmSm100.bulk_store_num_warps)
