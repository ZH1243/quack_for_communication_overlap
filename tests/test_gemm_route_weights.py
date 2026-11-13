"""Routed FP32 scales before output conversion on Hopper and Blackwell.

Remote: CUDA_VISIBLE_DEVICES=0 pytest -q tests/test_gemm_route_weights.py -x
"""

import math

import pytest
import torch

from quack.gemm import build_scatter_row_addresses, gemm, _gemm_plan_cache
from quack.gemm_config import BlackwellBulkStoreConfig
from run.hopper_gather_table_gemm import build_work_table


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10, 11),
    reason="route-weight gather epilogue requires Hopper or Blackwell",
)


def config(a, counts, n, *, wide=False):
    arch = torch.cuda.get_device_capability()[0]
    blackwell = arch in (10, 11)
    tile_m = 256 if blackwell else 128
    tile_n = 512 if blackwell and wide else 128
    table, _, _ = build_work_table(
        counts, output_dim=n, tile_m=tile_m, tile_n=tile_n,
        cluster_m=2, max_swizzle_size=1, device=a.device, device_capacity=arch,
    )
    kwargs = dict(
        C=None, tile_count_semaphore=None, tile_M=tile_m, tile_N=tile_n,
        cluster_M=2, cluster_N=1, persistent=True, pingpong=not blackwell,
        max_swizzle_size=1, A_idx=torch.arange(a.shape[0], device=a.device, dtype=torch.int32),
        gather_work_table=table,
    )
    if blackwell:
        kwargs["bulk_store_config"] = BlackwellBulkStoreConfig(
            padding_bytes=16, max_cols=128, buffer_count=1, num_warps=4,
        )
    return kwargs


@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("wide", [False, True], ids=["small", "wide_buffered"])
def test_route_weights_compact_reduction_cache_and_replay(dtype, wide):
    """Partial clusters, an empty expert, duplicate destinations and N groups.

    Exactly representable contributions make all-one and zero cases independent
    of reduction arrival order. Changing both values and tensor pointers checks
    that the plan/object caches specialize on metadata, never runtime weights.
    """
    counts, k, n = [0, 1, 129, 259], 64, 1024 if wide else 264
    routes = sum(counts)
    a = torch.zeros(routes, k, device="cuda", dtype=dtype)
    a[:, 0] = (torch.arange(routes, device=a.device) % 3 + 1).to(dtype)
    matrices = torch.zeros(len(counts), k, n, device=a.device, dtype=dtype)
    for expert in range(len(counts)):
        matrices[expert, 0] = (expert + 1) / 8
    kwargs = config(a, counts, n, wide=wide)
    tile_n = kwargs["tile_N"]
    # Two allocations model source GPUs. Each original token is hit by several
    # expert/source-list positions whose weights must remain distinct.
    row_counts = (73, 79)
    outputs = tuple(
        torch.empty(math.ceil(n / tile_n), rows * tile_n, device=a.device, dtype=dtype)
        for rows in row_counts
    )
    route = torch.arange(routes, device=a.device)
    buffer_id = route % 2
    row_id = (route // 2) % torch.tensor(row_counts, device=a.device)[buffer_id]
    mapping = torch.stack((buffer_id, row_id), dim=1)
    addresses = build_scatter_row_addresses(
        outputs, mapping, reduce=True, tile_n=tile_n, used_rows=row_counts,
    )
    kwargs.update(
        epilogue_store="bulk_rows_reduce", scatter_table=addresses,
        multi_buffer_scatter=True, n_tile_major_output=True,
        wave_store_counter=torch.empty(1, device=a.device, dtype=torch.int32),
    )
    original_values = torch.cat([
        (a[sum(counts[:e]):sum(counts[:e + 1])].float() @ matrices[e].float())
        for e in range(len(counts))
    ])
    destinations = row_id + buffer_id * row_counts[0]

    def collect():
        return torch.cat([
            out.view(-1, rows, tile_n).permute(1, 0, 2).reshape(rows, -1)[:, :n]
            for out, rows in zip(outputs, row_counts)
        ])

    def launch(scales):
        for output in outputs:
            output.zero_()
        return gemm(a, matrices.transpose(1, 2), outputs, route_weights=scales, **kwargs)

    launch(None)
    unweighted = collect().clone()
    ones = torch.ones(routes, device=a.device, dtype=torch.float32)
    launch(ones)
    torch.testing.assert_close(collect(), unweighted, atol=0, rtol=0)
    cache_entries = len(_gemm_plan_cache)
    scales = ((route % 8).float() / 8).contiguous()  # includes exact zero routes
    launch(scales)
    assert len(_gemm_plan_cache) == cache_entries
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(scales)
    for iteration in range(3):
        scales.copy_(((route + iteration) % 8).float() / 8)
        graph.replay()
        expected = torch.zeros(sum(row_counts), n, device=a.device, dtype=torch.float32)
        expected.index_add_(0, destinations, (original_values * scales[:, None]).to(dtype).float())
        torch.testing.assert_close(collect().float(), expected, atol=0, rtol=0)
        assert kwargs["wave_store_counter"].item() == len(kwargs["gather_work_table"]) * 2
    # An all-zero launch must still complete every CTA, including partial CTAs.
    launch(torch.zeros_like(scales))
    torch.testing.assert_close(collect(), torch.zeros_like(collect()), atol=0, rtol=0)
    assert kwargs["wave_store_counter"].item() == len(kwargs["gather_work_table"]) * 2
    # Check validation on the warm cache path, where tensor_key omits device.
    with pytest.raises(ValueError, match="input device"):
        launch(ones.cpu())
    for malformed in (ones[:-1], ones.to(dtype), torch.ones(routes * 2, device=a.device)[::2]):
        with pytest.raises(ValueError, match="contiguous CUDA FP32"):
            launch(malformed)


@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_route_weights_precede_output_conversion(dtype):
    """A halfway accumulator distinguishes scaling before vs after conversion."""
    rows, k, n = 259, 64, 264
    a = torch.zeros(rows, k, device="cuda", dtype=dtype)
    a[:, 0] = 1
    a[:, 1] = torch.finfo(dtype).eps / 2
    matrices = torch.zeros(1, k, n, device=a.device, dtype=dtype)
    matrices[:, :2, :] = 1
    scales = torch.full((rows,), 1.5, device=a.device, dtype=torch.float32)
    scales[::7] = 0
    out = torch.empty(rows, n, device=a.device, dtype=dtype)
    kwargs = config(a, [rows], n)
    kwargs["epilogue_store"] = "bulk_rows"
    gemm(a, matrices.transpose(1, 2), out, route_weights=scales, **kwargs)
    accumulator = a.float() @ matrices[0].float()
    expected = (accumulator * scales[:, None]).to(dtype)
    incorrectly_rounded_first = (accumulator.to(dtype).float() * scales[:, None]).to(dtype)
    assert not torch.equal(expected, incorrectly_rounded_first)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
