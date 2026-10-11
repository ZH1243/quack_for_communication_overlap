"""Hopper numerical regressions for pingpong wave-ordered bulk stores.

Run on Hopper: pytest tests/test_gemm_wave_store_barrier.py -x
"""

import math

import pytest
import torch

from quack.cute_dsl_utils import get_max_active_clusters
from quack.gemm import gemm
from run.hopper_gather_table_gemm import build_work_table


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="wave store barrier requires Hopper",
)


def launch(
    a, weights, output, table, counter, *, mode, cluster_m, scatter=None, max_swizzle_size=1
):
    indices = torch.arange(a.shape[0], device=a.device, dtype=torch.int32)
    return gemm(
        a, weights.transpose(1, 2), output,
        tile_M=128, tile_N=128, cluster_M=cluster_m, cluster_N=1,
        pingpong=True, persistent=True, is_dynamic_persistent=False,
        max_swizzle_size=max_swizzle_size, A_idx=indices, gather_work_table=table,
        epilogue_store=mode, scatter_table=scatter, wave_store_counter=counter,
    )


@torch.inference_mode()
@pytest.mark.parametrize("mode", ["bulk_rows", "bulk_rows_reduce"])
def test_wave_store_barrier_partial_wave(mode):
    """Odd final wave, a fully predicated CTA, and ordinary GEMM values."""
    torch.manual_seed(42)
    cluster_m, tile_m, k, n = 2, 128, 64, 136
    p = get_max_active_clusters(cluster_m)
    x = 2 if p % 2 == 0 else 1
    # 2P+1 M-clusters times two N tiles: four full waves and a partial wave.
    # The last M-cluster has one row, so its second CTA writes nothing but
    # must still contribute a completion report.
    routes = 2 * p * tile_m * cluster_m + 1
    a = torch.randn(routes, k, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(1, k, n, device="cuda", dtype=a.dtype) / math.sqrt(k)
    table, _, _ = build_work_table(
        [routes], output_dim=n, tile_m=tile_m, tile_n=128, cluster_m=cluster_m,
        max_swizzle_size=x, device=a.device, wave_work_ids=p,
    )
    output = torch.empty(routes, n, device=a.device, dtype=a.dtype)
    counter = torch.full((1,), 123456, device=a.device, dtype=torch.int32)
    reference = a.float() @ weights[0].float()
    baseline = (a @ weights[0]).float()
    atol = 2 * (baseline - reference).abs().max().item() + 1e-3
    for _ in range(2):
        output.zero_()
        launch(
            a, weights, output, table, counter, mode=mode, cluster_m=cluster_m,
            max_swizzle_size=x,
        )
        torch.testing.assert_close(output.float(), reference, atol=atol, rtol=1e-3)
        assert counter.item() == len(table) * x * cluster_m


@torch.inference_mode()
@pytest.mark.parametrize("mode", ["bulk_rows", "bulk_rows_reduce"])
@pytest.mark.parametrize("cluster_m", [1, 2])
def test_wave_store_barrier_order_and_graph_replay(mode, cluster_m):
    """Numerical output depends on cross-CTA store order, not just tile coverage.

    Three expert waves write the same destination rows, with no overlapping
    destinations *within* a wave. Rotate destination ownership across clusters
    so local pingpong handoff alone cannot order the writes. For reduction,
    FP32 rounding makes (+2**24, +1, -2**24) yield 0 in the required order;
    placing +1 last instead yields 1. Plain stores must retain the last wave.
    """
    tile_m, k, n = 128, 64, 128
    p = get_max_active_clusters(cluster_m)
    rows_per_cluster = tile_m * cluster_m
    rows_per_wave = p * rows_per_cluster
    routes = 3 * rows_per_wave
    a = torch.zeros(routes, k, device="cuda", dtype=torch.bfloat16)
    a[:, 0] = 1
    weights = torch.zeros(3, k, n, device=a.device, dtype=a.dtype)
    values = (2**24, 1, -(2**24)) if mode == "bulk_rows_reduce" else (1, 2, 3)
    for expert, value in enumerate(values):
        weights[expert, 0, :] = value
    table, _, _ = build_work_table(
        [rows_per_wave] * 3, output_dim=n, tile_m=tile_m, tile_n=128,
        cluster_m=cluster_m, max_swizzle_size=1, device=a.device, wave_work_ids=p,
    )
    local_rows = torch.arange(rows_per_wave, device=a.device, dtype=torch.int32)
    scatter = torch.cat([
        (local_rows + expert * rows_per_cluster) % rows_per_wave for expert in range(3)
    ])
    output = torch.empty(routes, n, device=a.device, dtype=torch.float32)
    counter = torch.empty(1, device=a.device, dtype=torch.int32)
    reference = torch.zeros_like(output)
    for expert in range(3):
        lo, hi = expert * rows_per_wave, (expert + 1) * rows_per_wave
        contribution = a[lo:hi].float() @ weights[expert].float()
        if mode == "bulk_rows_reduce":
            reference[scatter[lo:hi].long()] += contribution
        else:
            reference[scatter[lo:hi].long()] = contribution

    def run():
        output.zero_()
        launch(
            a, weights, output, table, counter, mode=mode,
            cluster_m=cluster_m, scatter=scatter,
        )

    run()  # Compile before capture.
    torch.testing.assert_close(output, reference, atol=0, rtol=0)
    assert counter.item() == 3 * p * cluster_m
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # Each captured invocation needs its own reset, not just each replay.
        run()
        run()
    for _ in range(3):
        counter.fill_(123456)
        graph.replay()
        torch.testing.assert_close(output, reference, atol=0, rtol=0)
        assert counter.item() == 3 * p * cluster_m
