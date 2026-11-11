"""Numerical regression for shared auxiliary stores without Hopper table scheduling."""

import math

import pytest
import torch
import torch.nn.functional as F

from quack.gemm_config import GemmConfig
from quack.gemm_interface import gemm_act


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11),
    reason="requires the SM100 GEMM implementation",
)


@pytest.mark.parametrize("tile_m,tile_n,cluster_m", [(128, 16, 1), (256, 256, 2)])
@torch.inference_mode()
def test_gelu_aux_store_without_gather_table(tile_m, tile_n, cluster_m):
    # The first config is the failing autotuning candidate. Pin it to exercise
    # TileStore.store_setup without searching or pruning away the failure.
    torch.manual_seed(0)
    m, n, k = 256, 256, 128
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) / math.sqrt(k)
    config = GemmConfig(
        tile_m=tile_m,
        tile_n=tile_n,
        cluster_m=cluster_m,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=True,
        swap_ab=False,
        device_capacity=torch.cuda.get_device_capability()[0],
    )
    preact, postact = gemm_act(
        a, b, activation="gelu_tanh_approx", out_dtype=a.dtype, tuned=False, config=config
    )
    reference = a.float() @ b.float()
    baseline = a @ b
    for output, ref, same_dtype in (
        (preact, reference, baseline),
        (postact, F.gelu(reference, approximate="tanh"), F.gelu(baseline, approximate="tanh")),
    ):
        baseline_error = (same_dtype.float() - ref).abs().max().item()
        torch.testing.assert_close(
            output.float(), ref, atol=max(1e-3, 2 * baseline_error), rtol=1e-2
        )
