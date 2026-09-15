"""Numerical coverage for Hopper's full-tile, per-row bulk-copy epilogue.

Run on Hopper: pytest tests/test_gemm_bulk_rows.py -x
"""

import math

import pytest
import torch

from quack.gemm import gemm


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="bulk_rows epilogue requires Hopper",
)


@torch.inference_mode()
@pytest.mark.parametrize("pingpong", [False, True], ids=["cooperative", "pingpong"])
@pytest.mark.parametrize("n", [128, 136], ids=["full_n", "tail_n"])
@pytest.mark.parametrize(
    ("dtype", "out_dtype"),
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.float32),
    ],
    ids=["bf16", "fp16", "fp32_output"],
)
def test_bulk_rows_gather(pingpong, n, dtype, out_dtype):
    """Ragged/empty experts, N tails, padded pitches, persistent reuse and graph replay.

    The large final expert has more tiles than resident CTAs on H100/H200,
    exercising buffer reuse as well as the handoff between pingpong warpgroups.
    Guard rows/columns catch stores outside the logical output, while comparison
    against PyTorch detects writes crossing from one expert into the next.
    """
    torch.manual_seed(42)
    counts = (0, 1, 129, 1025, 32769)
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    routes, tokens, k = offsets[-1], 257, 128
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    indices = torch.randint(tokens, (routes,), dtype=torch.int32, device="cuda")
    x = torch.randn(tokens, k, dtype=dtype, device="cuda")
    # Match the runner's physical [E, K, N] weights and logical [E, N, K] B.
    weights = torch.randn(len(counts), k, n, dtype=dtype, device="cuda") / math.sqrt(k)
    b = weights.transpose(1, 2)
    sentinel = -123.0
    storage = torch.full((routes + 2, n + 8), sentinel, dtype=out_dtype, device="cuda")
    output = storage[1:-1, :n]
    tma_output = torch.empty_like(output)
    config = dict(
        C=None,
        tile_count_semaphore=None,
        tile_M=128,
        tile_N=128,
        cluster_M=2,
        cluster_N=1,
        pingpong=pingpong,
        persistent=True,
        is_dynamic_persistent=False,
        cu_seqlens_m=cu_seqlens,
        A_idx=indices,
    )

    def launch(mode, out):
        gemm(x, b, out, epilogue_store=mode, **config)

    # Compile before capture. Calling the modes back-to-back also checks that
    # the host plan and JIT caches distinguish the store mode.
    launch("tma", tma_output)
    launch("bulk_rows", output)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch("bulk_rows", output)

    for replay in range(2):
        if replay:
            x.neg_()  # A new result must overwrite every row on the next replay.
        output.fill_(float("nan"))
        graph.replay()
        launch("tma", tma_output)
        torch.cuda.synchronize()

        # Stage depth and store layout must not change the GEMM arithmetic.
        torch.testing.assert_close(output, tma_output, atol=0, rtol=0)
        for expert, count in enumerate(counts):
            if not count:
                continue
            lo, hi = offsets[expert : expert + 2]
            gathered = x[indices[lo:hi].long()]
            reference = gathered.float() @ weights[expert].float()
            baseline = (gathered @ weights[expert]).float()
            actual = output[lo:hi].float()
            # Same-dtype matmul supplies the rounding baseline; compare values
            # directly to the float32 ground truth as well.
            baseline_error = (baseline - reference).abs().max().item()
            atol = 2 * baseline_error + (1e-5 if out_dtype == torch.float32 else 1e-3)
            torch.testing.assert_close(actual, reference, atol=atol, rtol=1e-3)
        torch.testing.assert_close(storage[0], torch.full_like(storage[0], sentinel))
        torch.testing.assert_close(storage[-1], torch.full_like(storage[-1], sentinel))
        torch.testing.assert_close(storage[1:-1, n:], torch.full_like(storage[1:-1, n:], sentinel))


@torch.inference_mode()
@pytest.mark.parametrize("tile_m,pingpong", [(256, False), (128, True)])
def test_bulk_rows_dense_linear_epilogue(tile_m, pingpong):
    """The full-tile store preserves C, alpha/beta, and batch addressing."""
    torch.manual_seed(7)
    dtype = torch.bfloat16
    l, m, n, k = 2, 517, 264, 128
    a = torch.randn(l, m, k, dtype=dtype, device="cuda")
    b = torch.randn(l, n, k, dtype=dtype, device="cuda") / math.sqrt(k)
    c = torch.randn(l, m, n, dtype=dtype, device="cuda")
    output = torch.empty_like(c)
    tma_output = torch.empty_like(c)
    config = dict(
        C=c,
        tile_count_semaphore=None,
        tile_M=tile_m,
        tile_N=128,
        cluster_M=2,
        cluster_N=1,
        pingpong=pingpong,
        alpha=0.5,
        beta=0.25,
    )
    for mode, out in (("tma", tma_output), ("bulk_rows", output)):
        gemm(a, b, out, epilogue_store=mode, **config)
    reference = 0.5 * (a.float() @ b.float().transpose(1, 2)) + 0.25 * c.float()
    baseline = (0.5 * (a @ b.transpose(1, 2)).float() + 0.25 * c.float()).to(dtype).float()
    torch.testing.assert_close(output, tma_output, atol=0, rtol=0)
    baseline_error = (baseline - reference).abs().max().item()
    torch.testing.assert_close(output.float(), reference, atol=2 * baseline_error + 1e-3, rtol=1e-3)

    # A different pointer with the same tensor metadata must still be checked
    # when it hits the warm plan cache. The successful result above also checks
    # that the mode's alignment validation accepts ordinary aligned allocations.
    backing = torch.empty(output.numel() + 1, dtype=dtype, device="cuda")
    unaligned = backing[1:].view_as(output)
    with pytest.raises(ValueError, match="16-byte-aligned output base"):
        gemm(a, b, unaligned, epilogue_store="bulk_rows", **config)
