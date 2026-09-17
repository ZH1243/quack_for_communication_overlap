"""Numerical coverage for Hopper's full-tile, per-row bulk-copy/reduction epilogue.

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
@pytest.mark.parametrize("routing", ["varlen", "table", "pregather_table"])
@pytest.mark.parametrize("pingpong", [False, True], ids=["cooperative", "pingpong"])
@pytest.mark.parametrize("mode", ["bulk_rows", "bulk_rows_reduce"])
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
def test_bulk_rows_gather(routing, pingpong, mode, n, dtype, out_dtype):
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
    if routing != "varlen":
        from run.hopper_gather_table_gemm import build_work_table

        table, _, _ = build_work_table(
            list(counts),
            output_dim=n,
            tile_m=128,
            tile_n=128,
            cluster_m=2,
            max_swizzle_size=8,
            device=x.device,
        )
        config.update(cu_seqlens_m=None, gather_work_table=table)
        if routing == "pregather_table":
            x = x[indices.long()]
            indices = torch.arange(routes, dtype=torch.int32, device=x.device)
            config["A_idx"] = indices

    def launch(mode, out):
        gemm(x, b, out, epilogue_store=mode, **config)

    # Compile before capture. Calling the modes back-to-back also checks that
    # the host plan and JIT caches distinguish the store mode.
    launch("tma", tma_output)
    if mode == "bulk_rows_reduce":
        output.zero_()
    launch(mode, output)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if mode == "bulk_rows_reduce":
            output.zero_()
        launch(mode, output)

    for replay in range(2):
        if replay:
            x.neg_()  # Each replay must produce the new result without accumulation.
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
@pytest.mark.parametrize("mode", ["bulk_rows", "bulk_rows_reduce"])
@pytest.mark.parametrize("scatter", [False, True], ids=["ordered", "scatter"])
def test_table_bulk_rows_output_view(mode, scatter):
    """Table offsets preserve plain D's N mode, row pitch, and partial-row bounds."""
    torch.manual_seed(7)
    routes, n, k = 3, 136, 128
    a = torch.randn(routes, k, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(2, k, n, device="cuda", dtype=a.dtype) / math.sqrt(k)
    indices = torch.arange(routes, device="cuda", dtype=torch.int32)
    table = torch.tensor([[0, 0, 1, 0], [1, 1, 3, 0]], device="cuda", dtype=torch.int32)
    sentinel = -123.0
    storage = torch.full((routes + 2, n + 8), sentinel, device="cuda", dtype=a.dtype)
    output = storage[1:-1, :n]
    # Nonzero initialization checks that the reduction path adds rather than copies.
    initial = torch.arange(1, routes + 1, device="cuda", dtype=output.dtype)[:, None]
    initial = initial.expand_as(output).contiguous() * 0.25
    mapping = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32) if scatter else None
    output.copy_(initial)
    gemm(
        a,
        weights.transpose(1, 2),
        output,
        C=None,
        tile_count_semaphore=None,
        tile_M=128,
        tile_N=128,
        cluster_M=2,
        cluster_N=1,
        persistent=True,
        is_dynamic_persistent=False,
        A_idx=indices,
        gather_work_table=table,
        epilogue_store=mode,
        scatter_table=mapping,
    )
    reference = torch.cat(
        (a[:1].float() @ weights[0].float(), a[1:].float() @ weights[1].float())
    )
    baseline = torch.cat((a[:1] @ weights[0], a[1:] @ weights[1])).float()
    actual = output if mapping is None else output[mapping.long()]
    initial = initial if mapping is None else initial[mapping.long()]
    if mode == "bulk_rows_reduce":
        reference = reference + initial.float()
        baseline = (baseline + initial.float()).to(output.dtype).float()
    baseline_error = (baseline - reference).abs().max().item()
    torch.testing.assert_close(actual.float(), reference, atol=2 * baseline_error + 1e-3, rtol=1e-3)
    torch.testing.assert_close(storage[0], torch.full_like(storage[0], sentinel))
    torch.testing.assert_close(storage[-1], torch.full_like(storage[-1], sentinel))
    torch.testing.assert_close(storage[1:-1, n:], torch.full_like(storage[1:-1, n:], sentinel))


@torch.inference_mode()
@pytest.mark.parametrize("mode", ["bulk_rows", "bulk_rows_reduce"])
@pytest.mark.parametrize("tile_m,pingpong", [(256, False), (128, True)])
def test_bulk_rows_dense_linear_epilogue(tile_m, pingpong, mode):
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
    output.zero_()
    for store_mode, out in (("tma", tma_output), (mode, output)):
        gemm(a, b, out, epilogue_store=store_mode, **config)
    reference = 0.5 * (a.float() @ b.float().transpose(1, 2)) + 0.25 * c.float()
    baseline = (0.5 * (a @ b.transpose(1, 2)).float() + 0.25 * c.float()).to(dtype).float()
    torch.testing.assert_close(output, tma_output, atol=0, rtol=0)
    baseline_error = (baseline - reference).abs().max().item()
    torch.testing.assert_close(output.float(), reference, atol=2 * baseline_error + 1e-3, rtol=1e-3)

    if mode == "bulk_rows_reduce":
        # A nonzero destination distinguishes a real reduction from a plain copy.
        initial = torch.randn_like(output)
        output.copy_(initial)
        gemm(a, b, output, epilogue_store=mode, **config)
        expected = (initial.float() + tma_output.float()).to(dtype)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        full_reference = initial.float() + reference
        full_baseline = (initial.float() + baseline).to(dtype).float()
        baseline_error = (full_baseline - full_reference).abs().max().item()
        torch.testing.assert_close(
            output.float(), full_reference, atol=2 * baseline_error + 1e-3, rtol=1e-3
        )

    # A different pointer with the same tensor metadata must still be checked
    # when it hits the warm plan cache. The successful result above also checks
    # that the mode's alignment validation accepts ordinary aligned allocations.
    backing = torch.empty(output.numel() + 1, dtype=dtype, device="cuda")
    unaligned = backing[1:].view_as(output)
    with pytest.raises(ValueError, match="16-byte-aligned output base"):
        gemm(a, b, unaligned, epilogue_store=mode, **config)


@torch.inference_mode()
@pytest.mark.parametrize("use_cuda_graph", [False, True], ids=["direct", "graph"])
def test_bulk_rows_reduce_benchmark(use_cuda_graph):
    """Every timed invocation/replay starts at zero and leaves a valid GEMM result."""
    from run.hopper_gather_gemm import benchmark

    torch.manual_seed(19)
    a = torch.randn(129, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(136, 128, device="cuda", dtype=torch.bfloat16) / math.sqrt(128)
    output = torch.empty(129, 136, device="cuda", dtype=torch.bfloat16)

    def launch():
        gemm(
            a,
            b,
            output,
            C=None,
            tile_count_semaphore=None,
            tile_M=128,
            tile_N=128,
            cluster_M=2,
            cluster_N=1,
            epilogue_store="bulk_rows_reduce",
        )

    output.zero_()
    launch()  # Compile outside capture/timing.
    torch.cuda.synchronize()
    for repeat in range(2):
        if repeat:
            a.neg_()
        output.fill_(float("nan"))
        timings = benchmark(
            launch,
            iterations=3,
            samples=2,
            use_cuda_graph=use_cuda_graph,
            reset_output=output.zero_,
        )
        reference = a.float() @ b.float().T
        baseline = (a @ b.T).float()
        baseline_error = (baseline - reference).abs().max().item()
        torch.testing.assert_close(
            output.float(), reference, atol=2 * baseline_error + 1e-3, rtol=1e-3
        )
        assert len(timings) == 2 and all(math.isfinite(t) and t > 0 for t in timings)


@pytest.mark.parametrize("use_cuda_graph", [False, True], ids=["direct", "graph"])
@pytest.mark.parametrize("runner_variant", ["gather", "pregather_scatter"])
def test_bulk_rows_reduce_runner_main(monkeypatch, use_cuda_graph, runner_variant):
    """Enter main outside inference mode, as the CLI does, and check its output."""
    import sys

    if runner_variant == "gather":
        from run import hopper_gather_gemm as runner
    else:
        from run import hopper_pregather_gemm as runner

    argv = (
        "hopper_gather_gemm.py --tokens 257 --hidden 128 --output-dim 136 "
        "--experts 2 --routes 258 --tile-m 128 --tile-n 128 "
        "--warmup 2 --iterations 2 --timing-samples 2 "
        "--epilogue-store bulk_rows_reduce --pingpong"
    ).split()
    if runner_variant == "pregather_scatter":
        argv.extend(("--gather-table", "--scatter-table"))
    if not use_cuda_graph:
        argv.append("--no-cuda-graph")
    monkeypatch.setattr(sys, "argv", argv)
    prepared = []
    prepare_inputs = runner.prepare_inputs

    def capture_inputs(args, device):
        inputs = prepare_inputs(args, device)
        prepared.append(inputs)
        return inputs

    monkeypatch.setattr(runner, "prepare_inputs", capture_inputs)
    # Do not decorate this test with inference_mode: that would hide the bug.
    with torch.inference_mode(False):
        runner.main()
    inputs = prepared[0]
    for expert in range(2):
        lo, hi = expert * 129, (expert + 1) * 129
        gathered = inputs.X[inputs.A_idx[lo:hi].long()]
        reference = gathered.float() @ inputs.W_up[expert].float()
        baseline = (gathered @ inputs.W_up[expert]).float()
        baseline_error = (baseline - reference).abs().max().item()
        actual = (
            inputs.output[lo:hi]
            if runner_variant == "gather"
            else inputs.output[inputs.scatter_table[lo:hi].long()]
        )
        torch.testing.assert_close(
            actual.float(), reference, atol=2 * baseline_error + 1e-3, rtol=1e-3
        )


@torch.inference_mode()
@pytest.mark.parametrize("routing", ["varlen", "table", "table_n_group_major"])
@pytest.mark.parametrize("pingpong", [False, True], ids=["cooperative", "pingpong"])
@pytest.mark.parametrize("mode", ["bulk_rows", "bulk_rows_reduce"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_bulk_rows_scatter(routing, pingpong, mode, dtype):
    """Global permutations, CTA offsets, tails, persistent reuse and runtime mappings.

    The second N group reverses M traversal. A padded output view catches
    accidental use of a shifted base or a contiguous pitch. Changing mapping
    pointers between launches and contents on graph replay catches stale plans.
    """
    torch.manual_seed(123)
    counts = (0, 1, 129, 1025, 32769)
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    routes, k, n = offsets[-1], 128, 392
    a = torch.randn(routes, k, device="cuda", dtype=dtype)
    weights = torch.randn(len(counts), k, n, device="cuda", dtype=dtype) / math.sqrt(k)
    cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32)
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
        max_swizzle_size=2,
        cu_seqlens_m=cu_seqlens,
    )
    if routing != "varlen":
        from run.hopper_gather_table_gemm import build_work_table

        table, _, _ = build_work_table(
            list(counts),
            output_dim=n,
            tile_m=128,
            tile_n=128,
            cluster_m=2,
            max_swizzle_size=2,
            device=a.device,
            n_group_major=routing == "table_n_group_major",
        )
        config.update(
            cu_seqlens_m=None,
            A_idx=torch.arange(routes, device="cuda", dtype=torch.int32),
            gather_work_table=table,
        )
    sentinel = -123.0
    storage = torch.full((routes + 2, n + 8), sentinel, device="cuda", dtype=dtype)
    output = storage[1:-1, :n]
    ordinary = torch.empty_like(output)
    reference = torch.cat(
        [
            a[lo:hi].float() @ weights[e].float()
            for e, (lo, hi) in enumerate(zip(offsets, offsets[1:]))
        ]
    )
    baseline = torch.cat(
        [a[lo:hi] @ weights[e] for e, (lo, hi) in enumerate(zip(offsets, offsets[1:]))]
    )
    atol = 2 * (baseline.float() - reference).abs().max().item() + 1e-3

    def launch(mapping):
        if mode == "bulk_rows_reduce":
            output.zero_()
        gemm(
            a,
            weights.transpose(1, 2),
            output,
            epilogue_store=mode,
            scatter_table=mapping,
            **config,
        )

    def check(mapping):
        actual = output[mapping.long()]
        torch.testing.assert_close(actual.float(), reference, atol=atol, rtol=1e-3)
        torch.testing.assert_close(actual, ordinary, atol=0, rtol=0)
        torch.testing.assert_close(storage[0], torch.full_like(storage[0], sentinel))
        torch.testing.assert_close(storage[-1], torch.full_like(storage[-1], sentinel))
        torch.testing.assert_close(storage[1:-1, n:], torch.full_like(storage[1:-1, n:], sentinel))

    # Exercise no-scatter and scatter specializations back-to-back.
    gemm(a, weights.transpose(1, 2), ordinary, epilogue_store="tma", **config)
    for mapping in (
        torch.arange(routes, device="cuda", dtype=torch.int32),
        torch.randperm(routes, device="cuda", dtype=torch.int32),
    ):
        output.fill_(float("nan"))
        launch(mapping)
        check(mapping)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(mapping)
    for _ in range(2):
        mapping.copy_(torch.randperm(routes, device="cuda", dtype=torch.int32))
        a.neg_()
        reference.neg_()
        output.fill_(float("nan"))
        graph.replay()
        gemm(a, weights.transpose(1, 2), ordinary, epilogue_store="tma", **config)
        check(mapping)
