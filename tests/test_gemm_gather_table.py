"""SM90 gather-table GEMM tests."""

import math
import time

import pytest
import torch

from quack.cute_dsl_utils import get_device_capacity
from quack.gemm import gemm as quack_gemm
from quack.gemm_config import GemmConfig
from quack.gemm_interface import gated_to_pytorch_fn_map, gemm_act


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_device_capacity(torch.device("cuda"))[0] != 9,
    reason="gather_work_table is SM90 only",
)


@torch.inference_mode()
@pytest.mark.parametrize(
    ("activation", "num_buffers", "pingpong"),
    [("relu", 1, False), ("relu", 1, True), ("swiglu", 3, True)],
)
def test_gather_table_activation_epilogue(activation, num_buffers, pingpong):
    """Activation TileStore follows table route offsets for single and multi-buffer gather."""
    torch.manual_seed(0)
    experts, tokens, routes, k, out_n = 2, 32, 17, 64, 128
    counts = (9, 8)
    gated = activation in gated_to_pytorch_fn_map
    gemm_n = out_n * (2 if gated else 1)
    x_buffers = tuple(
        torch.randn(tokens, k, dtype=torch.bfloat16, device="cuda")
        for _ in range(num_buffers)
    )
    idx_buffers = []
    for _ in range(num_buffers):
        idx_buffers.append(
            torch.cat(
                [
                    torch.randperm(tokens, dtype=torch.int32, device="cuda")[:count]
                    for count in counts
                ]
            )
        )
    idx_buffers = tuple(idx_buffers)
    weights = torch.randn(
        experts, gemm_n, k, dtype=torch.bfloat16, device="cuda"
    ).transpose(1, 2)
    weights.mul_(1 / math.sqrt(k))

    offsets = (0, counts[0], routes)
    if num_buffers == 1:
        rows = [(expert, offsets[expert], offsets[expert + 1], 0) for expert in range(experts)]
    else:
        rows = [
            (
                expert,
                0,
                *(
                    endpoint
                    for _ in x_buffers
                    for endpoint in (offsets[expert], offsets[expert + 1])
                ),
            )
            for expert in range(experts)
        ]
    work_table = torch.tensor(rows, dtype=torch.int32, device="cuda")
    ready_rows = (
        torch.tensor([len(rows)], dtype=torch.int32, device="cuda")
        if num_buffers > 1
        else None
    )
    output = torch.empty(num_buffers * routes, out_n, dtype=torch.bfloat16, device="cuda")
    config = GemmConfig(
        tile_m=64,
        tile_n=128,
        pingpong=pingpong,
        cluster_m=2,
        cluster_n=1,
        cluster_k=1,
        max_swizzle_size=8,
        device_capacity=9,
        use_tma_gather=False,
    )

    gemm_act(
        x_buffers if num_buffers > 1 else x_buffers[0],
        weights,
        activation=activation,
        postact_out=output,
        A_idx=idx_buffers if num_buffers > 1 else idx_buffers[0],
        gather_work_table=work_table,
        gather_work_table_ready=ready_rows,
        multi_buffer_gather=num_buffers > 1,
        store_preact=False,
        tuned=False,
        config=config,
        concat_layout=("B",) if gated else None,
    )

    reference_parts = []
    for expert, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        for X, A_idx in zip(x_buffers, idx_buffers):
            preact = X[A_idx[start:end].long()].float() @ weights[expert].float()
            if gated:
                gate, up = preact.chunk(2, dim=-1)
                preact = gated_to_pytorch_fn_map[activation](gate, up)
            else:
                preact = torch.relu(preact)
            reference_parts.append(preact)
    reference = torch.cat(reference_parts).to(output.dtype)
    torch.testing.assert_close(output, reference, atol=6e-2, rtol=2e-3)


@torch.inference_mode()
def test_gather_table_pingpong_runs_work_on_both_math_warpgroups():
    """More work than resident clusters makes WG1 execute real table descriptors."""
    torch.manual_seed(0)
    tokens, routes, k, n = 512, 512, 64, 1024
    tile_m, tile_n, rows_per_table_entry = 64, 128, 16
    X = torch.randn(tokens, k, dtype=torch.bfloat16, device="cuda")
    A_idx = torch.randperm(tokens, dtype=torch.int32, device="cuda")[:routes]
    weights = torch.randn(1, n, k, dtype=torch.bfloat16, device="cuda") / math.sqrt(k)
    work_table = torch.tensor(
        [
            (0, start, min(start + rows_per_table_entry, routes), 0)
            for start in range(0, routes, rows_per_table_entry)
        ],
        dtype=torch.int32,
        device="cuda",
    )
    output = torch.empty(routes, n, dtype=torch.bfloat16, device="cuda")

    quack_gemm(
        X,
        weights,
        output,
        C=None,
        tile_count_semaphore=None,
        tile_M=tile_m,
        tile_N=tile_n,
        cluster_M=1,
        cluster_N=1,
        cluster_K=1,
        pingpong=True,
        persistent=True,
        is_dynamic_persistent=False,
        max_swizzle_size=8,
        cu_seqlens_m=None,
        A_idx=A_idx,
        gather_work_table=work_table,
        use_tma_gather=False,
    )

    reference = X[A_idx.long()].float() @ weights[0].float().mT
    torch.testing.assert_close(output, reference.to(output.dtype), atol=3e-2, rtol=1e-3)


@torch.inference_mode()
@pytest.mark.parametrize(
    ("tile_m", "num_buffers", "pingpong"),
    [(64, 5, True), (128, 32, False), (192, 16, True)],
)
def test_multi_buffer_gather_selects_one_source_row(tile_m, num_buffers, pingpong):
    """Loader threads cooperatively prepare one address for each packed row."""
    torch.manual_seed(0)
    num_experts = 2
    tokens, routes, k, n = 32, 19, 64, 128
    counts = (10, 9)

    x_buffers = tuple(
        torch.randn(tokens, k, dtype=torch.bfloat16, device="cuda")
        for _ in range(num_buffers)
    )
    a_idx_buffers = tuple(
        torch.randperm(tokens, dtype=torch.int32, device="cuda")[:routes]
        for _ in range(num_buffers)
    )
    weights = torch.randn(
        num_experts, n, k, dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(k)

    # Build one work-table row per M cluster. This covers tile_M below, equal
    # to, and above the 128 address-preparation threads; CTA boundaries also
    # cross input-buffer boundaries and the final cluster is ragged.
    rows = []
    expert_offsets = (0, counts[0], routes)
    for expert in range(num_experts):
        start, end = expert_offsets[expert : expert + 2]
        count = end - start
        packed_count = num_buffers * count
        for packed_start in range(0, packed_count, 2 * tile_m):
            packed_end = min(packed_start + 2 * tile_m, packed_count)
            ranges = []
            for buffer_idx in range(num_buffers):
                buffer_start = buffer_idx * count
                local_start = min(max(packed_start - buffer_start, 0), count)
                local_end = min(max(packed_end - buffer_start, 0), count)
                ranges.extend((start + local_start, start + local_end))
            rows.append((expert, 0, *ranges))
    work_table = torch.tensor(rows, dtype=torch.int32, device="cuda")

    output = torch.empty(num_buffers * routes, n, dtype=torch.bfloat16, device="cuda")
    quack_gemm(
        x_buffers,
        weights,
        output,
        C=None,
        tile_count_semaphore=None,
        tile_M=tile_m,
        tile_N=128,
        cluster_M=2,
        cluster_N=1,
        cluster_K=1,
        pingpong=pingpong,
        persistent=True,
        is_dynamic_persistent=False,
        cu_seqlens_m=None,
        A_idx=a_idx_buffers,
        gather_work_table=work_table,
        multi_buffer_gather=True,
        use_tma_gather=False,
    )

    reference = torch.cat(
        [
            x_buffers[buffer_idx][
                a_idx_buffers[buffer_idx][start:end].long()
            ].float()
            @ weights[expert].float().mT
            for expert, (start, end) in enumerate(zip(expert_offsets[:-1], expert_offsets[1:]))
            for buffer_idx in range(num_buffers)
        ]
    ).to(output.dtype)
    torch.testing.assert_close(output, reference, atol=3e-2, rtol=1e-3)


@torch.inference_mode()
def test_multi_buffer_gather_waits_for_table_ready_prefix():
    """A copy-engine flag publication releases a scheduler initially polling HBM."""
    torch.manual_seed(0)
    tokens, routes, k, n = 32, 17, 64, 128
    x_buffers = tuple(
        torch.randn(tokens, k, dtype=torch.bfloat16, device="cuda") for _ in range(2)
    )
    a_idx_buffers = tuple(
        torch.randperm(tokens, dtype=torch.int32, device="cuda")[:routes] for _ in range(2)
    )
    weights = torch.randn(1, n, k, dtype=torch.bfloat16, device="cuda") / math.sqrt(k)
    work_table = torch.tensor(
        [[0, 0, 0, routes, 0, routes]], dtype=torch.int32, device="cuda"
    )
    ready_rows = torch.zeros(1, dtype=torch.int32, device="cuda")
    host_ready = torch.tensor([1], dtype=torch.int32, pin_memory=True)
    output = torch.empty(2 * routes, n, dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()

    quack_gemm(
        x_buffers,
        weights,
        output,
        C=None,
        tile_count_semaphore=None,
        tile_M=64,
        tile_N=128,
        cluster_M=2,
        cluster_N=1,
        cluster_K=1,
        pingpong=True,
        persistent=True,
        is_dynamic_persistent=False,
        cu_seqlens_m=None,
        A_idx=a_idx_buffers,
        gather_work_table=work_table,
        gather_work_table_ready=ready_rows,
        multi_buffer_gather=True,
        use_tma_gather=False,
    )
    time.sleep(0.01)
    copy_stream = torch.cuda.Stream()
    with torch.cuda.stream(copy_stream):
        ready_rows.copy_(host_ready, non_blocking=True)
    torch.cuda.synchronize()

    reference = torch.cat(
        [
            x[a_idx.long()].float() @ weights[0].float().mT
            for x, a_idx in zip(x_buffers, a_idx_buffers)
        ]
    ).to(output.dtype)
    torch.testing.assert_close(output, reference, atol=3e-2, rtol=1e-3)


@torch.inference_mode()
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize(
    ("activation", "cluster_m", "pingpong"),
    [(None, 1, False), (None, 2, True), ("relu", 2, False), ("swiglu", 2, True)],
)
def test_indexed_gather_reads_table_tokens(activation, cluster_m, pingpong, dtype):
    """Discrete/repeated tokens, N bundles, K tails, and partially empty M clusters."""
    from run.hopper_gather_table_gemm import (
        check_correctness,
        check_down_correctness,
        make_arg_parser,
        make_launch,
        prepare_inputs,
        validate_args,
    )

    args = make_arg_parser(include_down_projection=True).parse_args([])
    args.indexed_gather = True
    args.tokens, args.routes, args.experts = 97, 421, 3
    args.hidden, args.output_dim = 80, 768
    args.tile_m, args.tile_n, args.cluster_m = 64, 128, cluster_m
    args.max_swizzle_size = 2  # Multiple N groups per M cluster.
    args.pingpong, args.activation, args.dtype = pingpong, activation, dtype
    args.down_projection = activation is not None
    args.routing_with_replacement = True
    validate_args(args)
    torch.manual_seed(42)
    inputs = prepare_inputs(args, torch.device("cuda"))
    reference_indices = inputs.A_idx.clone()
    # A_idx must not be used for loading X in this mode. Poison it with an
    # out-of-range value; restore the independent reference after the launch.
    inputs.A_idx.fill_(args.tokens + 1000)
    make_launch(args, inputs)()
    torch.cuda.synchronize()
    inputs.A_idx = reference_indices
    check_correctness(inputs, activation=activation, atol=3e-2, rtol=1e-3)
    assert inputs.up_output.shape == (args.routes, args.output_dim)
    assert inputs.route_offsets[0][-1] == args.routes
    if args.down_projection:
        check_down_correctness(inputs, atol=3e-2, rtol=1e-3)


@pytest.mark.parametrize("proxy_mode", ["thread", "process"])
@pytest.mark.parametrize(
    ("routes", "extra_args"),
    [(2, []), (773, ["--pingpong"]),
     (773, ["--pingpong", "--activation", "swiglu", "--down-projection"])],
)
def test_stream_indexed_gather_proxy(proxy_mode, routes, extra_args):
    """Exercise proxy-built indexed rows, readiness waits, and numerical references."""
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    artifact = root / "run/cpu_proxy/build" / (
        "libstream_gather_proxy_thread.so" if proxy_mode == "thread" else "stream_gather_proxy"
    )
    if not artifact.is_file():
        pytest.skip("build run/cpu_proxy on Hopper before running proxy integration tests")
    result = subprocess.run(
        [
            sys.executable, str(root / "run/hopper_stream_gather_table_gemm.py"),
            "--indexed-gather", "--proxy-mode", proxy_mode,
            "--tokens", "512", "--routes", str(routes), "--experts", "3",
            "--hidden", "64", "--output-dim", "512",
            "--tile-m", "128", "--tile-n", "128", "--cluster-m", "2",
            "--max-swizzle-size", "2", "--warmup", "1", "--iterations", "2",
            "--timing-samples", "1", "--flush-entries", "1", "--flush-interval-us", "10",
            *extra_args,
        ],
        cwd=root, capture_output=True, text=True, timeout=300,
    )
    # The runner compares GEMM values against float32 PyTorch references;
    # the SwiGLU case additionally verifies the packed grouped down projection.
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Reference check: PASSED" in result.stdout
