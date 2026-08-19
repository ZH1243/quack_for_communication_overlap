#!/usr/bin/env python3
"""Benchmark Hopper grouped GEMM with fused gather-A and an HBM work table.

The GEMM receives no ``cu_seqlens_m``. Instead, each contiguous int32 table row
contains

    (expert_id, route_start, route_end, cid_n_base)

and expands to ``x = min(max_swizzle_size, ceil(N / tile_N))`` consecutive
AlongN work IDs. The runner requires ``ceil(N / tile_N) % x == 0``.

Tensor shapes:

    X:                 [T, K]
    W:                 [E, K, N]
    A_idx:             [R]
    gather_work_table: [Q, 4]
    output:            [R, N]

Example:

    python run/hopper_gather_table_gemm.py \
        --tokens 4096 --hidden 4096 --output-dim 4096 \
        --experts 8 --routes 8195 --warmup 5 --iterations 100
"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass

import torch


DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


@dataclass
class TableGatherInputs:
    X: torch.Tensor
    W: torch.Tensor
    A_idx: torch.Tensor
    work_table: torch.Tensor
    route_offsets: tuple[int, ...]
    output: torch.Tensor
    work_group_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark QuACK's SM90 table-scheduled grouped GEMM with gather-A."
    )
    parser.add_argument("--tokens", "-T", type=int, default=4096)
    parser.add_argument("--hidden", "-K", type=int, default=4096)
    parser.add_argument("--output-dim", "-N", type=int, default=4096)
    parser.add_argument("--experts", "-E", type=int, default=8)
    parser.add_argument("--routes", "-R", type=int, default=8195)
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tile-m", type=int, default=256)
    parser.add_argument("--tile-n", type=int, default=256)
    parser.add_argument("--tile-k", type=int, default=None)
    parser.add_argument("--cluster-m", type=int, default=2)
    parser.add_argument("--max-swizzle-size", type=int, default=8)
    parser.add_argument("--routing-with-replacement", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--timing-samples", type=int, default=5)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args()


def route_counts(total_routes: int, experts: int) -> list[int]:
    """Near-even counts; a nonzero remainder exercises true varlen-M."""
    base, remainder = divmod(total_routes, experts)
    return [base + int(expert < remainder) for expert in range(experts)]


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "tokens",
        "hidden",
        "output_dim",
        "experts",
        "routes",
        "tile_m",
        "tile_n",
        "cluster_m",
        "max_swizzle_size",
        "iterations",
        "timing_samples",
    ):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if args.warmup < 0:
        raise ValueError(f"warmup must be nonnegative, got {args.warmup}")
    counts = route_counts(args.routes, args.experts)
    if not args.routing_with_replacement and max(counts) > args.tokens:
        raise ValueError(
            f"an expert receives {max(counts)} routes but only {args.tokens} tokens exist; "
            "pass --routing-with-replacement if intended"
        )
    clusters_n = math.ceil(args.output_dim / args.tile_n)
    x = min(args.max_swizzle_size, clusters_n)
    if clusters_n % x:
        raise ValueError(
            f"table expansion requires clusters_n % x == 0, got {clusters_n} % {x}"
        )


def build_work_table(
    counts: list[int],
    *,
    output_dim: int,
    tile_m: int,
    tile_n: int,
    cluster_m: int,
    max_swizzle_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, ...], int]:
    """Encode the current AlongN group/serpentine order into table rows."""
    route_offsets = [0]
    for count in counts:
        route_offsets.append(route_offsets[-1] + count)

    clusters_n = math.ceil(output_dim / tile_n)
    x = min(max_swizzle_size, clusters_n)
    assert clusters_n % x == 0
    cluster_rows = tile_m * cluster_m
    rows: list[tuple[int, int, int, int]] = []

    for expert, count in enumerate(counts):
        expert_start, expert_end = route_offsets[expert : expert + 2]
        clusters_m = math.ceil(count / cluster_rows)
        for n_group in range(clusters_n // x):
            m_clusters = range(clusters_m)
            if n_group % 2:
                m_clusters = reversed(range(clusters_m))
            cid_n_base = n_group * x
            for cid_m in m_clusters:
                start = expert_start + cid_m * cluster_rows
                end = min(start + cluster_rows, expert_end)
                rows.append((expert, start, end, cid_n_base))

    table = torch.tensor(rows, dtype=torch.int32, device=device)
    return table, tuple(route_offsets), x


@torch.inference_mode()
def prepare_inputs(args: argparse.Namespace, device: torch.device) -> TableGatherInputs:
    dtype = DTYPES[args.dtype]
    counts = route_counts(args.routes, args.experts)
    X = torch.randn((args.tokens, args.hidden), dtype=dtype, device=device)
    W = torch.randn((args.experts, args.hidden, args.output_dim), dtype=dtype, device=device)
    W.mul_(1.0 / math.sqrt(args.hidden))

    A_idx = torch.empty(args.routes, dtype=torch.int32, device=device)
    offset = 0
    for count in counts:
        if args.routing_with_replacement:
            indices = torch.randint(args.tokens, (count,), dtype=torch.int32, device=device)
        else:
            indices = torch.randperm(args.tokens, dtype=torch.int32, device=device)[:count]
        A_idx[offset : offset + count].copy_(indices)
        offset += count

    work_table, offsets, x = build_work_table(
        counts,
        output_dim=args.output_dim,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        cluster_m=args.cluster_m,
        max_swizzle_size=args.max_swizzle_size,
        device=device,
    )
    output = torch.empty((args.routes, args.output_dim), dtype=dtype, device=device)
    return TableGatherInputs(X, W, A_idx, work_table, offsets, output, x)


def make_launch(args: argparse.Namespace, inputs: TableGatherInputs):
    from quack.gemm import gemm as quack_gemm

    B = inputs.W.transpose(1, 2)  # QuACK convention: [E, N, K].

    def launch() -> None:
        quack_gemm(
            inputs.X,
            B,
            inputs.output,
            C=None,
            tile_count_semaphore=None,
            tile_M=args.tile_m,
            tile_N=args.tile_n,
            tile_K=args.tile_k,
            cluster_M=args.cluster_m,
            cluster_N=1,
            cluster_K=1,
            pingpong=False,
            persistent=True,
            is_dynamic_persistent=False,
            max_swizzle_size=args.max_swizzle_size,
            cu_seqlens_m=None,
            A_idx=inputs.A_idx,
            gather_work_table=inputs.work_table,
            use_tma_gather=False,
        )

    return launch


@torch.inference_mode()
def benchmark(launch, *, iterations: int, samples: int, use_cuda_graph: bool) -> list[float]:
    graph = None
    if use_cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(iterations):
                launch()
        graph.replay()
        torch.cuda.synchronize()

    timings_ms = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if graph is not None:
            graph.replay()
        else:
            for _ in range(iterations):
                launch()
        end.record()
        end.synchronize()
        timings_ms.append(start.elapsed_time(end) / iterations)
    return timings_ms


@torch.inference_mode()
def check_correctness(inputs: TableGatherInputs, *, atol: float, rtol: float) -> float:
    max_abs_error = 0.0
    for expert, (start, end) in enumerate(
        zip(inputs.route_offsets[:-1], inputs.route_offsets[1:])
    ):
        if start == end:
            continue
        reference = inputs.X[inputs.A_idx[start:end]].float() @ inputs.W[expert].float()
        reference = reference.to(inputs.output.dtype)
        actual = inputs.output[start:end]
        max_abs_error = max(max_abs_error, (actual - reference).abs().max().item())
        torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
    return max_abs_error


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 9:
        raise RuntimeError(f"this runner requires Hopper SM90, got capability {capability}")

    torch.manual_seed(args.seed)
    inputs = prepare_inputs(args, device)
    torch.cuda.synchronize(device)
    counts = [b - a for a, b in zip(inputs.route_offsets[:-1], inputs.route_offsets[1:])]
    print(f"Device: {torch.cuda.get_device_name(device)} (SM{capability[0]}{capability[1]})")
    print(f"X: {tuple(inputs.X.shape)}, W: {tuple(inputs.W.shape)}, dtype: {inputs.X.dtype}")
    print(f"Routes per expert: {counts}")
    print(
        f"Work table: {tuple(inputs.work_table.shape)}, x={inputs.work_group_size}, "
        f"expanded work IDs={inputs.work_table.shape[0] * inputs.work_group_size}"
    )
    print(
        f"Kernel: tile=({args.tile_m}, {args.tile_n}, {args.tile_k or 'auto'}), "
        f"cluster=({args.cluster_m}, 1, 1), static persistent, table gather=cp.async"
    )
    print("Compiling and warming up the specialized QuACK kernel...")

    launch = make_launch(args, inputs)
    for _ in range(max(1, args.warmup)):
        launch()
    torch.cuda.synchronize(device)

    timings_ms = benchmark(
        launch,
        iterations=args.iterations,
        samples=args.timing_samples,
        use_cuda_graph=not args.no_cuda_graph,
    )
    median_ms = statistics.median(timings_ms)
    tflops = 2 * args.routes * args.hidden * args.output_dim / (median_ms * 1e9)
    print(f"Per-kernel samples (ms): {[round(value, 4) for value in timings_ms]}")
    print(f"Median kernel time: {median_ms:.4f} ms")
    print(f"Effective throughput: {tflops:.2f} TFLOP/s")

    if not args.skip_check:
        max_abs_error = check_correctness(inputs, atol=args.atol, rtol=args.rtol)
        print(f"Reference check: PASSED (max absolute error {max_abs_error:.6g})")


if __name__ == "__main__":
    main()
