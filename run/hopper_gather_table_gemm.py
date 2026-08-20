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

With ``--multi-buffer-gather``, the kernel reads separately allocated token
and route-index buffers without materializing their concatenation:

    X_j:               [tokens_per_buffer, K]
    A_idx_j:           [routes_per_buffer]
    gather_work_table: [Q, 2 + 2 * num_input_buffers]
    output:            [num_input_buffers * routes_per_buffer, N]

Multi-buffer table rows contain ``(expert_id, cid_n_base, start_0, end_0,
..., start_b-1, end_b-1)``. Routes are packed in expert-major order and, within
an expert, in increasing buffer order.

Example:

    python run/hopper_gather_table_gemm.py \
        --tokens 4096 --hidden 4096 --output-dim 4096 \
        --experts 8 --routes 8195 --warmup 5 --iterations 100

    python run/hopper_gather_table_gemm.py --multi-buffer-gather \
        --num-input-buffers 3 --tokens-per-buffer 4096 \
        --routes-per-buffer 8195 --hidden 4096 --output-dim 4096 --experts 8
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
    X: torch.Tensor | tuple[torch.Tensor, ...]
    W: torch.Tensor
    A_idx: torch.Tensor | tuple[torch.Tensor, ...]
    work_table: torch.Tensor
    route_offsets: tuple[tuple[int, ...], ...]
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
    parser.add_argument(
        "--multi-buffer-gather",
        action="store_true",
        help="Gather directly from multiple independent X/A_idx buffers",
    )
    parser.add_argument("--num-input-buffers", type=int, default=2)
    parser.add_argument(
        "--tokens-per-buffer",
        type=int,
        default=None,
        help="Rows in each X_j; defaults to --tokens",
    )
    parser.add_argument(
        "--routes-per-buffer",
        type=int,
        default=None,
        help="Routes in each A_idx_j; defaults to --routes",
    )
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


def multi_buffer_route_counts(
    routes_per_buffer: int, experts: int, num_buffers: int
) -> list[list[int]]:
    """Near-even counts with the remainder rotated across buffers."""
    base, remainder = divmod(routes_per_buffer, experts)
    return [
        [base + int((expert - buffer_idx) % experts < remainder) for expert in range(experts)]
        for buffer_idx in range(num_buffers)
    ]


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
    tokens = args.tokens_per_buffer if args.multi_buffer_gather else args.tokens
    routes = args.routes_per_buffer if args.multi_buffer_gather else args.routes
    if tokens is None:
        tokens = args.tokens
    if routes is None:
        routes = args.routes
    if tokens <= 0 or routes <= 0:
        raise ValueError("tokens-per-buffer and routes-per-buffer must be positive")
    if args.multi_buffer_gather and args.num_input_buffers < 2:
        raise ValueError("num-input-buffers must be at least 2 in multi-buffer mode")
    counts = (
        multi_buffer_route_counts(routes, args.experts, args.num_input_buffers)
        if args.multi_buffer_gather
        else [route_counts(routes, args.experts)]
    )
    if not args.routing_with_replacement and max(max(c) for c in counts) > tokens:
        raise ValueError(
            "an expert receives more routes than tokens in one buffer; "
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
) -> tuple[torch.Tensor, tuple[tuple[int, ...], ...], int]:
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
    return table, (tuple(route_offsets),), x


def build_multi_buffer_work_table(
    counts_by_buffer: list[list[int]],
    *,
    output_dim: int,
    tile_m: int,
    tile_n: int,
    cluster_m: int,
    max_swizzle_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[tuple[int, ...], ...], int]:
    """Build expert-major cluster rows spanning independent route buffers."""
    num_buffers = len(counts_by_buffer)
    experts = len(counts_by_buffer[0])
    route_offsets: list[list[int]] = []
    for counts in counts_by_buffer:
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        route_offsets.append(offsets)

    clusters_n = math.ceil(output_dim / tile_n)
    x = min(max_swizzle_size, clusters_n)
    assert clusters_n % x == 0
    cluster_rows = tile_m * cluster_m
    rows: list[tuple[int, ...]] = []

    for expert in range(experts):
        expert_counts = [counts_by_buffer[j][expert] for j in range(num_buffers)]
        packed_offsets = [0]
        for count in expert_counts:
            packed_offsets.append(packed_offsets[-1] + count)
        clusters_m = math.ceil(packed_offsets[-1] / cluster_rows)
        for n_group in range(clusters_n // x):
            m_clusters = range(clusters_m)
            if n_group % 2:
                m_clusters = reversed(range(clusters_m))
            cid_n_base = n_group * x
            for cid_m in m_clusters:
                packed_start = cid_m * cluster_rows
                packed_end = min(packed_start + cluster_rows, packed_offsets[-1])
                ranges: list[int] = []
                for j in range(num_buffers):
                    count = expert_counts[j]
                    local_start = min(max(packed_start - packed_offsets[j], 0), count)
                    local_end = min(max(packed_end - packed_offsets[j], 0), count)
                    ranges.extend(
                        (
                            route_offsets[j][expert] + local_start,
                            route_offsets[j][expert] + local_end,
                        )
                    )
                rows.append((expert, cid_n_base, *ranges))

    table = torch.tensor(rows, dtype=torch.int32, device=device)
    return table, tuple(tuple(offsets) for offsets in route_offsets), x


@torch.inference_mode()
def prepare_inputs(args: argparse.Namespace, device: torch.device) -> TableGatherInputs:
    dtype = DTYPES[args.dtype]
    W = torch.randn((args.experts, args.hidden, args.output_dim), dtype=dtype, device=device)
    W.mul_(1.0 / math.sqrt(args.hidden))
    if args.multi_buffer_gather:
        tokens = args.tokens_per_buffer or args.tokens
        routes = args.routes_per_buffer or args.routes
        counts_by_buffer = multi_buffer_route_counts(
            routes, args.experts, args.num_input_buffers
        )
        X = tuple(
            torch.randn((tokens, args.hidden), dtype=dtype, device=device)
            for _ in range(args.num_input_buffers)
        )
        idx_buffers = []
        for counts in counts_by_buffer:
            A_idx_j = torch.empty(routes, dtype=torch.int32, device=device)
            offset = 0
            for count in counts:
                if args.routing_with_replacement:
                    indices = torch.randint(tokens, (count,), dtype=torch.int32, device=device)
                else:
                    indices = torch.randperm(tokens, dtype=torch.int32, device=device)[:count]
                A_idx_j[offset : offset + count].copy_(indices)
                offset += count
            idx_buffers.append(A_idx_j)
        A_idx = tuple(idx_buffers)
        work_table, offsets, x = build_multi_buffer_work_table(
            counts_by_buffer,
            output_dim=args.output_dim,
            tile_m=args.tile_m,
            tile_n=args.tile_n,
            cluster_m=args.cluster_m,
            max_swizzle_size=args.max_swizzle_size,
            device=device,
        )
        total_routes = args.num_input_buffers * routes
    else:
        counts = route_counts(args.routes, args.experts)
        X = torch.randn((args.tokens, args.hidden), dtype=dtype, device=device)
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
        total_routes = args.routes
    output = torch.empty((total_routes, args.output_dim), dtype=dtype, device=device)
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
            multi_buffer_gather=args.multi_buffer_gather,
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
    X_buffers = inputs.X if isinstance(inputs.X, tuple) else (inputs.X,)
    idx_buffers = inputs.A_idx if isinstance(inputs.A_idx, tuple) else (inputs.A_idx,)
    experts = len(inputs.route_offsets[0]) - 1
    for expert in range(experts):
        output_offset = sum(offsets[expert] for offsets in inputs.route_offsets)
        for X_j, A_idx_j, offsets in zip(X_buffers, idx_buffers, inputs.route_offsets):
            start, end = offsets[expert : expert + 2]
            if start == end:
                continue
            reference = X_j[A_idx_j[start:end].long()].float() @ inputs.W[expert].float()
            reference = reference.to(inputs.output.dtype)
            actual = inputs.output[output_offset : output_offset + end - start]
            max_abs_error = max(max_abs_error, (actual - reference).abs().max().item())
            torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
            output_offset += end - start
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
    counts = [
        [b - a for a, b in zip(offsets[:-1], offsets[1:])]
        for offsets in inputs.route_offsets
    ]
    print(f"Device: {torch.cuda.get_device_name(device)} (SM{capability[0]}{capability[1]})")
    X_buffers = inputs.X if isinstance(inputs.X, tuple) else (inputs.X,)
    print(
        f"X buffers: {len(X_buffers)} x {tuple(X_buffers[0].shape)}, "
        f"W: {tuple(inputs.W.shape)}, dtype: {X_buffers[0].dtype}"
    )
    print(f"Routes per expert by buffer: {counts}")
    print(
        f"Work table: {tuple(inputs.work_table.shape)}, x={inputs.work_group_size}, "
        f"expanded work IDs={inputs.work_table.shape[0] * inputs.work_group_size}"
    )
    print(
        f"Kernel: tile=({args.tile_m}, {args.tile_n}, {args.tile_k or 'auto'}), "
        f"cluster=({args.cluster_m}, 1, 1), static persistent, table gather=cp.async, "
        f"multi-buffer={args.multi_buffer_gather}"
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
    total_routes = inputs.output.shape[0]
    tflops = 2 * total_routes * args.hidden * args.output_dim / (median_ms * 1e9)
    print(f"Per-kernel samples (ms): {[round(value, 4) for value in timings_ms]}")
    print(f"Median kernel time: {median_ms:.4f} ms")
    print(f"Effective throughput: {tflops:.2f} TFLOP/s")

    if not args.skip_check:
        max_abs_error = check_correctness(inputs, atol=args.atol, rtol=args.rtol)
        print(f"Reference check: PASSED (max absolute error {max_abs_error:.6g})")


if __name__ == "__main__":
    main()
