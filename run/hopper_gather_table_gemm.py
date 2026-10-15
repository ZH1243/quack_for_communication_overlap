#!/usr/bin/env python3
"""Benchmark Hopper grouped GEMM with fused gather-A and an HBM work table.

The up GEMM receives no ``cu_seqlens_m``. Instead, each contiguous int32 table
row contains

    (expert_id, route_start, route_end, cid_n_base)

and expands to ``x = min(max_swizzle_size, ceil(N / tile_N))`` consecutive
AlongN work IDs. The runner requires ``ceil(N / tile_N) % x == 0``.

Tensor shapes:

    X:                 [T, K]
    W_up:              [E, K, N] (or [E, K, 2N] for a gated activation)
    W_down:            [E, N, K] (only with --down-projection)
    A_idx:             [R]
    gather_work_table: [Q, 4]
    up_output:         [R, N]
    output:            [R, N], or [R, K] with --down-projection

With ``--indexed-gather`` (single-buffer only), rows instead contain
``(expert_id, cid_n_base, output_start, output_end, token_idx_0, ..., token_idx_{C-1})``, where
``C = tile_m * cluster_m``. Indices directly address X, bypassing A_idx.
Each M-cluster has consecutive N-group rows in increasing cid_n_base order.
Output rows are packed using explicit output ranges. Partial tiles retain
trailing -1 token slots in the table, without padding the output. A_idx remains
the original reference/sizing vector and is not read by the indexed kernel.

With ``--multi-buffer-gather``, the kernel reads separately allocated token
and route-index buffers without materializing their concatenation:

    X_j:               [tokens_per_buffer, K]
    A_idx_j:           [routes_per_buffer]
    gather_work_table: [Q, 2 + 2 * num_input_buffers]
    output:            [num_input_buffers * routes_per_buffer, N or K]

Multi-buffer table rows contain ``(expert_id, cid_n_base, start_0, end_0,
..., start_b-1, end_b-1)``. Routes are packed in expert-major order and, within
an expert, in increasing buffer order. ``--balanced-multi-buffer-gather``
instead balances every M-cluster tile across the nonempty input buffers.
``--round-robin-m-clusters`` keeps all N groups for one expert/M-cluster
consecutive, then interleaves those bundles across experts.

Example:

    python run/hopper_gather_table_gemm.py \
        --tokens 4096 --hidden 4096 --output-dim 4096 \
        --experts 8 --routes 8195 --warmup 5 --iterations 100

    python run/hopper_gather_table_gemm.py --multi-buffer-gather \
        --num-input-buffers 3 --tokens-per-buffer 4096 \
        --routes-per-buffer 8195 --hidden 4096 --output-dim 4096 --experts 8

    python run/hopper_gather_table_gemm.py --multi-buffer-gather \
        --balanced-multi-buffer-gather --num-input-buffers 3

    python run/hopper_gather_table_gemm.py --multi-buffer-gather \
        --round-robin-m-clusters --num-input-buffers 3

Fuse a SwiGLU up-projection (the work table covers the 2N preactivation):

    python run/hopper_gather_table_gemm.py \
        --tokens 4096 --hidden 4096 --output-dim 14336 \
        --experts 8 --routes 8195 --activation swiglu --down-projection
"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass

import torch

from quack.gemm_interface import act_to_pytorch_fn_map, gated_to_pytorch_fn_map


if __package__:
    from . import hopper_pregather_gemm as pregather
else:
    import hopper_pregather_gemm as pregather


DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

ACTIVATIONS = tuple(name for name in act_to_pytorch_fn_map if name is not None)
GATED_ACTIVATIONS = tuple(gated_to_pytorch_fn_map)


@dataclass
class TableGatherInputs:
    X: torch.Tensor | tuple[torch.Tensor, ...]
    W: torch.Tensor
    A_idx: torch.Tensor | tuple[torch.Tensor, ...]
    work_table: torch.Tensor
    route_offsets: tuple[tuple[int, ...], ...]
    output_segments: tuple[tuple[int, int, int, int], ...]
    output: torch.Tensor | tuple[torch.Tensor, ...]
    work_group_size: int
    W_down: torch.Tensor | None = None
    up_output: torch.Tensor | None = None
    cu_seqlens_m: torch.Tensor | None = None
    down_inputs: pregather.PregatherInputs | None = None
    down_args: argparse.Namespace | None = None
    original_X: tuple[torch.Tensor, ...] | None = None


def make_arg_parser(
    description: str = "Benchmark QuACK's SM90 table-scheduled grouped GEMM with gather-A.",
    *,
    include_activation: bool = True,
    include_down_projection: bool = False,
    include_down_projection2: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description
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
        "--balanced-multi-buffer-gather",
        action="store_true",
        help="Balance each M-cluster tile across all nonempty input buffers",
    )
    parser.add_argument(
        "--round-robin-m-clusters",
        action="store_true",
        help="Interleave completed M-cluster bundles across experts",
    )
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
    if include_activation:
        parser.add_argument(
            "--activation",
            choices=(*ACTIVATIONS, *GATED_ACTIVATIONS),
            default=None,
            help=(
                "Fuse this activation into the GEMM epilogue. Gated activations treat "
                "--output-dim as the final width and use a 2x-wide projection."
            ),
        )
    if include_activation and include_down_projection:
        down_group = parser.add_mutually_exclusive_group()
        down_group.add_argument(
            "--down-projection",
            action="store_true",
            help="Run a grouped down projection after activation (requires --activation)",
        )
        if include_down_projection2:
            down_group.add_argument(
                "--down-projection2", action="store_true",
                help="Use the pregather plain-GEMM features for down projection (requires --activation)",
            )
            pregather.add_kernel_arguments(parser, prefix="down-")
            parser.add_argument(
                "--down-combine-in-place", action="store_true",
                help="Reduce routed down outputs into the original input allocations; "
                "requires multi-buffer-gather and down-projection2. Enables multi-buffer "
                "scatter and bulk_rows_reduce; benchmark restores inputs every invocation.",
            )
    parser.add_argument("--tile-m", type=int, default=256)
    parser.add_argument("--tile-n", type=int, default=256)
    parser.add_argument("--tile-k", type=int, default=None)
    parser.add_argument("--cluster-m", type=int, default=2)
    parser.add_argument("--max-swizzle-size", type=int, default=8)
    parser.add_argument("--pingpong", action="store_true")
    parser.add_argument("--routing-with-replacement", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--timing-samples", type=int, default=5)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser


def parse_args() -> argparse.Namespace:
    parser = make_arg_parser(include_down_projection=True, include_down_projection2=True)
    parser.add_argument(
        "--indexed-gather",
        action="store_true",
        help="Single-buffer table with direct token indices and -1 tail padding",
    )
    return parser.parse_args()


def has_down_projection(args: argparse.Namespace) -> bool:
    return getattr(args, "down_projection", False) or getattr(args, "down_projection2", False)


def down_projection_args(args: argparse.Namespace, routes: int) -> argparse.Namespace:
    """Translate independent down options into the shared plain-GEMM configuration."""
    values = vars(args).copy()
    values.update({
        key[5:]: value for key, value in vars(args).items()
        if key.startswith("down_") and key not in ("down_projection", "down_projection2")
    })
    values.update(
        hidden=args.output_dim, output_dim=args.hidden, routes=routes,
        activation=None, down_projection=False,
    )
    if getattr(args, "down_combine_in_place", False):
        values.update(
            scatter_table=True, multi_buffer_scatter=True,
            scatter_table_with_replacement=True, epilogue_store="bulk_rows_reduce",
            scatter_output_buffers=args.num_input_buffers,
        )
    down = argparse.Namespace(**values)
    try:
        pregather.validate_args(down, prepacked=True)
    except ValueError as exc:
        raise ValueError(f"down-projection2 configuration: {exc}") from exc
    return down


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
    if args.pingpong:
        if args.tile_m not in (64, 128, 192):
            raise ValueError("pingpong requires tile-m to be 64, 128, or 192")
        tile_n_max = 256 if args.tile_m == 64 else (208 if args.tile_m == 128 else 128)
        if args.tile_n % 16 or args.tile_n > tile_n_max:
            raise ValueError(
                f"pingpong with tile-m={args.tile_m} requires tile-n divisible by 16 "
                f"and no larger than {tile_n_max}"
            )
    if getattr(args, "down_combine_in_place", False):
        if not getattr(args, "down_projection2", False) or not args.multi_buffer_gather:
            raise ValueError("--down-combine-in-place requires --down-projection2 and --multi-buffer-gather")
        if (args.down_scatter_table_with_replacement
                or args.down_scatter_table_destination_rows is not None
                or args.down_scatter_output_rows is not None
                or args.down_scatter_output_capacity_rows is not None):
            raise ValueError("in-place combine derives destinations and buffer sizes from input routing; "
                             "omit random-destination and output-row options")
        if args.down_tile_n <= 0:
            raise ValueError("down-tile-n must be positive")
        if args.down_n_tile_major_output and args.hidden % args.down_tile_n:
            raise ValueError("in-place N-tile-major output requires hidden divisible by down-tile-n "
                             "so padding does not exceed the original allocation")
    if getattr(args, "down_projection2", False) and getattr(args, "down_projection", False):
        raise ValueError("--down-projection and --down-projection2 are mutually exclusive")
    if has_down_projection(args) and getattr(args, "activation", None) is None:
        raise ValueError("--down-projection/--down-projection2 requires --activation")
    tokens = args.tokens_per_buffer if args.multi_buffer_gather else args.tokens
    routes = args.routes_per_buffer if args.multi_buffer_gather else args.routes
    if tokens is None:
        tokens = args.tokens
    if routes is None:
        routes = args.routes
    if getattr(args, "down_projection2", False):
        down_projection_args(args, routes * (args.num_input_buffers if args.multi_buffer_gather else 1))
    if tokens <= 0 or routes <= 0:
        raise ValueError("tokens-per-buffer and routes-per-buffer must be positive")
    if getattr(args, "indexed_gather", False) and args.multi_buffer_gather:
        raise ValueError("--indexed-gather is only supported in single-buffer mode")
    if args.multi_buffer_gather and args.num_input_buffers < 2:
        raise ValueError("num-input-buffers must be at least 2 in multi-buffer mode")
    if args.balanced_multi_buffer_gather and not args.multi_buffer_gather:
        raise ValueError("balanced-multi-buffer-gather requires multi-buffer-gather")
    if args.round_robin_m_clusters and not args.multi_buffer_gather:
        raise ValueError("round-robin-m-clusters requires multi-buffer-gather")
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
    activation = getattr(args, "activation", None)
    gemm_output_dim = args.output_dim * (2 if activation in GATED_ACTIVATIONS else 1)
    clusters_n = math.ceil(gemm_output_dim / args.tile_n)
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
    n_group_major: bool = False,
    wave_work_ids: int | None = None,
) -> tuple[torch.Tensor, tuple[tuple[int, ...], ...], int]:
    """Encode serpentine order, optionally keeping each wave on one expert.

    wave_work_ids enables global N-group traversal with same-expert borrowing
    from later groups. Experts may share a wave only when one exhausts all work.
    """
    route_offsets = [0]
    for count in counts:
        route_offsets.append(route_offsets[-1] + count)

    clusters_n = math.ceil(output_dim / tile_n)
    x = min(max_swizzle_size, clusters_n)
    assert clusters_n % x == 0
    if wave_work_ids is not None and (wave_work_ids <= 0 or wave_work_ids % x):
        raise ValueError(
            "wave_work_ids must be positive and divisible by effective_swizzle_size, "
            f"got wave_work_ids={wave_work_ids}, effective_swizzle_size={x}"
        )
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

    if wave_work_ids is not None:
        # The default rows are already each expert's complete serpentine stream.
        # A cursor also records groups consumed early to fill a previous wave.
        expert_rows = [[] for _ in counts]
        for row in rows:
            expert_rows[row[0]].append(row)
        cursors = [0] * len(counts)
        rows_per_wave = wave_work_ids // x
        rows = []
        for group in range(clusters_n // x):
            experts = range(len(counts))
            if group % 2:
                experts = reversed(experts)
            for expert in experts:
                stream = expert_rows[expert]
                group_end = (group + 1) * math.ceil(counts[expert] / cluster_rows)
                while cursors[expert] < group_end:
                    room = rows_per_wave - len(rows) % rows_per_wave
                    end = min(cursors[expert] + room, len(stream))
                    rows.extend(stream[cursors[expert] : end])
                    cursors[expert] = end
    elif n_group_major:
        # Global route offsets order both experts and their M clusters. Reverse
        # that complete traversal on odd N groups, keeping each descriptor intact.
        rows.sort(key=lambda row: (row[3], -row[1] if (row[3] // x) % 2 else row[1]))

    table = torch.tensor(rows, dtype=torch.int32, device=device)
    return table, (tuple(route_offsets),), x


def build_indexed_work_table(
    counts: list[int],
    token_indices: torch.Tensor,
    *,
    output_dim: int,
    tile_m: int,
    tile_n: int,
    cluster_m: int,
    max_swizzle_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[tuple[int, ...], ...], int]:
    """Expand the legacy A_idx route ranges into direct X indices.

    Each tile copies token_indices[route_start:route_end], using exactly the
    same route ranges as build_work_table. No tokens are sampled here. N groups
    are reordered into consecutive M-major bundles for the indexed scheduler.
    Each row stores its packed output range before the direct token indices.
    Partial tiles have trailing -1 token slots, but no output padding.
    The original index vector is retained for sizing and reference checks only.
    The kernel loads indices directly from the work table.
    """
    cluster_rows = tile_m * cluster_m
    clusters_n = math.ceil(output_dim / tile_n)
    x = min(max_swizzle_size, clusters_n)
    if clusters_n % x:
        raise ValueError("indexed gather requires clusters_n divisible by the work group size")
    num_n_groups = clusters_n // x
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    num_clusters = sum(math.ceil(count / cluster_rows) for count in counts)
    table = torch.full(
        (num_clusters * num_n_groups, 4 + cluster_rows),
        -1, dtype=torch.int32, device=device,
    )
    row = 0
    for expert, count in enumerate(counts):
        for start in range(offsets[expert], offsets[expert + 1], cluster_rows):
            end = min(start + cluster_rows, offsets[expert + 1])
            for n_group in range(num_n_groups):
                table[row, 0] = expert
                table[row, 1] = n_group * x
                table[row, 2] = start
                table[row, 3] = end
                table[row, 4 : 4 + end - start] = token_indices[start:end]
                row += 1
    return table, token_indices, (tuple(offsets),), x


def build_multi_buffer_work_table(
    counts_by_buffer: list[list[int]],
    *,
    output_dim: int,
    tile_m: int,
    tile_n: int,
    cluster_m: int,
    max_swizzle_size: int,
    device: torch.device,
    balance_buffers: bool = False,
    round_robin_m_clusters: bool = False,
) -> tuple[
    torch.Tensor,
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, int, int, int], ...],
    int,
]:
    """Build M-cluster rows spanning independent route buffers.

    In the default mode, each M-cluster tile drains one buffer before moving
    to the next. In balanced mode, each tile uses a max-min fair allocation
    across all buffers that still have routes for the current expert. These
    choices affect only the ranges within a cluster. Round-robin mode instead
    changes the order of completed expert/M-cluster bundles in the table.
    """
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
    output_segments: list[tuple[int, int, int, int]] = []
    cluster_ranges_by_expert: list[list[tuple[int, ...]]] = []

    for expert in range(experts):
        expert_counts = [counts_by_buffer[j][expert] for j in range(num_buffers)]
        consumed = [0] * num_buffers
        cluster_ranges: list[tuple[int, ...]] = []
        while sum(consumed) < sum(expert_counts):
            remaining = [count - cursor for count, cursor in zip(expert_counts, consumed)]
            capacity = min(cluster_rows, sum(remaining))
            allocations = (
                balanced_buffer_allocations(remaining, capacity)
                if balance_buffers
                else sequential_buffer_allocations(remaining, capacity)
            )
            ranges: list[int] = []
            for j, allocation in enumerate(allocations):
                start = route_offsets[j][expert] + consumed[j]
                end = start + allocation
                ranges.extend((start, end))
                if allocation:
                    output_segments.append((expert, j, start, end))
                consumed[j] += allocation
            cluster_ranges.append(tuple(ranges))
        cluster_ranges_by_expert.append(cluster_ranges)

    num_n_groups = clusters_n // x
    if round_robin_m_clusters:
        max_m_clusters = max(map(len, cluster_ranges_by_expert), default=0)
        for cid_m in range(max_m_clusters):
            for expert, cluster_ranges in enumerate(cluster_ranges_by_expert):
                if cid_m >= len(cluster_ranges):
                    continue
                for n_group in range(num_n_groups):
                    rows.append((expert, n_group * x, *cluster_ranges[cid_m]))
    else:
        for expert, cluster_ranges in enumerate(cluster_ranges_by_expert):
            for n_group in range(num_n_groups):
                m_clusters = range(len(cluster_ranges))
                if n_group % 2:
                    m_clusters = reversed(range(len(cluster_ranges)))
                cid_n_base = n_group * x
                for cid_m in m_clusters:
                    rows.append((expert, cid_n_base, *cluster_ranges[cid_m]))

    table = torch.tensor(rows, dtype=torch.int32, device=device)
    return (
        table,
        tuple(tuple(offsets) for offsets in route_offsets),
        tuple(output_segments),
        x,
    )


def sequential_buffer_allocations(remaining: list[int], capacity: int) -> list[int]:
    """Fill one tile by draining buffers in increasing index order."""
    allocations = [0] * len(remaining)
    for j, available in enumerate(remaining):
        allocation = min(available, capacity)
        allocations[j] = allocation
        capacity -= allocation
        if capacity == 0:
            break
    return allocations


def balanced_buffer_allocations(remaining: list[int], capacity: int) -> list[int]:
    """Max-min balance one tile across buffers, without exceeding availability."""
    allocations = [0] * len(remaining)
    available = remaining.copy()
    active = [j for j, count in enumerate(available) if count]
    while capacity and active:
        share = capacity // len(active)
        minimum = min(available[j] for j in active)
        if minimum <= share:
            for j in active:
                allocations[j] += minimum
                available[j] -= minimum
            capacity -= minimum * len(active)
            active = [j for j in active if available[j]]
            continue

        for j in active:
            allocations[j] += share
            available[j] -= share
        capacity -= share * len(active)
        if capacity:
            # Prefer assigning the small integer remainder to one buffer. If
            # none can supply it alone, drain the residual capacities in order.
            remainder_owner = next((j for j in active if available[j] >= capacity), None)
            if remainder_owner is not None:
                allocations[remainder_owner] += capacity
                capacity = 0
            else:
                for j in active:
                    allocation = min(available[j], capacity)
                    allocations[j] += allocation
                    capacity -= allocation
                    if capacity == 0:
                        break
    return allocations


@torch.inference_mode()
def prepare_inputs(args: argparse.Namespace, device: torch.device) -> TableGatherInputs:
    dtype = DTYPES[args.dtype]
    gated = args.activation in GATED_ACTIVATIONS
    gemm_output_dim = args.output_dim * (2 if gated else 1)
    if gated:
        W = torch.randn(
            (args.experts, gemm_output_dim, args.hidden), dtype=dtype, device=device
        ).transpose(1, 2)
    else:
        W = torch.randn(
            (args.experts, args.hidden, gemm_output_dim), dtype=dtype, device=device
        )
    W.mul_(1.0 / math.sqrt(args.hidden))
    W_down = None
    if has_down_projection(args):
        W_down = torch.randn(
            (args.experts, args.output_dim, args.hidden), dtype=dtype, device=device
        )
        W_down.mul_(1.0 / math.sqrt(args.output_dim))
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
        work_table, offsets, output_segments, x = build_multi_buffer_work_table(
            counts_by_buffer,
            output_dim=gemm_output_dim,
            tile_m=args.tile_m,
            tile_n=args.tile_n,
            cluster_m=args.cluster_m,
            max_swizzle_size=args.max_swizzle_size,
            device=device,
            balance_buffers=args.balanced_multi_buffer_gather,
            round_robin_m_clusters=args.round_robin_m_clusters,
        )
        down_counts = [
            sum(counts_by_buffer[buffer_idx][expert] for buffer_idx in range(len(X)))
            for expert in range(args.experts)
        ]
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
        # Both modes consume the same X, W, and sampled A_idx. Indexed mode
        # only embeds those routes and their packed output ranges in the table.
        if getattr(args, "indexed_gather", False):
            work_table, A_idx, offsets, x = build_indexed_work_table(
                counts, A_idx, output_dim=gemm_output_dim,
                tile_m=args.tile_m, tile_n=args.tile_n, cluster_m=args.cluster_m,
                max_swizzle_size=args.max_swizzle_size, device=device,
            )
            counts = [end - start for start, end in zip(offsets[0][:-1], offsets[0][1:])]
        else:
            work_table, offsets, x = build_work_table(
                counts,
                output_dim=gemm_output_dim,
                tile_m=args.tile_m,
                tile_n=args.tile_n,
                cluster_m=args.cluster_m,
                max_swizzle_size=args.max_swizzle_size,
                device=device,
            )
        output_segments = tuple(
            (expert, 0, offsets[0][expert], offsets[0][expert + 1])
            for expert in range(args.experts)
        )
        down_counts = counts
        total_routes = A_idx.shape[0]
    allocate_up = torch.zeros if getattr(args, "indexed_gather", False) else torch.empty
    up_output = allocate_up((total_routes, args.output_dim), dtype=dtype, device=device)
    output = (
        torch.empty((total_routes, args.hidden), dtype=dtype, device=device)
        if args.down_projection
        else up_output
    )
    cu_seqlens_m = None
    if has_down_projection(args):
        cumulative_routes = [0]
        for count in down_counts:
            cumulative_routes.append(cumulative_routes[-1] + count)
        assert cumulative_routes[-1] == total_routes
        cu_seqlens_m = torch.tensor(cumulative_routes, dtype=torch.int32, device=device)
    inputs = TableGatherInputs(
        X,
        W,
        A_idx,
        work_table,
        offsets,
        output_segments,
        output,
        x,
        W_down=W_down,
        up_output=up_output,
        cu_seqlens_m=cu_seqlens_m,
    )

    if getattr(args, "down_projection2", False):
        down_args = down_projection_args(args, total_routes)
        inputs.down_args = down_args
        output_buffers = None
        scatter_mapping = None
        if getattr(args, "down_combine_in_place", False):
            inputs.original_X = tuple(buffer.clone() for buffer in X)
            output_buffers = tuple(
                buffer.view(args.hidden // down_args.tile_n, -1)
                if down_args.n_tile_major_output else buffer
                for buffer in X
            )
            scatter_mapping = build_original_token_mapping(inputs)
        inputs.down_inputs = pregather.prepare_grouped_inputs(
            down_args, up_output, W_down, down_counts,
            output_buffers=output_buffers, scatter_mapping=scatter_mapping,
        )
        inputs.output = inputs.down_inputs.output
    return inputs


def build_original_token_mapping(inputs: TableGatherInputs) -> torch.Tensor:
    """Map packed intermediate rows back to (input buffer, original token)."""
    if not isinstance(inputs.A_idx, tuple):
        raise ValueError("original-token combine requires multi-buffer routing")
    mapping = torch.empty(
        (inputs.up_output.shape[0], 2), dtype=torch.int64, device=inputs.up_output.device
    )
    offset = 0
    for _, buffer_id, start, end in inputs.output_segments:
        count = end - start
        mapping[offset:offset + count, 0] = buffer_id
        mapping[offset:offset + count, 1] = inputs.A_idx[buffer_id][start:end]
        offset += count
    if offset != mapping.shape[0]:
        raise ValueError("output segments do not cover all intermediate routes")
    return mapping


def make_launch(args: argparse.Namespace, inputs: TableGatherInputs):
    from quack.gemm import gemm as quack_gemm
    from quack.gemm_config import GemmConfig
    from quack.gemm_interface import gemm_act

    B = inputs.W.transpose(1, 2)  # QuACK convention: [E, N, K].
    B_down = inputs.W_down.transpose(1, 2) if inputs.W_down is not None else None
    up_output = inputs.up_output if inputs.up_output is not None else inputs.output

    if args.activation is not None:
        config = GemmConfig(
            tile_m=args.tile_m,
            tile_n=args.tile_n,
            tile_k=args.tile_k,
            pingpong=args.pingpong,
            is_dynamic_persistent=False,
            cluster_m=args.cluster_m,
            cluster_n=1,
            cluster_k=1,
            max_swizzle_size=args.max_swizzle_size,
            device_capacity=9,
            use_tma_gather=False,
        )

    down_launch = (
        pregather.make_launch(inputs.down_args, inputs.down_inputs)
        if inputs.down_inputs is not None else None
    )

    # prepare_inputs creates inference tensors. Restores, reduction resets, and
    # kernel-side counter resets must be valid even when the caller is not in
    # inference mode (the CLI and CUDA graph capture both use this closure).
    @torch.inference_mode()
    def launch(*, reset_events=None) -> None:
        if inputs.original_X is not None:
            # Benchmark each invocation on identical inputs, including graph replay.
            for buffer, original in zip(inputs.X, inputs.original_X):
                buffer.copy_(original)
        if args.activation is not None:
            gemm_act(
                inputs.X,
                inputs.W,
                activation=args.activation,
                postact_out=up_output,
                A_idx=inputs.A_idx,
                gather_work_table=inputs.work_table,
                multi_buffer_gather=args.multi_buffer_gather,
                store_preact=False,
                dynamic_scheduler=False,
                tuned=False,
                config=config,
                concat_layout=("B",) if args.activation in GATED_ACTIVATIONS else None,
            )
        else:
            quack_gemm(
                inputs.X,
                B,
                up_output,
                C=None,
                tile_count_semaphore=None,
                tile_M=args.tile_m,
                tile_N=args.tile_n,
                tile_K=args.tile_k,
                cluster_M=args.cluster_m,
                cluster_N=1,
                cluster_K=1,
                pingpong=args.pingpong,
                persistent=True,
                is_dynamic_persistent=False,
                max_swizzle_size=args.max_swizzle_size,
                cu_seqlens_m=None,
                A_idx=inputs.A_idx,
                gather_work_table=inputs.work_table,
                multi_buffer_gather=args.multi_buffer_gather,
                use_tma_gather=False,
            )

        if down_launch is not None:
            # A reduction must start fresh on every invocation, including graph replay.
            if inputs.down_args.epilogue_store == "bulk_rows_reduce":
                if reset_events is not None:
                    reset_events[0].record()
                inputs.down_inputs.zero_up_output()
                if reset_events is not None:
                    reset_events[1].record()
            down_launch()
        elif args.down_projection:
            assert B_down is not None and inputs.cu_seqlens_m is not None
            # The gather-table output is packed in expert-contiguous order, so
            # the down projection is an ordinary varlen grouped GEMM.
            quack_gemm(
                up_output,
                B_down,
                inputs.output,
                C=None,
                tile_count_semaphore=None,
                tile_M=args.tile_m,
                tile_N=args.tile_n,
                tile_K=args.tile_k,
                cluster_M=args.cluster_m,
                cluster_N=1,
                cluster_K=1,
                pingpong=args.pingpong,
                persistent=True,
                is_dynamic_persistent=False,
                max_swizzle_size=args.max_swizzle_size,
                cu_seqlens_m=inputs.cu_seqlens_m,
                A_idx=None,
                use_tma_gather=False,
            )

    return launch


@dataclass
class PipelineTimings:
    including_reset_ms: list[float]
    reset_ms: list[float]
    excluding_reset_ms: list[float]


@torch.inference_mode()
def benchmark(
    launch, *, iterations: int, samples: int, use_cuda_graph: bool, measure_reset: bool = False
) -> PipelineTimings:
    """Time the whole pipeline and subtract its measured output-reset intervals.

    Reset events stay between the up and down GEMMs. External events update on
    graph replay. Input restoration (if enabled) remains in both pipeline times.
    Ordinary launches can include host enqueue gaps; prefer CUDA graphs.
    """
    reset_events = [
        (
            torch.cuda.Event(enable_timing=True, external=use_cuda_graph),
            torch.cuda.Event(enable_timing=True, external=use_cuda_graph),
        )
        for _ in range(iterations)
    ] if measure_reset else []

    def measured_launches():
        for i in range(iterations):
            if measure_reset:
                launch(reset_events=reset_events[i])
            else:
                launch()

    graph = None
    if use_cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            measured_launches()
        graph.replay()
        torch.cuda.synchronize()

    timings = PipelineTimings([], [], [])
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if graph is not None:
            graph.replay()
        else:
            measured_launches()
        end.record()
        end.synchronize()
        total_ms = start.elapsed_time(end) / iterations
        reset_ms = sum(a.elapsed_time(b) for a, b in reset_events) / iterations
        timings.including_reset_ms.append(total_ms)
        timings.reset_ms.append(reset_ms)
        timings.excluding_reset_ms.append(total_ms - reset_ms)
    return timings


@torch.inference_mode()
def check_correctness(
    inputs: TableGatherInputs,
    *,
    activation: str | None = None,
    atol: float,
    rtol: float,
) -> tuple[float, float]:
    max_abs_error = 0.0
    max_allowed_error = 0.0
    X_buffers = inputs.original_X or (inputs.X if isinstance(inputs.X, tuple) else (inputs.X,))
    idx_buffers = inputs.A_idx if isinstance(inputs.A_idx, tuple) else (inputs.A_idx,)
    up_output = inputs.up_output if inputs.up_output is not None else inputs.output
    output_offset = 0
    for expert, buffer_idx, start, end in inputs.output_segments:
        if start == end:
            continue
        X_j, A_idx_j = X_buffers[buffer_idx], idx_buffers[buffer_idx]
        valid = A_idx_j[start:end] >= 0
        indices = A_idx_j[start:end][valid].long()
        if indices.numel() == 0:
            output_offset += end - start
            continue
        reference = X_j[indices].float() @ inputs.W[expert].float()
        if activation in GATED_ACTIVATIONS:
            gate, up = reference.chunk(2, dim=-1)
            reference = gated_to_pytorch_fn_map[activation](gate, up)
        elif activation is not None:
            reference = act_to_pytorch_fn_map[activation](reference)
        actual = up_output[output_offset : output_offset + end - start][valid]
        abs_error = (actual.float() - reference).abs().max().item()
        max_abs_error = max(max_abs_error, abs_error)
        if activation is None:
            reference_out = reference.to(actual.dtype)
            torch.testing.assert_close(actual, reference_out, atol=atol, rtol=rtol)
            allowed_error = atol + rtol * reference_out.float().abs().max().item()
        else:
            baseline = X_j[indices] @ inputs.W[expert]
            if activation in GATED_ACTIVATIONS:
                gate, up = baseline.chunk(2, dim=-1)
                baseline = gated_to_pytorch_fn_map[activation](gate, up)
            else:
                baseline = act_to_pytorch_fn_map[activation](baseline)
            baseline_error = (baseline.float() - reference).abs().max().item()
            allowed_error = max(
                atol + rtol * reference.abs().max().item(), 2 * baseline_error + 1e-5
            )
            if abs_error > allowed_error:
                raise AssertionError(
                    f"activation output error {abs_error:.6g} exceeds {allowed_error:.6g}"
                )
        max_allowed_error = max(max_allowed_error, allowed_error)
        output_offset += end - start
    assert output_offset == up_output.shape[0]
    return max_abs_error, max_allowed_error


@torch.inference_mode()
def check_down_correctness(
    inputs: TableGatherInputs,
    *,
    atol: float,
    rtol: float,
) -> tuple[float, float]:
    """Check the optional down projection against per-expert float32 references."""
    if inputs.down_inputs is not None:
        error, allowed, _, _ = pregather.check_correctness(
            inputs.down_inputs, activation=None, atol=atol, rtol=rtol
        )
        return error, allowed
    assert inputs.W_down is not None
    assert inputs.up_output is not None
    assert inputs.cu_seqlens_m is not None
    cumulative_routes = inputs.cu_seqlens_m.cpu().tolist()
    max_abs_error = 0.0
    max_allowed_error = 0.0
    for expert, (start, end) in enumerate(zip(cumulative_routes[:-1], cumulative_routes[1:])):
        if start == end:
            continue
        up_actual = inputs.up_output[start:end]
        W_down_expert = inputs.W_down[expert]
        reference = up_actual.float() @ W_down_expert.float()
        baseline = up_actual @ W_down_expert
        actual = inputs.output[start:end]
        if not torch.isfinite(actual).all():
            raise AssertionError(f"expert {expert} down output contains NaN or infinity")
        abs_error = (actual.float() - reference).abs().max().item()
        baseline_error = (baseline.float() - reference).abs().max().item()
        allowed_error = max(
            atol + rtol * reference.abs().max().item(), 2 * baseline_error + 1e-5
        )
        if abs_error > allowed_error:
            raise AssertionError(
                f"expert {expert} down output error {abs_error:.6g} exceeds "
                f"the permitted bound {allowed_error:.6g} "
                f"(same-dtype PyTorch baseline error {baseline_error:.6g})"
            )
        max_abs_error = max(max_abs_error, abs_error)
        max_allowed_error = max(max_allowed_error, allowed_error)
    return max_abs_error, max_allowed_error


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
        f"W_up: {tuple(inputs.W.shape)}, dtype: {X_buffers[0].dtype}"
    )
    if inputs.W_down is not None:
        assert inputs.up_output is not None
        output_shapes = (
            [tuple(t.shape) for t in inputs.down_inputs.output_buffers()]
            if inputs.down_inputs is not None else [tuple(inputs.output.shape)]
        )
        print(
            f"W_down: {tuple(inputs.W_down.shape)}, up output: {tuple(inputs.up_output.shape)}, "
            f"final output buffers: {output_shapes}"
        )
    else:
        print(f"Output: {tuple(inputs.output.shape)}, down projection: disabled")
    print(f"Routes per expert by buffer: {counts}")
    print(
        f"Work table: {tuple(inputs.work_table.shape)}, x={inputs.work_group_size}, "
        f"expanded work IDs={inputs.work_table.shape[0] * inputs.work_group_size}"
    )
    if inputs.down_args is not None:
        down_options = {k: v for k, v in vars(args).items() if k.startswith("down_")}
        print(f"Down-projection2 options: {down_options}")
        if inputs.original_X is not None:
            print("In-place local combine: original-token destinations, unweighted sum; "
                  "pipeline timing includes input restoration and output zeroing.")
        if inputs.down_args.epilogue_store == "bulk_rows_reduce":
            print("Pipeline timing includes down-output zeroing before each reduction.")
    down_description = (
        ", down=pregather configurable GEMM" if inputs.down_inputs is not None
        else ", down=expert-contiguous TMA" if args.down_projection else ""
    )
    print(
        f"Kernel: tile=({args.tile_m}, {args.tile_n}, {args.tile_k or 'auto'}), "
        f"cluster=({args.cluster_m}, 1, 1), static persistent, table gather=cp.async, "
        f"multi-buffer={args.multi_buffer_gather}, indexed-gather={args.indexed_gather}, "
        f"balanced-buffers={args.balanced_multi_buffer_gather}, "
        f"round-robin-m-clusters={args.round_robin_m_clusters}{down_description}"
    )
    print(f"Fused activation: {args.activation or 'disabled'}")
    compile_target = "up and down kernels" if has_down_projection(args) else "kernel"
    print(f"Compiling and warming up the specialized QuACK {compile_target}...")

    launch = make_launch(args, inputs)
    for _ in range(max(1, args.warmup)):
        launch()
    torch.cuda.synchronize(device)

    measure_reset = (
        inputs.down_inputs is not None and inputs.down_args.epilogue_store == "bulk_rows_reduce"
    )
    timings = benchmark(
        launch,
        iterations=args.iterations,
        samples=args.timing_samples,
        use_cuda_graph=not args.no_cuda_graph,
        measure_reset=measure_reset,
    )
    median_ms = statistics.median(timings.including_reset_ms)
    total_routes = (args.routes if args.indexed_gather else inputs.up_output.shape[0])
    up_flops = 2 * total_routes * args.hidden * inputs.W.shape[-1]
    down_flops = (
        2 * inputs.up_output.shape[0] * args.output_dim * args.hidden
        if has_down_projection(args) else 0
    )
    tflops = (up_flops + down_flops) / (median_ms * 1e9)
    operation_name = "two-GEMM MLP" if has_down_projection(args) else "up GEMM"
    if measure_reset:
        median_without_reset_ms = statistics.median(timings.excluding_reset_ms)
        for label, values in (
            ("Output reset", timings.reset_ms),
            ("Pipeline excluding reset", timings.excluding_reset_ms),
            ("Pipeline including reset", timings.including_reset_ms),
        ):
            print(f"{label} per-operation samples (ms): {[round(v, 4) for v in values]}")
        print(f"Median output reset time: {statistics.median(timings.reset_ms):.4f} ms")
        print(
            f"Median {operation_name} time (excluding reset): {median_without_reset_ms:.4f} ms"
        )
        print(f"Median {operation_name} time (including reset): {median_ms:.4f} ms")
        print(
            "Effective throughput (excluding reset): "
            f"{(up_flops + down_flops) / (median_without_reset_ms * 1e9):.2f} TFLOP/s"
        )
        print(f"Effective throughput (including reset): {tflops:.2f} TFLOP/s")
        print("Reset-excluded time subtracts output zeroing from the same pipeline samples.")
        if inputs.original_X is not None:
            print("Both throughput measurements include input restoration.")
    else:
        print(
            f"Per-operation samples (ms): {[round(v, 4) for v in timings.including_reset_ms]}"
        )
        print(f"Median {operation_name} time: {median_ms:.4f} ms")
        print(f"Effective throughput: {tflops:.2f} TFLOP/s")

    if not args.skip_check:
        max_abs_error, max_allowed_error = check_correctness(
            inputs, activation=args.activation, atol=args.atol, rtol=args.rtol
        )
        if has_down_projection(args):
            down_error, down_allowed = check_down_correctness(
                inputs, atol=args.atol, rtol=args.rtol
            )
            print(
                "Reference check: PASSED "
                f"(up max error {max_abs_error:.6g}, permitted {max_allowed_error:.6g}; "
                f"down max error {down_error:.6g}, permitted {down_allowed:.6g})"
            )
        else:
            print(
                "Reference check: PASSED "
                f"(max absolute error {max_abs_error:.6g}, "
                f"permitted bound {max_allowed_error:.6g})"
            )


if __name__ == "__main__":
    main()
