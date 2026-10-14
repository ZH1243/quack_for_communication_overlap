#!/usr/bin/env python3
"""Benchmark a QuACK Hopper grouped GEMM after materializing gathered tokens.

This is the no-gather-fusion counterpart of ``hopper_gather_gemm.py``. It first
creates the same MoE inputs on CUDA:

    X:             [T, K]
    W_up:          [E, K, N] (or [E, K, 2N] for a gated activation)
    W_down:        [E, N, K] (only with --down-projection)
    A_idx:         [R]
    cu_seqlens_m:  [E + 1]
    up_output:     [R, N]
    output:        [R, N], or [R, K] with --down-projection

It then materializes ``A = X[A_idx]`` as a contiguous ``[R, K]`` CUDA tensor
before invoking QuACK. By default, the timed up-projection receives A and cu_seqlens_m, but
does not receive A_idx, so its A operand uses the ordinary contiguous TMA path.
With ``--gather-table``, it instead uses the single-buffer, non-indexed table
kernel from ``hopper_gather_table_gemm.py`` with A as the token buffer and
identity int32 indices. The table replaces cu_seqlens_m in the up GEMM;
A loads still use cp.async gather. Table and identity-index construction are
excluded from timing. All three epilogue-store modes support table scheduling;
the bulk modes require plain GEMM without a fused activation.
Add ``--gather-table-n-group-major`` to traverse all experts for each N group,
reversing both expert and M-cluster order on alternate N groups.
Add ``--gather-table-expert-waves`` instead to fill logical persistent work
waves with one expert, borrowing from its later N groups when necessary.
Only exhausted experts may share a wave with the next expert. Wave boundaries
are not GPU-wide synchronization points unless --gather-table-wave-barrier is enabled.
This opt-in requires --pingpong and bulk_rows/bulk_rows_reduce. Each CTA drains
its tile stores completely before reporting completion; the next wave waits
for every preceding-wave CTA before entering its epilogue. The kernel uses a
cooperative launch (an oversized grid is rejected by CUDA). The per-call counter
reset is included in timing and CUDA Graph capture. The persistent cluster count must be
divisible by the effective swizzle size.
When ``--down-projection`` is enabled, the activated output is already
expert-contiguous and feeds a second grouped GEMM using the same cu_seqlens_m.
The pre-gather operation, compilation, warmup, graph capture, and correctness
check are excluded from the reported operation time.

Example:

    python run/hopper_pregather_gemm.py \
        --tokens 4096 --hidden 4096 --output-dim 4096 \
        --experts 8 --routes 8192 --warmup 5 --iterations 100

Select full-tile per-row bulk stores (also supports --pingpong):

    python run/hopper_pregather_gemm.py \
        --tile-m 128 --tile-n 128 --epilogue-store bulk_rows --pingpong

Use table scheduling with pre-gathered tokens:

    python run/hopper_pregather_gemm.py --gather-table

The default --epilogue-store tma retains the existing subtile tensor stores.
Add --bulk-row-swizzle with either bulk mode to retain TMA-style swizzled
output subtiles and issue 16-byte bulk copies/reductions from their physical
addresses. This trades more bulk operations for the TMA path's R2S bank pattern.
It supports scatter, N tails, and --bulk-row-double-buffer. Without this flag,
bulk stores retain the existing unswizzled, contiguous-row layout.
Add --bulk-row-double-buffer with --pingpong and either bulk-row mode to give
each warp group its own full-tile output SMEM buffer. Epilogues remain ordered,
but stores only wait for SMEM reads before the issuing warp group reuses its
own buffer. This costs one extra output tile of SMEM and may reduce A/B stages.
The optional wave store barrier still drains destination writes at every wave.

Diagnostic store ablation: --bulk-store-rows ROWS enables stores of only the
first ROWS (1..tile-m) of every CTA output tile, restarting at each expert.
For tile-m=128, compare 32, 64, 96, and 128. Omitting the flag preserves full
stores. MMA, full-tile R2S staging, SMEM allocation, and synchronization remain
unchanged; fewer store operations AND fewer bytes/reductions are issued.
Only ordinary pregather without gather-table, scatter, activation, or down
projection is supported by this runner. The reference check verifies computed
rows and exact zeros in omitted rows. Reported TFLOP/s still counts full GEMM
work and is a diagnostic metric for an intentionally incomplete output.

Both bulk_rows and bulk_rows_reduce require plain GEMM without --activation.
Output rows must have a 16-byte-multiple width (N divisible by 8 for BF16/FP16).
Full-tile staging uses more shared memory; use a 128x128 tile to start.
Add --scatter-table to either bulk mode to write A[i]'s result to
output[scatter_table[i]], using a seeded random CUDA int32 permutation of [0, R).
This works with or without --gather-table. Mapping construction is untimed;
index loads and scattered stores are included in GEMM timing.
Add --scatter-table-with-replacement with bulk_rows_reduce to sample destinations
with replacement. Contributions to each destination are summed; unused rows stay
zero. --scatter-table-destination-rows R1 restricts sampling to [0, R1), with
1 <= R1 <= R (default R), leaving output[R1:] zero. Smaller R1 increases the
average number of contributions per eligible destination to R/R1.
Add --n-tile-major-output with --scatter-table and either bulk mode to allocate
Y_prime directly as [ceil(N/tile_n), routes * tile_n]. Element (dst_row, n) is
stored at Y_prime[n // tile_n, dst_row * tile_n + n % tile_n]. The last tile is
padded; only valid columns are written and N * element_bytes must be divisible
by 16. Reduction resets the used region, including column padding, before each
launch. With --multi-buffer-scatter, each buffer instead has shape
[ceil(N/tile_n), rows_in_buffer * tile_n]; per-route metadata supplies its
tile-0 address and N-tile stride. Down projection is unsupported with this flag.
Consumers can recover ordinary Y with
Y_prime.view(-1, routes, tile_n).permute(1, 0, 2).reshape(routes, -1)[:, :N].
This may copy: reshaping alone does not restore row-major Y. The runner performs
this conversion only for correctness checking, outside GEMM timing.
For multiple buffers, apply the conversion with each buffer's row count and
concatenate the logical rows for checking.

Add --multi-buffer-scatter --scatter-output-buffers B to distribute the same
logical destination space across B separately allocated output buffers (default
B=2). A (buffer_id, row_index) mapping is converted to int64 destination-row
addresses before timing. No single [R,N] output is allocated in this mode;
concatenation is used only for untimed reference checking. This works with
expert-wave scheduling and both bulk store modes. For example:

    python run/hopper_pregather_gemm.py --gather-table \
        --gather-table-expert-waves --max-swizzle-size 1 \
        --scatter-table --multi-buffer-scatter --scatter-output-buffers 4 \
        --epilogue-store bulk_rows --tile-m 128 --tile-n 128

Add --n-tile-major-output to this command to use tiled storage in every buffer.
Use --scatter-output-rows U0 U1 ... to choose used rows per buffer (sum: routes),
and --scatter-output-capacity-rows C0 C1 ... to overallocate (Cb >= Ub > 0).
A single count broadcasts to all buffers. Defaults are a balanced used-row split
and capacity equal to used rows. All N tiles are packed into the first
ceil(N/tile_n) * Ub * tile_n elements of buffer b; unused capacity follows this
prefix and is neither written by GEMM nor cleared by the runner. The tensor's
allocated shape describes capacity, not the packed data's logical layout.

The correctness check accounts for output-dtype accumulation rounding.
All epilogue-store modes zero the output before every launch and use one
event pair per operation, excluding zeroing from the reported GEMM time.
This gives tma, bulk_rows, and bulk_rows_reduce the same reset/timing protocol.

Fuse a SwiGLU up-projection (the weight uses concatenated [gate | up] columns):

    python run/hopper_pregather_gemm.py \
        --tokens 4096 --hidden 4096 --output-dim 14336 \
        --experts 8 --routes 8192 --activation swiglu --down-projection
"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass

import torch

from quack.cute_dsl_utils import get_max_active_clusters
from quack.gemm import build_scatter_row_addresses, gemm as quack_gemm
from quack.gemm_config import GemmConfig
from quack.gemm_interface import act_to_pytorch_fn_map, gated_to_pytorch_fn_map, gemm_act


DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

ACTIVATIONS = tuple(name for name in act_to_pytorch_fn_map if name is not None)
GATED_ACTIVATIONS = tuple(gated_to_pytorch_fn_map)


@dataclass
class PregatherInputs:
    X: torch.Tensor
    W_up: torch.Tensor
    W_down: torch.Tensor | None
    A_idx: torch.Tensor
    cu_seqlens_m: torch.Tensor
    A: torch.Tensor
    up_output: torch.Tensor | tuple[torch.Tensor, ...]
    output: torch.Tensor | tuple[torch.Tensor, ...]
    gather_work_table: torch.Tensor | None = None
    identity_A_idx: torch.Tensor | None = None
    scatter_table: torch.Tensor | None = None
    scatter_row_addresses: torch.Tensor | None = None
    scatter_mapping: torch.Tensor | None = None
    scatter_with_replacement: bool = False
    n_tile_major_tile_n: int | None = None
    scatter_output_rows: tuple[int, ...] | None = None
    wave_work_ids: int | None = None

    def output_buffers(self) -> tuple[torch.Tensor, ...]:
        return self.up_output if isinstance(self.up_output, tuple) else (self.up_output,)

    def collected_up_output(self) -> torch.Tensor:
        # Reference checking only: no conversion on the GEMM launch path.
        if self.n_tile_major_tile_n is not None:
            tile_n, n = self.n_tile_major_tile_n, self.W_up.shape[-1]
            outputs = [
                buffer.view(-1)[:math.ceil(n / tile_n) * rows * tile_n]
                .view(-1, rows, tile_n)
                .permute(1, 0, 2)
                .reshape(rows, -1)[:, :n]
                for buffer, rows in zip(
                    self.output_buffers(),
                    self.scatter_output_rows
                    or tuple(b.shape[1] // tile_n for b in self.output_buffers()),
                )
            ]
            return torch.cat(outputs) if isinstance(self.up_output, tuple) else outputs[0]
        return torch.cat(self.up_output) if isinstance(self.up_output, tuple) else self.up_output

    def zero_up_output(self) -> None:
        for i, buffer in enumerate(self.output_buffers()):
            if self.scatter_output_rows is not None and self.n_tile_major_tile_n is not None:
                count = buffer.shape[0] * self.scatter_output_rows[i] * self.n_tile_major_tile_n
                buffer.view(-1)[:count].zero_()
            else:
                buffer.zero_()


def add_kernel_arguments(parser: argparse.ArgumentParser, *, prefix: str = "") -> None:
    """Share plain-GEMM options with composed runners, optionally namespaced."""

    def add_argument(*flags, **kwargs):
        if "help" in kwargs and prefix:
            kwargs["help"] = kwargs["help"].replace("--", "--" + prefix)
            kwargs["help"] = kwargs["help"].replace("--" + prefix + "routes", "--routes")
            kwargs["help"] = kwargs["help"].replace(
                "with output zeroing excluded from timing",
                "with output zeroing included in pipeline timing",
            )
        parser.add_argument(*(flag.replace("--", "--" + prefix, 1) for flag in flags), **kwargs)

    add_argument("--tile-m", type=int, default=256)
    add_argument("--tile-n", type=int, default=256)
    add_argument(
        "--tile-k",
        type=int,
        default=None,
        help="Optional K tile; by default QuACK chooses it from the operand dtype",
    )
    add_argument("--cluster-m", type=int, default=2)
    add_argument("--max-swizzle-size", type=int, default=8)
    add_argument("--pingpong", action="store_true")
    add_argument(
        "--bulk-store-rows",
        type=int,
        default=None,
        help="Enable diagnostic partial stores: write only the first ROWS of each M tile "
        "(1..tile-m). Full GEMM and R2S staging still execute. Bulk modes only; "
        "requires ordinary pregather without scatter, activation, or down projection.",
    )
    add_argument(
        "--bulk-row-swizzle",
        action="store_true",
        help="Use TMA-style swizzled output SMEM with 16-byte bulk copies; "
        "requires --epilogue-store bulk_rows or bulk_rows_reduce.",
    )
    add_argument(
        "--bulk-row-double-buffer",
        action="store_true",
        help="Use one output SMEM buffer per pingpong WG; requires --pingpong and "
        "--epilogue-store bulk_rows or bulk_rows_reduce. Costs one extra full output tile.",
    )
    add_argument(
        "--gather-table",
        action="store_true",
        help="Use table-scheduled cp.async gather on pre-gathered A with identity indices",
    )
    add_argument(
        "--gather-table-n-group-major",
        action="store_true",
        help="Traverse all experts per N group in serpentine order (requires --gather-table)",
    )
    add_argument(
        "--gather-table-expert-waves",
        action="store_true",
        help=(
            "Use global serpentine order with single-expert logical waves, borrowing "
            "later N groups until expert exhaustion (requires --gather-table; "
            "takes precedence over --gather-table-n-group-major)"
        ),
    )
    add_argument(
        "--gather-table-wave-barrier",
        action="store_true",
        help="Order bulk output stores across expert waves (requires "
        "--gather-table-expert-waves, --pingpong, and a bulk-row epilogue; "
        "uses a cooperative CUDA launch)",
    )
    add_argument(
        "--scatter-table",
        action="store_true",
        help=(
            "Scatter routed output rows using a random permutation (requires "
            "--epilogue-store bulk_rows or bulk_rows_reduce; works with either scheduler)"
        ),
    )
    add_argument(
        "--n-tile-major-output",
        action="store_true",
        help="Store each output buffer as [ceil(N/tile_n), rows_in_buffer*tile_n]",
    )
    add_argument(
        "--multi-buffer-scatter",
        action="store_true",
        help="Scatter directly to separately allocated output buffers (requires --scatter-table)",
    )
    add_argument(
        "--scatter-output-buffers",
        type=int,
        default=2,
        help="Number of output buffers for --multi-buffer-scatter (default: 2)",
    )
    add_argument(
        "--scatter-output-rows", type=int, nargs="+", metavar="ROWS",
        help="Used rows per N-tile-major buffer; one count broadcasts, otherwise one per buffer. "
        "Counts must sum to --routes (default: balanced split).",
    )
    add_argument(
        "--scatter-output-capacity-rows", type=int, nargs="+", metavar="ROWS",
        help="Allocated row capacity per N-tile-major buffer; one count broadcasts. "
        "Must cover used rows (default: used rows).",
    )
    add_argument(
        "--scatter-table-with-replacement",
        action="store_true",
        help=(
            "Sample scatter destinations with replacement, allowing duplicate and unused rows "
            "(requires --scatter-table and --epilogue-store bulk_rows_reduce)"
        ),
    )
    add_argument(
        "--scatter-table-destination-rows",
        type=int,
        default=None,
        metavar="R1",
        help=(
            "Sample scatter destinations from [0, R1), with 1 <= R1 <= routes "
            "(default: routes; requires --scatter-table-with-replacement). "
            "Output retains routes rows; rows R1: remain zero."
        ),
    )
    add_argument(
        "--epilogue-store",
        choices=("tma", "bulk_rows", "bulk_rows_reduce"),
        default="tma",
        help=(
            "Output I/O: tma uses the existing subtile tensor stores; bulk_rows stages "
            "the full tile in SMEM and issues one linear cp.async.bulk per valid row "
            "(plain GEMM only; supports --pingpong). bulk_rows_reduce uses per-row "
            "reduce-add instead, with output zeroing excluded from timing."
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark QuACK's SM90 grouped GEMM after pre-gathering routed tokens."
    )
    parser.add_argument("--tokens", "-T", type=int, default=4096, help="Number of X rows (T)")
    parser.add_argument("--hidden", "-K", type=int, default=4096, help="Input dimension (K)")
    parser.add_argument(
        "--output-dim",
        "-N",
        type=int,
        default=4096,
        help="Expert intermediate dimension (N)",
    )
    parser.add_argument("--experts", "-E", type=int, default=8, help="Number of experts (E)")
    parser.add_argument("--routes", "-R", type=int, default=8192, help="Routed assignments (R)")
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--activation",
        choices=(*ACTIVATIONS, *GATED_ACTIVATIONS),
        default=None,
        help=(
            "Fuse this activation into the GEMM epilogue. Gated activations such as "
            "swiglu interpret --output-dim as the final width and use a 2x-wide projection."
        ),
    )
    parser.add_argument(
        "--down-projection",
        action="store_true",
        help="Run a grouped down projection after the fused activation (requires --activation)",
    )

    add_kernel_arguments(parser)

    parser.add_argument(
        "--routing-with-replacement",
        action="store_true",
        help="Allow the same token to occur more than once for one expert",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Untimed operation launches")
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Operation launches captured in each timing sample",
    )
    parser.add_argument("--timing-samples", type=int, default=5)
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Time ordinary launches instead of a CUDA graph (less reliable for short kernels)",
    )
    parser.add_argument(
        "--skip-check", action="store_true", help="Skip the float32 reference check"
    )
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args()


def scatter_output_sizes(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    """Resolve logical rows separately from allocation capacities."""
    count = args.scatter_output_buffers
    balanced = [args.routes // count + (i < args.routes % count) for i in range(count)]

    def expand(values, default):
        result = default if values is None else values * count if len(values) == 1 else values
        if len(result) != count or any(value <= 0 for value in result):
            raise ValueError("scatter output rows require one positive count or one per buffer")
        return result

    used = expand(args.scatter_output_rows, balanced)
    capacity = expand(args.scatter_output_capacity_rows, used)
    if sum(used) != args.routes:
        raise ValueError("--scatter-output-rows must sum to --routes")
    if any(rows > allocated for rows, allocated in zip(used, capacity)):
        raise ValueError("scatter output used rows must not exceed allocated capacity")
    return used, capacity


def validate_args(args: argparse.Namespace, *, prepacked: bool = False) -> None:
    named_positive = {
        "tokens": args.tokens,
        "hidden": args.hidden,
        "output_dim": args.output_dim,
        "experts": args.experts,
        "routes": args.routes,
        "tile_m": args.tile_m,
        "tile_n": args.tile_n,
        "cluster_m": args.cluster_m,
        "iterations": args.iterations,
        "timing_samples": args.timing_samples,
    }
    for name, value in named_positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if args.warmup < 0:
        raise ValueError(f"warmup must be nonnegative, got {args.warmup}")
    if args.n_tile_major_output:
        if not args.scatter_table:
            raise ValueError("--n-tile-major-output requires --scatter-table")
        if args.output_dim % 8 or args.tile_n % 8:
            raise ValueError("--n-tile-major-output requires 16-byte-aligned N and tile_n widths")
        if args.down_projection or args.activation is not None:
            raise ValueError("--n-tile-major-output requires plain GEMM without down projection")
    if args.multi_buffer_scatter:
        if not args.scatter_table:
            raise ValueError("--multi-buffer-scatter requires --scatter-table")
        if not 2 <= args.scatter_output_buffers <= args.routes:
            raise ValueError("--scatter-output-buffers must be between 2 and routes")
        scatter_output_sizes(args)
    if args.scatter_output_rows is not None or args.scatter_output_capacity_rows is not None:
        if not args.multi_buffer_scatter or not args.n_tile_major_output:
            raise ValueError("scatter output row options require multi-buffer N-tile-major output")
    if args.scatter_table_destination_rows is not None:
        if not args.scatter_table_with_replacement:
            raise ValueError(
                "--scatter-table-destination-rows requires --scatter-table-with-replacement"
            )
        if not 1 <= args.scatter_table_destination_rows <= args.routes:
            raise ValueError("--scatter-table-destination-rows must be between 1 and routes")
    if args.scatter_table_with_replacement:
        if not args.scatter_table:
            raise ValueError("--scatter-table-with-replacement requires --scatter-table")
        if args.epilogue_store != "bulk_rows_reduce":
            raise ValueError(
                "--scatter-table-with-replacement requires --epilogue-store bulk_rows_reduce"
            )
    if args.bulk_store_rows is not None:
        if args.epilogue_store not in ("bulk_rows", "bulk_rows_reduce"):
            raise ValueError("--bulk-store-rows requires bulk_rows or bulk_rows_reduce")
        if not 1 <= args.bulk_store_rows <= args.tile_m:
            raise ValueError("--bulk-store-rows must be in [1, --tile-m]")
        if args.gather_table or args.scatter_table or args.activation or args.down_projection:
            raise ValueError(
                "--bulk-store-rows requires ordinary pregather without gather-table, "
                "scatter-table, activation, or down-projection"
            )
    if args.bulk_row_swizzle and args.epilogue_store not in ("bulk_rows", "bulk_rows_reduce"):
        raise ValueError("--bulk-row-swizzle requires bulk_rows or bulk_rows_reduce")
    if args.bulk_row_double_buffer and (
        not args.pingpong or args.epilogue_store not in ("bulk_rows", "bulk_rows_reduce")
    ):
        raise ValueError(
            "--bulk-row-double-buffer requires --pingpong and "
            "--epilogue-store bulk_rows or bulk_rows_reduce"
        )
    if args.scatter_table and args.epilogue_store not in ("bulk_rows", "bulk_rows_reduce"):
        raise ValueError("--scatter-table requires --epilogue-store bulk_rows or bulk_rows_reduce")
    if args.gather_table_n_group_major and not args.gather_table:
        raise ValueError("--gather-table-n-group-major requires --gather-table")
    if args.gather_table_wave_barrier and (
        not args.gather_table_expert_waves
        or not args.pingpong
        or args.epilogue_store not in ("bulk_rows", "bulk_rows_reduce")
    ):
        raise ValueError(
            "--gather-table-wave-barrier requires --gather-table-expert-waves, "
            "--pingpong, and --epilogue-store bulk_rows or bulk_rows_reduce"
        )
    if args.gather_table_expert_waves and not args.gather_table:
        raise ValueError("--gather-table-expert-waves requires --gather-table")
    if args.gather_table:
        if args.max_swizzle_size <= 0:
            raise ValueError("--gather-table requires positive --max-swizzle-size")
        gemm_output_dim = args.output_dim * (2 if args.activation in GATED_ACTIVATIONS else 1)
        clusters_n = math.ceil(gemm_output_dim / args.tile_n)
        x = min(args.max_swizzle_size, clusters_n)
        if clusters_n % x:
            raise ValueError(
                f"table expansion requires clusters_n % x == 0, got {clusters_n} % {x}"
            )
    if args.down_projection and args.activation is None:
        raise ValueError("--down-projection requires --activation")
    if args.epilogue_store in ("bulk_rows", "bulk_rows_reduce") and args.activation is not None:
        raise ValueError(
            f"--epilogue-store {args.epilogue_store} requires --activation to be omitted"
        )
    if prepacked:
        return
    if args.routes % args.experts != 0:
        raise ValueError(f"routes ({args.routes}) must be divisible by experts ({args.experts})")
    routes_per_expert = args.routes // args.experts
    if not args.routing_with_replacement and routes_per_expert > args.tokens:
        raise ValueError(
            "routes/experts cannot exceed tokens when sampling without replacement: "
            f"{routes_per_expert} > {args.tokens}. Pass --routing-with-replacement if intended."
        )


@torch.inference_mode()
def prepare_inputs(args: argparse.Namespace, device: torch.device) -> PregatherInputs:
    """Create routing inputs and materialize the expert-major A tensor on CUDA."""
    dtype = DTYPES[args.dtype]
    routes_per_expert = args.routes // args.experts
    X = torch.randn((args.tokens, args.hidden), dtype=dtype, device=device)
    gated = args.activation in GATED_ACTIVATIONS
    gemm_output_dim = args.output_dim * (2 if gated else 1)
    if gated:
        # concat_layout=("B",) acts on B's non-contiguous dimension. Materialize
        # the conventional [E, 2N, K] [gate | up] weight and expose its
        # [E, K, 2N] transpose so QuACK interleaves output columns, not K rows.
        W_up = torch.randn(
            (args.experts, gemm_output_dim, args.hidden), dtype=dtype, device=device
        ).transpose(1, 2)
    else:
        W_up = torch.randn(
            (args.experts, args.hidden, gemm_output_dim), dtype=dtype, device=device
        )
    W_up.mul_(1.0 / math.sqrt(args.hidden))

    W_down = None
    if args.down_projection:
        W_down = torch.randn(
            (args.experts, args.output_dim, args.hidden), dtype=dtype, device=device
        )
        W_down.mul_(1.0 / math.sqrt(args.output_dim))

    if args.routing_with_replacement:
        A_idx = torch.randint(
            args.tokens,
            (args.routes,),
            dtype=torch.int32,
            device=device,
        )
    else:
        A_idx = torch.empty(args.routes, dtype=torch.int32, device=device)
        routes_2d = A_idx.view(args.experts, routes_per_expert)
        for expert in range(args.experts):
            routes_2d[expert].copy_(
                torch.randperm(args.tokens, dtype=torch.int32, device=device)[:routes_per_expert]
            )

    # This is the only token-gather operation. Advanced indexing launches a
    # CUDA kernel and returns a new contiguous [R, K] tensor in expert-major
    # route order. It runs before all GEMM warmup and timing regions.
    A = X[A_idx]
    assert A.is_contiguous()
    inputs = prepare_grouped_inputs(args, A, W_up, [routes_per_expert] * args.experts, W_down)
    inputs.X, inputs.A_idx = X, A_idx
    return inputs


@torch.inference_mode()
def prepare_grouped_inputs(
    args: argparse.Namespace,
    A: torch.Tensor,
    W_up: torch.Tensor,
    counts: list[int],
    W_down: torch.Tensor | None = None,
    *,
    output_buffers: tuple[torch.Tensor, ...] | None = None,
    scatter_mapping: torch.Tensor | None = None,
) -> PregatherInputs:
    """Prepare output and scheduling metadata around an existing expert-major input."""
    if len(counts) != args.experts or sum(counts) != A.shape[0] or any(c < 0 for c in counts):
        raise ValueError("expert counts must partition the input rows")
    if A.shape != (args.routes, args.hidden) or W_up.shape != (
        args.experts, args.hidden,
        args.output_dim * (2 if args.activation in GATED_ACTIVATIONS else 1),
    ):
        raise ValueError("grouped operands do not match the configured GEMM dimensions")
    device, dtype = A.device, A.dtype
    gemm_output_dim = W_up.shape[-1]
    cu_seqlens_m = torch.tensor([0, *counts], dtype=torch.int32, device=device).cumsum(
        0, dtype=torch.int32
    )
    wave_work_ids = None
    if args.gather_table_expert_waves:
        gemm_output_dim = args.output_dim * (2 if args.activation in GATED_ACTIVATIONS else 1)
        clusters_n = math.ceil(gemm_output_dim / args.tile_n)
        x = min(args.max_swizzle_size, clusters_n)
        num_work = (
            sum(math.ceil(count / (args.tile_m * args.cluster_m)) for count in counts)
            * clusters_n
        )
        # Match GatherTableTileScheduler.get_grid_shape and GEMM's capacity query.
        wave_work_ids = min(get_max_active_clusters(args.cluster_m), num_work)
        if wave_work_ids <= 0 or wave_work_ids % x:
            raise ValueError(
                "--gather-table-expert-waves requires wave_work_ids % "
                "effective_swizzle_size == 0, "
                f"got {wave_work_ids} % {x}; choose a compatible --max-swizzle-size"
            )
    gather_work_table = None
    identity_A_idx = None
    if args.gather_table:
        # Support both direct script execution and python -m run.hopper_pregather_gemm.
        if __package__:
            from .hopper_gather_table_gemm import build_work_table
        else:
            from hopper_gather_table_gemm import build_work_table

        gather_work_table, _, _ = build_work_table(
            counts,
            output_dim=gemm_output_dim,
            tile_m=args.tile_m,
            tile_n=args.tile_n,
            cluster_m=args.cluster_m,
            max_swizzle_size=args.max_swizzle_size,
            device=device,
            n_group_major=args.gather_table_n_group_major,
            wave_work_ids=wave_work_ids,
        )
        # Preserve the original routing indices separately from the kernel indices.
        identity_A_idx = torch.arange(args.routes, dtype=torch.int32, device=A.device)
    if output_buffers is not None:
        if not args.multi_buffer_scatter or scatter_mapping is None or args.down_projection:
            raise ValueError("external outputs require mapped multi-buffer plain GEMM")
        sizes = [
            b.shape[1] // args.tile_n if args.n_tile_major_output else b.shape[0]
            for b in output_buffers
        ]
        up_output = output_buffers
    elif args.multi_buffer_scatter:
        # Split logical destination space evenly; no monolithic output allocation.
        sizes, capacities = scatter_output_sizes(args)
        up_output = tuple(
            torch.empty(
                (math.ceil(args.output_dim / args.tile_n), size * args.tile_n)
                if args.n_tile_major_output
                else (size, args.output_dim),
                dtype=dtype,
                device=device,
            )
            for size in capacities
        )
    elif args.n_tile_major_output:
        up_output = torch.empty(
            (math.ceil(args.output_dim / args.tile_n), args.routes * args.tile_n),
            dtype=dtype,
            device=device,
        )
    else:
        up_output = torch.empty((args.routes, args.output_dim), dtype=dtype, device=device)
    output = (
        torch.empty((args.routes, args.hidden), dtype=dtype, device=device)
        if args.down_projection
        else up_output
    )
    scatter_table = None
    if args.scatter_table and output_buffers is None:
        scatter_table = (
            torch.randint(
                args.scatter_table_destination_rows
                if args.scatter_table_destination_rows is not None
                else args.routes,
                (args.routes,),
                dtype=torch.int32,
                device=device,
            )
            if args.scatter_table_with_replacement
            else torch.randperm(args.routes, dtype=torch.int32, device=device)
        )
    scatter_row_addresses = None
    if args.multi_buffer_scatter:
        offsets = torch.tensor([0, *sizes], device=device, dtype=torch.int64).cumsum(0)
        if scatter_mapping is None:
            destinations = scatter_table.long()
            buffer_ids = torch.bucketize(destinations, offsets[1:], right=True)
            scatter_mapping = torch.stack((buffer_ids, destinations - offsets[buffer_ids]), dim=1)
        else:
            # Flatten token destinations only for the numerical reference.
            scatter_table = offsets[scatter_mapping[:, 0].long()] + scatter_mapping[:, 1]
        scatter_row_addresses = build_scatter_row_addresses(
            up_output,
            scatter_mapping,
            reduce=args.epilogue_store == "bulk_rows_reduce",
            tile_n=args.tile_n if args.n_tile_major_output else None,
            used_rows=sizes if args.n_tile_major_output else None,
        )
    return PregatherInputs(
        n_tile_major_tile_n=args.tile_n if args.n_tile_major_output else None,
        scatter_output_rows=tuple(sizes) if args.multi_buffer_scatter else None,
        scatter_table=scatter_table,
        scatter_mapping=scatter_mapping,
        scatter_row_addresses=scatter_row_addresses,
        scatter_with_replacement=args.scatter_table_with_replacement,
        X=A,
        W_up=W_up,
        W_down=W_down,
        A_idx=torch.arange(args.routes, dtype=torch.int32, device=device),
        cu_seqlens_m=cu_seqlens_m,
        A=A,
        up_output=up_output,
        output=output,
        gather_work_table=gather_work_table,
        identity_A_idx=identity_A_idx,
        wave_work_ids=wave_work_ids,
    )


def make_launch(args: argparse.Namespace, inputs: PregatherInputs):
    # QuACK's plain low-level API expects B as [E, N, K]. This is a zero-copy
    # view of each user-facing [E, input, output] weight tensor. The activation
    # API below uses the torch-convention W_up view directly.
    B_up = inputs.W_up.transpose(1, 2)
    B_down = inputs.W_down.transpose(1, 2) if inputs.W_down is not None else None
    up_cu_seqlens_m = inputs.cu_seqlens_m
    up_A_idx = None
    table_kwargs = {}
    if args.gather_table:
        assert inputs.gather_work_table is not None and inputs.identity_A_idx is not None
        up_cu_seqlens_m = None
        up_A_idx = inputs.identity_A_idx
        table_kwargs = dict(gather_work_table=inputs.gather_work_table, multi_buffer_gather=False)
        if args.gather_table_wave_barrier:
            # Private to this launch closure; quack_gemm resets it on its stream
            # for every invocation, including invocations captured in a graph.
            table_kwargs["wave_store_counter"] = torch.empty(
                (1,), dtype=torch.int32, device=inputs.A.device
            )

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

    def launch() -> None:
        if args.activation is not None:
            gemm_act(
                inputs.A,
                inputs.W_up,
                activation=args.activation,
                postact_out=inputs.up_output,
                cu_seqlens_m=up_cu_seqlens_m,
                A_idx=up_A_idx,
                **table_kwargs,
                store_preact=False,
                dynamic_scheduler=False,
                tuned=False,
                config=config,
                concat_layout=("B",) if args.activation in GATED_ACTIVATIONS else None,
            )
        else:
            quack_gemm(
                inputs.A,
                B_up,
                inputs.up_output,
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
                cu_seqlens_m=up_cu_seqlens_m,
                # Default: ordinary TMA. Table mode: cp.async with identity indices.
                A_idx=up_A_idx,
                **table_kwargs,
                use_tma_gather=False,
                epilogue_store=args.epilogue_store,
                bulk_row_double_buffer=args.bulk_row_double_buffer,
                bulk_row_swizzle=args.bulk_row_swizzle,
                bulk_store_rows=args.bulk_store_rows,
                scatter_table=(
                    inputs.scatter_row_addresses
                    if args.multi_buffer_scatter
                    else inputs.scatter_table
                ),
                multi_buffer_scatter=args.multi_buffer_scatter,
                n_tile_major_output=args.n_tile_major_output,
            )

        if args.down_projection:
            assert B_down is not None
            quack_gemm(
                inputs.up_output,
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
                # The activated rows are already expert-contiguous.
                A_idx=None,
                use_tma_gather=False,
                epilogue_store=args.epilogue_store,
                bulk_row_double_buffer=args.bulk_row_double_buffer,
                bulk_row_swizzle=args.bulk_row_swizzle,
                bulk_store_rows=args.bulk_store_rows,
            )

    return launch


@torch.inference_mode()
def benchmark(
    launch,
    *,
    iterations: int,
    samples: int,
    use_cuda_graph: bool,
    reset_output=None,
) -> list[float]:
    """Return per-operation milliseconds, excluding optional output resets.

    With output resets, timing uses one event pair per launch. External events become
    actual record nodes in a CUDA graph, so their timestamps update on replay.
    Zeroing precedes the start event on the same stream, outside the interval.
    Ordinary-launch timing can still include host enqueue gaps; prefer graphs.
    """
    if reset_output is not None:
        events = [
            (
                torch.cuda.Event(enable_timing=True, external=use_cuda_graph),
                torch.cuda.Event(enable_timing=True, external=use_cuda_graph),
            )
            for _ in range(iterations)
        ]

        def measured_launches():
            for start, end in events:
                reset_output()
                start.record()
                launch()
                end.record()

        graph = None
        if use_cuda_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                measured_launches()
            graph.replay()
            torch.cuda.synchronize()

        timings_ms = []
        for _ in range(samples):
            if graph is not None:
                graph.replay()
            else:
                measured_launches()
            events[-1][1].synchronize()
            timings_ms.append(sum(start.elapsed_time(end) for start, end in events) / iterations)
        return timings_ms

    graph = None
    if use_cuda_graph:
        # One graph contains many back-to-back kernel nodes. This removes the
        # host enqueue gaps that distort CUDA-event timings for short kernels.
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
def check_scatter_reduction(
    inputs: PregatherInputs, *, atol: float, rtol: float
) -> tuple[float, float, None, None]:
    """Check all destinations, accounting for output-dtype reduction rounding.

    Contributions are rounded before reduction. For c contributions, repeated
    rounding has the conservative bound ((1 + u)**(c - 1) - 1) * sum(abs(y)),
    with u = eps/2. This is independent of the hardware's addition order.
    The same-dtype GEMM baseline separately accounts for GEMM rounding error.
    All reference work runs outside timing/capture.
    """
    assert inputs.scatter_table is not None
    indices = inputs.scatter_table.long()
    output = inputs.collected_up_output()
    reference = torch.zeros_like(output, dtype=torch.float32)
    magnitude = torch.zeros_like(reference)
    gemm_error = torch.zeros_like(reference)
    offsets = inputs.cu_seqlens_m.tolist()
    for expert, (start, end) in enumerate(zip(offsets, offsets[1:])):
        a, w = inputs.A[start:end], inputs.W_up[expert]
        values = a.float() @ w.float()
        baseline = (a @ w).to(output.dtype).float()
        destinations = indices[start:end]
        reference.index_add_(0, destinations, values)
        magnitude.index_add_(0, destinations, baseline.abs())
        gemm_error.index_add_(0, destinations, (baseline - values).abs())

    counts = torch.bincount(indices, minlength=output.shape[0])
    unit_roundoff = torch.finfo(output.dtype).eps / 2
    reduction_factor = torch.expm1(
        (counts.float() - 1).clamp_min(0) * math.log1p(unit_roundoff)
    )[:, None]
    # Include GEMM error in the magnitude bound for the actual contributions.
    allowed = (
        atol
        + rtol * reference.abs()
        + 2 * gemm_error
        + reduction_factor * (magnitude + 2 * gemm_error)
    )
    if not torch.isfinite(output).all():
        raise AssertionError("scatter reduction output contains NaN or infinity")
    unused = counts == 0
    torch.testing.assert_close(output[unused], torch.zeros_like(output[unused]), atol=0, rtol=0)
    error = (output.float() - reference).abs()
    if (error > allowed).any():
        raise AssertionError(
            "scatter reduction exceeds its per-element rounding bound: "
            f"max error={error.max().item():.6g}, "
            f"max excess={(error - allowed).max().item():.6g}"
        )
    return error.max().item(), allowed.max().item(), None, None


@torch.inference_mode()
def check_correctness(
    inputs: PregatherInputs,
    *,
    activation: str | None,
    atol: float,
    rtol: float,
    bulk_store_rows: int | None = None,
    tile_m: int = 128,
) -> tuple[float, float, float | None, float | None]:
    """Check every enabled GEMM output against per-expert float32 references."""
    if inputs.scatter_with_replacement:
        assert activation is None and inputs.W_down is None
        return check_scatter_reduction(inputs, atol=atol, rtol=rtol)
    up_output = inputs.collected_up_output()
    offsets = inputs.cu_seqlens_m.tolist()
    max_up_error = 0.0
    max_up_allowed = 0.0
    max_down_error = 0.0 if inputs.W_down is not None else None
    max_down_allowed = 0.0 if inputs.W_down is not None else None
    for expert, (start, end) in enumerate(zip(offsets, offsets[1:])):
        if start == end:
            continue
        A_expert = inputs.A[start:end]
        W_up_expert = inputs.W_up[expert]
        up_reference = A_expert.float() @ W_up_expert.float()
        if activation in GATED_ACTIVATIONS:
            gate, up = up_reference.chunk(2, dim=-1)
            up_reference = gated_to_pytorch_fn_map[activation](gate, up)
        elif activation is not None:
            up_reference = act_to_pytorch_fn_map[activation](up_reference)

        up_baseline = A_expert @ W_up_expert
        if activation in GATED_ACTIVATIONS:
            gate, up = up_baseline.chunk(2, dim=-1)
            up_baseline = gated_to_pytorch_fn_map[activation](gate, up)
        elif activation is not None:
            up_baseline = act_to_pytorch_fn_map[activation](up_baseline)

        up_actual = (
            up_output[start:end]
            if inputs.scatter_table is None
            else up_output[inputs.scatter_table[start:end].long()]
        )
        if bulk_store_rows is not None:
            omitted = torch.arange(end - start, device=up_actual.device) % tile_m >= bulk_store_rows
            up_reference[omitted] = 0
            up_baseline[omitted] = 0
            # Require untouched rows to remain exactly zero, independent of GEMM tolerances.
            torch.testing.assert_close(
                up_actual[omitted], torch.zeros_like(up_actual[omitted]), atol=0, rtol=0
            )
        if not torch.isfinite(up_actual).all():
            raise AssertionError(f"expert {expert} up output contains NaN or infinity")
        up_error = (up_actual.float() - up_reference).abs().max().item()
        max_up_error = max(max_up_error, up_error)

        if activation is None:
            reference_out = up_reference.to(up_actual.dtype)
            torch.testing.assert_close(up_actual, reference_out, atol=atol, rtol=rtol)
            allowed_error = atol + rtol * reference_out.float().abs().max().item()
        else:
            # QuACK's activation tests bound the fused kernel against the error
            # of an equivalent same-dtype PyTorch computation.
            baseline_error = (up_baseline.float() - up_reference).abs().max().item()
            fixed_error = atol + rtol * up_reference.abs().max().item()
            allowed_error = max(fixed_error, 2 * baseline_error + 1e-5)
            if up_error > allowed_error:
                raise AssertionError(
                    f"expert {expert} activation output error {up_error:.6g} exceeds "
                    f"the permitted bound {allowed_error:.6g} "
                    f"(same-dtype PyTorch baseline error {baseline_error:.6g})"
                )
        max_up_allowed = max(max_up_allowed, allowed_error)

        if inputs.W_down is not None:
            W_down_expert = inputs.W_down[expert]
            down_reference = up_actual.float() @ W_down_expert.float()
            down_baseline = up_actual @ W_down_expert
            down_actual = inputs.output[start:end]
            if not torch.isfinite(down_actual).all():
                raise AssertionError(f"expert {expert} down output contains NaN or infinity")
            down_error = (down_actual.float() - down_reference).abs().max().item()
            down_baseline_error = (down_baseline.float() - down_reference).abs().max().item()
            down_fixed_error = atol + rtol * down_reference.abs().max().item()
            down_allowed = max(down_fixed_error, 2 * down_baseline_error + 1e-5)
            if down_error > down_allowed:
                raise AssertionError(
                    f"expert {expert} down output error {down_error:.6g} exceeds "
                    f"the permitted bound {down_allowed:.6g} "
                    f"(same-dtype PyTorch down-GEMM baseline error {down_baseline_error:.6g})"
                )
            assert max_down_error is not None and max_down_allowed is not None
            max_down_error = max(max_down_error, down_error)
            max_down_allowed = max(max_down_allowed, down_allowed)

    return max_up_error, max_up_allowed, max_down_error, max_down_allowed


def gib(nbytes: int) -> float:
    return nbytes / (1024**3)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 9:
        raise RuntimeError(
            "This runner targets Hopper (SM90), but "
            f"device {args.device} has capability {capability}"
        )

    torch.manual_seed(args.seed)
    inputs = prepare_inputs(args, device)
    torch.cuda.synchronize(device)

    allocated_tensors = [
        inputs.X,
        inputs.W_up,
        inputs.A_idx,
        inputs.cu_seqlens_m,
        inputs.A,
        *inputs.output_buffers(),
    ]
    if inputs.W_down is not None:
        allocated_tensors.extend((inputs.W_down, inputs.output))
    if inputs.gather_work_table is not None:
        allocated_tensors.extend((inputs.gather_work_table, inputs.identity_A_idx))
    if inputs.scatter_table is not None:
        allocated_tensors.append(inputs.scatter_table)
    if inputs.scatter_row_addresses is not None:
        allocated_tensors.extend((inputs.scatter_row_addresses, inputs.scatter_mapping))
    tensor_bytes = sum(tensor.numel() * tensor.element_size() for tensor in allocated_tensors)
    routes_per_expert = args.routes // args.experts
    print(f"Device: {torch.cuda.get_device_name(device)} (SM{capability[0]}{capability[1]})")
    print(f"X: {tuple(inputs.X.shape)}, W_up: {tuple(inputs.W_up.shape)}, dtype: {inputs.X.dtype}")
    print(f"Pre-gathered A: {tuple(inputs.A.shape)}, contiguous={inputs.A.is_contiguous()}")
    if inputs.W_down is not None:
        print(
            f"W_down: {tuple(inputs.W_down.shape)}, up output: {tuple(inputs.up_output.shape)}, "
            f"final output: {tuple(inputs.output.shape)}"
        )
    elif args.n_tile_major_output:
        print(
            f"Output: N-tile-major physical {[tuple(b.shape) for b in inputs.output_buffers()]}, "
            f"logical ({args.routes}, {args.output_dim}), tile_n={args.tile_n}"
        )
        if args.multi_buffer_scatter:
            print(f"Output used rows: {list(inputs.scatter_output_rows)}")
    elif args.multi_buffer_scatter:
        print(f"Output buffers: {[tuple(b.shape) for b in inputs.output_buffers()]}")
    else:
        print(f"Output: {tuple(inputs.output.shape)}, down projection: disabled")
    print(
        f"Routes: {args.routes} total, {routes_per_expert} per expert, "
        f"sampling {'with' if args.routing_with_replacement else 'without'} replacement"
    )
    gather_description = (
        "table (cp.async A load, identity indices)"
        if args.gather_table
        else "none (TMA A load)"
    )
    print(
        f"Kernel: tile=({args.tile_m}, {args.tile_n}, {args.tile_k or 'auto'}), "
        f"cluster=({args.cluster_m}, 1, 1), persistent=True, gather={gather_description}"
    )
    if inputs.gather_work_table is not None:
        print(f"Gather work table: {tuple(inputs.gather_work_table.shape)}")
        table_order = (
            "N-group-major, expert waves"
            if inputs.wave_work_ids is not None
            else "N-group-major, global serpentine"
            if args.gather_table_n_group_major
            else "expert-major, per-expert serpentine"
        )
        print(f"Gather table order: {table_order}")
        if inputs.wave_work_ids is not None:
            print(
                f"Logical wave: {inputs.wave_work_ids} work IDs; "
                "experts share a wave only on exhaustion"
            )
    print(f"Fused activation: {args.activation or 'disabled'}")
    print(f"Epilogue store: {args.epilogue_store}, pingpong: {args.pingpong}")
    if args.bulk_store_rows is not None:
        print(
            f"Diagnostic partial stores: first {args.bulk_store_rows}/{args.tile_m} rows per tile; "
            "remaining rows stay zero. Throughput counts full GEMM FLOPs."
        )
    if args.bulk_row_swizzle:
        print("Bulk-row swizzle: enabled (TMA-style SMEM, 16-byte bulk stores)")
    if args.bulk_row_double_buffer:
        print("Bulk-row double buffer: enabled (one full output tile per warp group)")
    barrier_description = (
        "enabled (cooperative launch)" if args.gather_table_wave_barrier else "disabled"
    )
    print(f"Wave store barrier: {barrier_description}")
    if inputs.scatter_table is not None:
        scatter_description = (
            "random destinations sampled with replacement"
            if inputs.scatter_with_replacement
            else "random permutation"
        )
        print(f"Scatter table: {scatter_description}, {args.routes} routed output rows")
        if inputs.scatter_with_replacement:
            destination_rows = (
                args.scatter_table_destination_rows
                if args.scatter_table_destination_rows is not None
                else args.routes
            )
            print(
                f"Scatter destination range: [0, {destination_rows}), "
                f"{args.routes / destination_rows:.3f} mean contributions per eligible row"
            )
    print(f"Approximate tensor storage: {gib(tensor_bytes):.3f} GiB")
    compile_target = "up and down kernels" if args.down_projection else "kernel"
    print(f"Compiling and warming up the specialized QuACK {compile_target}...")

    launch = make_launch(args, inputs)
    # Match memory preparation and event placement across all epilogue-store modes.
    reset_output = inputs.zero_up_output
    for _ in range(max(1, args.warmup)):
        reset_output()
        launch()
    torch.cuda.synchronize(device)

    timings_ms = benchmark(
        launch,
        iterations=args.iterations,
        samples=args.timing_samples,
        use_cuda_graph=not args.no_cuda_graph,
        reset_output=reset_output,
    )
    median_ms = statistics.median(timings_ms)
    up_flops = 2 * args.routes * args.hidden * inputs.W_up.shape[-1]
    down_flops = 2 * args.routes * args.output_dim * args.hidden if args.down_projection else 0
    flops = up_flops + down_flops
    tflops = flops / (median_ms * 1e9)
    timing_kind = (
        "CUDA graph + per-GEMM events" if not args.no_cuda_graph else "per-GEMM CUDA events"
    )
    timing_kind += " (output zeroing excluded)"
    formatted_samples = ", ".join(f"{value:.4f}" for value in timings_ms)
    print(f"Timing method: {timing_kind}")
    operation_name = "two-GEMM MLP" if args.down_projection else "up GEMM"
    print(f"Per-operation samples (ms): [{formatted_samples}]")
    print(f"Median {operation_name} time: {median_ms:.4f} ms")
    print(f"Effective throughput: {tflops:.2f} TFLOP/s")

    if not args.skip_check:
        max_up_error, max_up_allowed, max_down_error, max_down_allowed = check_correctness(
            inputs,
            activation=args.activation,
            atol=args.atol,
            rtol=args.rtol,
            bulk_store_rows=args.bulk_store_rows,
            tile_m=args.tile_m,
        )
        if max_down_error is not None and max_down_allowed is not None:
            print(
                "Reference check: PASSED "
                f"(up max error {max_up_error:.6g}, permitted {max_up_allowed:.6g}; "
                f"down max error {max_down_error:.6g}, permitted {max_down_allowed:.6g})"
            )
        else:
            print(
                "Reference check: PASSED "
                f"(max error {max_up_error:.6g}, permitted {max_up_allowed:.6g})"
            )


if __name__ == "__main__":
    main()
