#!/usr/bin/env python3
"""Prepare and benchmark a QuACK Hopper GEMM with fused token gather.

The user-facing tensors follow the MoE notation

    X:             [T, K]
    W_up:          [E, K, N] (or [E, K, 2N] for a gated activation)
    W_down:        [E, N, K] (only with --down-projection)
    A_idx:         [R]
    cu_seqlens_m:  [E + 1]
    up_output:     [R, N]
    output:        [R, N], or [R, K] with --down-projection

``A_idx[cu_seqlens_m[e]:cu_seqlens_m[e + 1]]`` contains the token rows used
with ``W_up[e]``. When ``--down-projection`` is enabled, the fused gather
up-projection writes rows in expert-contiguous order and the down-projection
reuses ``cu_seqlens_m`` without gathering A again. All input tensors, including
the routing metadata, are created directly on the selected CUDA device.
QuACK's low-level API takes weights as [E, N, K], so zero-copy transposed views
of the user-facing weights are passed to it.

Example:

    python run/hopper_gather_gemm.py \
        --tokens 4096 --hidden 4096 --output-dim 4096 \
        --experts 8 --routes 8192 --warmup 5 --iterations 100

Fuse a SwiGLU up-projection (the weight uses concatenated [gate | up] columns):

    python run/hopper_gather_gemm.py \
        --tokens 4096 --hidden 4096 --output-dim 14336 \
        --experts 8 --routes 8192 --activation swiglu --down-projection

The first call may take a while because QuACK compiles the specialized kernels.
Compilation, warmup, graph capture, and validation are not timed. With
``--down-projection``, the reported time covers the two-GEMM expert MLP;
route-weighted scatter back to token order is outside this runner.
"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass

import torch

from quack.gemm import gemm as quack_gemm
from quack.gemm_config import GemmConfig
from quack.gemm_interface import act_to_pytorch_fn_map, gated_to_pytorch_fn_map, gemm_act


DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

ACTIVATIONS = tuple(name for name in act_to_pytorch_fn_map if name is not None)
GATED_ACTIVATIONS = tuple(gated_to_pytorch_fn_map)


@dataclass
class GatherInputs:
    X: torch.Tensor
    W_up: torch.Tensor
    W_down: torch.Tensor | None
    A_idx: torch.Tensor
    cu_seqlens_m: torch.Tensor
    up_output: torch.Tensor
    output: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark QuACK's SM90 GEMM with fused gather-A and an optional "
            "down projection for uniform MoE routing."
        )
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

    parser.add_argument("--tile-m", type=int, default=256)
    parser.add_argument("--tile-n", type=int, default=256)
    parser.add_argument(
        "--tile-k",
        type=int,
        default=None,
        help="Optional K tile; by default QuACK chooses it from the operand dtype",
    )
    parser.add_argument("--cluster-m", type=int, default=2)
    parser.add_argument("--max-swizzle-size", type=int, default=8)
    parser.add_argument("--pingpong", action="store_true")

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


def validate_args(args: argparse.Namespace) -> None:
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
    if args.down_projection and args.activation is None:
        raise ValueError("--down-projection requires --activation")
    if args.routes % args.experts != 0:
        raise ValueError(f"routes ({args.routes}) must be divisible by experts ({args.experts})")
    routes_per_expert = args.routes // args.experts
    if not args.routing_with_replacement and routes_per_expert > args.tokens:
        raise ValueError(
            "routes/experts cannot exceed tokens when sampling without replacement: "
            f"{routes_per_expert} > {args.tokens}. Pass --routing-with-replacement if intended."
        )


@torch.inference_mode()
def prepare_inputs(args: argparse.Namespace, device: torch.device) -> GatherInputs:
    """Allocate and initialize X, expert weights, routing metadata, and outputs in GPU HBM."""
    dtype = DTYPES[args.dtype]
    routes_per_expert = args.routes // args.experts

    # torch.randn on a CUDA device initializes these tensors with CUDA kernels;
    # no full-size input is staged through host memory.
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
        # User-facing down weights are [E, N, K], mirroring the conventional
        # linear-layer layout used for W_up. The low-level launch below transposes
        # this to QuACK's [E, K, N] B convention without copying.
        W_down = torch.randn(
            (args.experts, args.output_dim, args.hidden), dtype=dtype, device=device
        )
        W_down.mul_(1.0 / math.sqrt(args.output_dim))

    cu_seqlens_m = torch.arange(args.experts + 1, dtype=torch.int32, device=device)
    cu_seqlens_m.mul_(routes_per_expert)

    if args.routing_with_replacement:
        A_idx = torch.randint(
            args.tokens,
            (args.routes,),
            dtype=torch.int32,
            device=device,
        )
    else:
        # Each expert independently samples distinct token IDs. A token may still
        # appear for different experts, as it can in top-k MoE routing.
        A_idx = torch.empty(args.routes, dtype=torch.int32, device=device)
        routes_2d = A_idx.view(args.experts, routes_per_expert)
        for expert in range(args.experts):
            routes_2d[expert].copy_(
                torch.randperm(args.tokens, dtype=torch.int32, device=device)[:routes_per_expert]
            )

    up_output = torch.empty((args.routes, args.output_dim), dtype=dtype, device=device)
    output = (
        torch.empty((args.routes, args.hidden), dtype=dtype, device=device)
        if args.down_projection
        else up_output
    )
    return GatherInputs(
        X=X,
        W_up=W_up,
        W_down=W_down,
        A_idx=A_idx,
        cu_seqlens_m=cu_seqlens_m,
        up_output=up_output,
        output=output,
    )


def make_launch(args: argparse.Namespace, inputs: GatherInputs):
    # QuACK's low-level B convention is [E, N, K]. These are only views: the
    # allocations retained in GatherInputs use conventional [E, input, output] weights.
    B_up = inputs.W_up.transpose(1, 2)
    B_down = inputs.W_down.transpose(1, 2) if inputs.W_down is not None else None

    if args.activation is not None:
        config = GemmConfig(
            tile_m=args.tile_m,
            tile_n=args.tile_n,
            tile_k=args.tile_k,
            pingpong=args.pingpong,
            is_dynamic_persistent=False,
            cluster_m=args.cluster_m,
            cluster_n=1,  # Required by gather_A.
            cluster_k=1,
            max_swizzle_size=args.max_swizzle_size,
            device_capacity=9,
            use_tma_gather=False,
        )

    def launch() -> None:
        if args.activation is not None:
            gemm_act(
                inputs.X,
                inputs.W_up,
                activation=args.activation,
                postact_out=inputs.up_output,
                cu_seqlens_m=inputs.cu_seqlens_m,
                A_idx=inputs.A_idx,
                store_preact=False,
                dynamic_scheduler=False,
                tuned=False,
                config=config,
                # Use the common [all gate | all up] weight layout. QuACK
                # performs the logical interleave inside the fused kernel.
                concat_layout=("B",) if args.activation in GATED_ACTIVATIONS else None,
            )
        else:
            quack_gemm(
                inputs.X,
                B_up,
                inputs.up_output,
                C=None,
                tile_count_semaphore=None,
                tile_M=args.tile_m,
                tile_N=args.tile_n,
                tile_K=args.tile_k,
                cluster_M=args.cluster_m,
                cluster_N=1,  # Required by gather_A.
                cluster_K=1,
                pingpong=args.pingpong,
                persistent=True,
                is_dynamic_persistent=False,
                max_swizzle_size=args.max_swizzle_size,
                cu_seqlens_m=inputs.cu_seqlens_m,
                A_idx=inputs.A_idx,
                # Hopper gather-A uses QuACK's cp.async path. TMA gather4 is an
                # SM100/SM110-only implementation in this repository.
                use_tma_gather=False,
            )

        if args.down_projection:
            assert B_down is not None
            # up_output is already partitioned into expert-contiguous ranges,
            # so this is a varlen grouped GEMM but not a gather-A GEMM. Passing
            # A_idx here would index route rows using original token IDs.
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
                A_idx=None,
                use_tma_gather=False,
            )

    return launch


@torch.inference_mode()
def benchmark(
    launch,
    *,
    iterations: int,
    samples: int,
    use_cuda_graph: bool,
) -> list[float]:
    """Return per-operation milliseconds for each timing sample."""
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
def check_correctness(
    inputs: GatherInputs,
    *,
    activation: str | None,
    atol: float,
    rtol: float,
) -> tuple[float, float, float | None, float | None]:
    """Check every enabled GEMM output against per-expert float32 references."""
    num_experts = inputs.W_up.shape[0]
    routes_per_expert = inputs.A_idx.numel() // num_experts
    max_up_error = 0.0
    max_up_allowed = 0.0
    max_down_error = 0.0 if inputs.W_down is not None else None
    max_down_allowed = 0.0 if inputs.W_down is not None else None
    for expert in range(num_experts):
        start = expert * routes_per_expert
        end = start + routes_per_expert
        token_idx = inputs.A_idx[start:end]
        X_expert = inputs.X[token_idx]
        W_up_expert = inputs.W_up[expert]

        up_reference = X_expert.float() @ W_up_expert.float()
        if activation in GATED_ACTIVATIONS:
            gate, up = up_reference.chunk(2, dim=-1)
            up_reference = gated_to_pytorch_fn_map[activation](gate, up)
        elif activation is not None:
            up_reference = act_to_pytorch_fn_map[activation](up_reference)

        up_baseline = X_expert @ W_up_expert
        if activation in GATED_ACTIVATIONS:
            gate, up = up_baseline.chunk(2, dim=-1)
            up_baseline = gated_to_pytorch_fn_map[activation](gate, up)
        elif activation is not None:
            up_baseline = act_to_pytorch_fn_map[activation](up_baseline)

        up_actual = inputs.up_output[start:end]
        if not torch.isfinite(up_actual).all():
            raise AssertionError(f"expert {expert} up output contains NaN or infinity")
        up_error = (up_actual.float() - up_reference).abs().max().item()
        max_up_error = max(max_up_error, up_error)

        if activation is None:
            # Preserve the original plain-GEMM elementwise check.
            reference_out = up_reference.to(up_actual.dtype)
            torch.testing.assert_close(up_actual, reference_out, atol=atol, rtol=rtol)
            allowed_error = atol + rtol * reference_out.float().abs().max().item()
        else:
            # Match QuACK's activation tests: compare the fused kernel's error
            # with a same-dtype PyTorch baseline. Gated activations multiply two
            # independently rounded projections, so a fixed plain-GEMM tolerance
            # can reject a valid one-BF16-ULP result.
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
            # Validate the down GEMM against the exact materialized tensor that
            # it consumes. The up GEMM was checked separately above, so this
            # isolates a down-projection error from the preceding operation.
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
        inputs.up_output,
    ]
    if inputs.W_down is not None:
        allocated_tensors.extend((inputs.W_down, inputs.output))
    input_bytes = sum(tensor.numel() * tensor.element_size() for tensor in allocated_tensors)
    routes_per_expert = args.routes // args.experts
    print(f"Device: {torch.cuda.get_device_name(device)} (SM{capability[0]}{capability[1]})")
    print(f"X: {tuple(inputs.X.shape)}, W_up: {tuple(inputs.W_up.shape)}, dtype: {inputs.X.dtype}")
    if inputs.W_down is not None:
        print(
            f"W_down: {tuple(inputs.W_down.shape)}, up output: {tuple(inputs.up_output.shape)}, "
            f"final output: {tuple(inputs.output.shape)}"
        )
    else:
        print(f"Output: {tuple(inputs.output.shape)}, down projection: disabled")
    print(
        f"Routes: {args.routes} total, {routes_per_expert} per expert, "
        f"sampling {'with' if args.routing_with_replacement else 'without'} replacement"
    )
    kernel_description = (
        "up gather=cp.async, down input=expert-contiguous"
        if args.down_projection
        else "gather=cp.async"
    )
    print(
        f"Kernel: tile=({args.tile_m}, {args.tile_n}, {args.tile_k or 'auto'}), "
        f"cluster=({args.cluster_m}, 1, 1), persistent=True; {kernel_description}"
    )
    print(f"Fused activation: {args.activation or 'disabled'}")
    print(f"Approximate tensor storage: {gib(input_bytes):.3f} GiB")
    compile_target = "up and down kernels" if args.down_projection else "kernel"
    print(f"Compiling and warming up the specialized QuACK {compile_target}...")

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
    up_flops = 2 * args.routes * args.hidden * inputs.W_up.shape[-1]
    down_flops = 2 * args.routes * args.output_dim * args.hidden if args.down_projection else 0
    flops = up_flops + down_flops
    tflops = flops / (median_ms * 1e9)
    timing_kind = "CUDA graph + events" if not args.no_cuda_graph else "batched CUDA events"
    formatted_samples = ", ".join(f"{value:.4f}" for value in timings_ms)
    print(f"Timing method: {timing_kind}")
    operation_name = "two-GEMM MLP" if args.down_projection else "up GEMM"
    print(f"Per-operation samples (ms): [{formatted_samples}]")
    print(f"Median {operation_name} time: {median_ms:.4f} ms")
    print(f"Effective throughput: {tflops:.2f} TFLOP/s")

    # The most recent graph/direct launch populated output before this check.
    if not args.skip_check:
        max_up_error, max_up_allowed, max_down_error, max_down_allowed = check_correctness(
            inputs,
            activation=args.activation,
            atol=args.atol,
            rtol=args.rtol,
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
