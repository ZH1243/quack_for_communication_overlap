#!/usr/bin/env python3
"""Prepare and benchmark a QuACK Hopper GEMM with fused token gather.

The user-facing tensors follow the MoE notation

    X:             [T, K]
    W:             [E, K, N] (or [E, K, 2N] for a gated activation)
    A_idx:         [R]
    cu_seqlens_m:  [E + 1]
    output:        [R, N]

``A_idx[cu_seqlens_m[e]:cu_seqlens_m[e + 1]]`` contains the token rows used
with ``W[e]``. All input tensors, including the routing metadata, are created
directly on the selected CUDA device. QuACK takes weights as [E, N, K], so a
zero-copy transposed view of W is passed to the low-level GEMM API.

Example:

    python run/hopper_gather_gemm.py \
        --tokens 4096 --hidden 4096 --output-dim 4096 \
        --experts 8 --routes 8192 --warmup 5 --iterations 100

Fuse a SwiGLU up-projection (the weight uses concatenated [gate | up] columns):

    python run/hopper_gather_gemm.py \
        --tokens 4096 --hidden 4096 --output-dim 14336 \
        --experts 8 --routes 8192 --activation swiglu

The first GEMM call may take a while because QuACK compiles the specialized
kernel. Compilation, warmup, graph capture, and validation are not timed.
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
    W: torch.Tensor
    A_idx: torch.Tensor
    cu_seqlens_m: torch.Tensor
    output: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark QuACK's SM90 GEMM with fused gather-A for uniform MoE routing."
    )
    parser.add_argument("--tokens", "-T", type=int, default=4096, help="Number of X rows (T)")
    parser.add_argument("--hidden", "-K", type=int, default=4096, help="Input dimension (K)")
    parser.add_argument("--output-dim", "-N", type=int, default=4096, help="Output dimension (N)")
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
    parser.add_argument("--warmup", type=int, default=5, help="Untimed GEMM launches")
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="GEMM launches captured in each timing sample",
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
    """Allocate and initialize X, W, routing metadata, and output in GPU HBM."""
    dtype = DTYPES[args.dtype]
    routes_per_expert = args.routes // args.experts

    # torch.randn on a CUDA device initializes these tensors with CUDA kernels;
    # no full-size input is staged through host memory.
    X = torch.randn((args.tokens, args.hidden), dtype=dtype, device=device)
    gemm_output_dim = args.output_dim * (2 if args.activation in GATED_ACTIVATIONS else 1)
    W = torch.randn((args.experts, args.hidden, gemm_output_dim), dtype=dtype, device=device)
    W.mul_(1.0 / math.sqrt(args.hidden))

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

    output = torch.empty((args.routes, args.output_dim), dtype=dtype, device=device)
    return GatherInputs(X=X, W=W, A_idx=A_idx, cu_seqlens_m=cu_seqlens_m, output=output)


def make_launch(args: argparse.Namespace, inputs: GatherInputs):
    # QuACK's low-level B convention is [E, N, K]. This is only a view: the
    # allocation retained in GatherInputs remains the requested W[E, K, N].
    B = inputs.W.transpose(1, 2)

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
                inputs.W,
                activation=args.activation,
                postact_out=inputs.output,
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
            return
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

    return launch


@torch.inference_mode()
def benchmark(
    launch,
    *,
    iterations: int,
    samples: int,
    use_cuda_graph: bool,
) -> list[float]:
    """Return per-kernel milliseconds for each timing sample."""
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
) -> float:
    """Check every output value against per-expert float32 PyTorch references."""
    num_experts = inputs.W.shape[0]
    routes_per_expert = inputs.A_idx.numel() // num_experts
    max_abs_error = 0.0
    for expert in range(num_experts):
        start = expert * routes_per_expert
        end = start + routes_per_expert
        token_idx = inputs.A_idx[start:end]
        reference = inputs.X[token_idx].float() @ inputs.W[expert].float()
        if activation in GATED_ACTIVATIONS:
            gate, up = reference.chunk(2, dim=-1)
            reference = gated_to_pytorch_fn_map[activation](gate, up)
        elif activation is not None:
            reference = act_to_pytorch_fn_map[activation](reference)
        reference = reference.to(inputs.output.dtype)
        actual = inputs.output[start:end]
        max_abs_error = max(max_abs_error, (actual - reference).abs().max().item())
        torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
    return max_abs_error


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

    input_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (inputs.X, inputs.W, inputs.A_idx, inputs.cu_seqlens_m, inputs.output)
    )
    routes_per_expert = args.routes // args.experts
    print(f"Device: {torch.cuda.get_device_name(device)} (SM{capability[0]}{capability[1]})")
    print(f"X: {tuple(inputs.X.shape)}, W: {tuple(inputs.W.shape)}, dtype: {inputs.X.dtype}")
    print(
        f"Routes: {args.routes} total, {routes_per_expert} per expert, "
        f"sampling {'with' if args.routing_with_replacement else 'without'} replacement"
    )
    print(
        f"Kernel: tile=({args.tile_m}, {args.tile_n}, {args.tile_k or 'auto'}), "
        f"cluster=({args.cluster_m}, 1, 1), persistent=True, gather=cp.async"
    )
    print(f"Fused activation: {args.activation or 'disabled'}")
    print(f"Approximate tensor storage: {gib(input_bytes):.3f} GiB")
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
    flops = 2 * args.routes * args.hidden * inputs.W.shape[-1]
    tflops = flops / (median_ms * 1e9)
    timing_kind = "CUDA graph + events" if not args.no_cuda_graph else "batched CUDA events"
    formatted_samples = ", ".join(f"{value:.4f}" for value in timings_ms)
    print(f"Timing method: {timing_kind}")
    print(f"Per-kernel samples (ms): [{formatted_samples}]")
    print(f"Median kernel time: {median_ms:.4f} ms")
    print(f"Effective throughput: {tflops:.2f} TFLOP/s")

    # The most recent graph/direct launch populated output before this check.
    if not args.skip_check:
        max_abs_error = check_correctness(
            inputs,
            activation=args.activation,
            atol=args.atol,
            rtol=args.rtol,
        )
        print(f"Reference check: PASSED (max absolute error {max_abs_error:.6g})")


if __name__ == "__main__":
    main()
