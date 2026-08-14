#!/usr/bin/env python3
"""Benchmark a QuACK Hopper grouped GEMM after materializing gathered tokens.

This is the no-gather-fusion counterpart of ``hopper_gather_gemm.py``. It first
creates the same MoE inputs on CUDA:

    X:             [T, K]
    W:             [E, K, N]
    A_idx:         [R]
    cu_seqlens_m:  [E + 1]

It then materializes ``A = X[A_idx]`` as a contiguous ``[R, K]`` CUDA tensor
before invoking QuACK. The timed grouped GEMM receives A and cu_seqlens_m, but
does not receive A_idx, so its A operand uses the ordinary contiguous TMA path.
The pre-gather operation, compilation, warmup, graph capture, and correctness
check are excluded from the reported GEMM time.

Example:

    python run/hopper_pregather_gemm.py \
        --tokens 4096 --hidden 4096 --output-dim 4096 \
        --experts 8 --routes 8192 --warmup 5 --iterations 100
"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass

import torch

from quack.gemm import gemm as quack_gemm


DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


@dataclass
class PregatherInputs:
    X: torch.Tensor
    W: torch.Tensor
    A_idx: torch.Tensor
    cu_seqlens_m: torch.Tensor
    A: torch.Tensor
    output: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark QuACK's SM90 grouped GEMM after pre-gathering routed tokens."
    )
    parser.add_argument("--tokens", "-T", type=int, default=4096, help="Number of X rows (T)")
    parser.add_argument("--hidden", "-K", type=int, default=4096, help="Input dimension (K)")
    parser.add_argument("--output-dim", "-N", type=int, default=4096, help="Output dimension (N)")
    parser.add_argument("--experts", "-E", type=int, default=8, help="Number of experts (E)")
    parser.add_argument("--routes", "-R", type=int, default=8192, help="Routed assignments (R)")
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--seed", type=int, default=0)

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
        help="Grouped GEMM launches captured in each timing sample",
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
def prepare_inputs(args: argparse.Namespace, device: torch.device) -> PregatherInputs:
    """Create routing inputs and materialize the expert-major A tensor on CUDA."""
    dtype = DTYPES[args.dtype]
    routes_per_expert = args.routes // args.experts

    X = torch.randn((args.tokens, args.hidden), dtype=dtype, device=device)
    W = torch.randn((args.experts, args.hidden, args.output_dim), dtype=dtype, device=device)
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
    output = torch.empty((args.routes, args.output_dim), dtype=dtype, device=device)
    return PregatherInputs(
        X=X,
        W=W,
        A_idx=A_idx,
        cu_seqlens_m=cu_seqlens_m,
        A=A,
        output=output,
    )


def make_launch(args: argparse.Namespace, inputs: PregatherInputs):
    # QuACK expects B as [E, N, K]. This is a zero-copy n-major view of the
    # user-facing W[E, K, N] allocation.
    B = inputs.W.transpose(1, 2)

    def launch() -> None:
        quack_gemm(
            inputs.A,
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
            pingpong=args.pingpong,
            persistent=True,
            is_dynamic_persistent=False,
            max_swizzle_size=args.max_swizzle_size,
            cu_seqlens_m=inputs.cu_seqlens_m,
            # A is already route-major and contiguous. Omitting A_idx selects
            # QuACK's ordinary TMA A-load path instead of fused gather-A.
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
    """Return per-grouped-GEMM milliseconds for every timing sample."""
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
def check_correctness(
    inputs: PregatherInputs,
    *,
    atol: float,
    rtol: float,
) -> float:
    """Check the grouped result against per-expert float32 PyTorch GEMMs."""
    num_experts = inputs.W.shape[0]
    routes_per_expert = inputs.A.shape[0] // num_experts
    max_abs_error = 0.0
    for expert in range(num_experts):
        start = expert * routes_per_expert
        end = start + routes_per_expert
        reference = inputs.A[start:end].float() @ inputs.W[expert].float()
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

    element_size = inputs.X.element_size()
    tensor_bytes = (
        args.tokens * args.hidden * element_size
        + args.experts * args.hidden * args.output_dim * element_size
        + args.routes * args.hidden * element_size
        + args.routes * args.output_dim * element_size
        + args.routes * torch.iinfo(torch.int32).bits // 8
        + (args.experts + 1) * torch.iinfo(torch.int32).bits // 8
    )
    routes_per_expert = args.routes // args.experts
    print(f"Device: {torch.cuda.get_device_name(device)} (SM{capability[0]}{capability[1]})")
    print(f"X: {tuple(inputs.X.shape)}, W: {tuple(inputs.W.shape)}, dtype: {inputs.X.dtype}")
    print(f"Pre-gathered A: {tuple(inputs.A.shape)}, contiguous={inputs.A.is_contiguous()}")
    print(
        f"Routes: {args.routes} total, {routes_per_expert} per expert, "
        f"sampling {'with' if args.routing_with_replacement else 'without'} replacement"
    )
    print(
        f"Kernel: tile=({args.tile_m}, {args.tile_n}, {args.tile_k or 'auto'}), "
        f"cluster=({args.cluster_m}, 1, 1), persistent=True, gather=none (TMA A load)"
    )
    print(f"Approximate tensor storage: {gib(tensor_bytes):.3f} GiB")
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
    flops = 2 * args.routes * args.hidden * args.output_dim
    tflops = flops / (median_ms * 1e9)
    timing_kind = "CUDA graph + events" if not args.no_cuda_graph else "batched CUDA events"
    formatted_samples = ", ".join(f"{value:.4f}" for value in timings_ms)
    print(f"Timing method: {timing_kind}")
    print(f"Per-kernel samples (ms): [{formatted_samples}]")
    print(f"Median kernel time: {median_ms:.4f} ms")
    print(f"Effective throughput: {tflops:.2f} TFLOP/s")

    if not args.skip_check:
        max_abs_error = check_correctness(inputs, atol=args.atol, rtol=args.rtol)
        print(f"Reference check: PASSED (max absolute error {max_abs_error:.6g})")


if __name__ == "__main__":
    main()
