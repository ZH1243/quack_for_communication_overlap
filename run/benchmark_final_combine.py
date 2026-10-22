"""Numerical check and interleaved GPU timings: python -m run.benchmark_final_combine.

Run on one otherwise idle Hopper GPU. Every timed round queues a GPU burner
before the event pairs to avoid measuring Python enqueue gaps. Variant order
rotates each round. Results describe resident receive buffers, not communication.
"""

import argparse
import statistics
import time

import torch


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--tile-n", type=int, default=128, help="0 for row-major input")
    parser.add_argument("--peers", type=int, default=8)
    parser.add_argument("--route-probability", type=float, default=0.5)
    parser.add_argument("--ctas", type=int, default=4)
    parser.add_argument("--elements-per-thread", type=int, default=32)
    parser.add_argument("--threads-per-cta", type=int, default=512)
    parser.add_argument("--stages", type=int, nargs="+", choices=(2, 3, 4), default=[2, 3, 4])
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--burner-cycles", type=int, default=20_000_000)
    args = parser.parse_args()
    if min(args.source_tokens, args.hidden, args.peers, args.rounds, args.burner_cycles) <= 0:
        parser.error("tokens, hidden, peers, rounds and burner cycles must be positive")
    if args.tile_n < 0 or (args.tile_n and args.hidden % args.tile_n):
        parser.error("tile-n must be zero or a positive divisor of hidden")
    if not 0 <= args.route_probability <= 1:
        parser.error("route-probability must be in [0, 1]")
    from run.final_combine import FinalCombine

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    generator = torch.Generator().manual_seed(42)
    indices = tuple(torch.where(torch.rand(args.source_tokens, generator=generator)
                                < args.route_probability)[0] for _ in range(args.peers))
    generator = torch.Generator(device="cuda").manual_seed(43)
    received = []
    reference = torch.zeros((args.source_tokens, args.hidden), device="cuda", dtype=torch.float32)
    for index in indices:
        logical = torch.randn((index.numel(), args.hidden), dtype=dtype,
                              device="cuda", generator=generator)
        reference.index_add_(0, index.cuda(), logical.float())
        received.append(logical if not args.tile_n else
                        logical.reshape(index.numel(), args.hidden // args.tile_n, args.tile_n)
                        .permute(1, 0, 2).contiguous())
    variants = [("direct-row", False, False, 3)]
    if args.tile_n:
        variants.append(("direct-column", False, True, 3))
    variants.extend((f"pipeline-{stages}", True, bool(args.tile_n), stages)
                    for stages in args.stages)
    combines = {}
    for name, pipelined, column_major, stages in variants:
        combine = FinalCombine(
            received, indices, args.source_tokens, args.hidden, args.tile_n,
            num_ctas=args.ctas, elements_per_thread=args.elements_per_thread,
            threads_per_cta=args.threads_per_cta, column_group_major=column_major,
            pipelined=pipelined, pipeline_stages=stages,
        )
        combine.output.fill_(float("nan"))
        torch.testing.assert_close(combine(), reference.to(dtype), rtol=0, atol=0)
        combines[name] = combine
    torch.cuda.synchronize()
    print(f"Numerical checks passed on {torch.cuda.get_device_name()}.")
    deadline = time.monotonic() + args.warmup_seconds
    while time.monotonic() < deadline:
        for _ in range(10):
            for combine in combines.values():
                combine()
        torch.cuda.synchronize()

    # Materialize events before timing; event creation must not consume the backlog.
    events = {name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for name in combines}
    for start, end in events.values():
        start.record()
        end.record()
    torch.cuda.synchronize()
    timings = {name: [] for name in combines}
    names = list(combines)
    for round_index in range(args.rounds):
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
        torch.cuda._sleep(args.burner_cycles)
        for name in order:
            start, end = events[name]
            start.record()
            combines[name]()
            end.record()
        torch.cuda.synchronize()
        for name in names:
            start, end = events[name]
            timings[name].append(start.elapsed_time(end) * 1000)
    output_bytes = args.source_tokens * args.hidden * 2
    total_bytes = output_bytes + sum(buffer.numel() * 2 for buffer in received)
    baseline = statistics.median(timings["direct-row"])
    print("variant            median us    min us    p90 us   output GB/s   read+write GB/s   speedup")
    for name, samples in timings.items():
        median = statistics.median(samples)
        p90 = sorted(samples)[min(len(samples) - 1, int(0.9 * len(samples)))]
        print(f"{name:18s} {median:9.2f} {min(samples):9.2f} {p90:9.2f} "
              f"{output_bytes / median / 1000:13.2f} {total_bytes / median / 1000:17.2f} "
              f"{baseline / median:8.3f}x")
    print("Bandwidth counts logical payload bytes (metadata excluded), not measured HBM traffic.")


if __name__ == "__main__":
    main()
