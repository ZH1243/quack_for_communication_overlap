#!/usr/bin/env python3
"""Interleaved, CUDA-graph-backed TFLOPS comparison; all variants checked first."""

import argparse
import math
import statistics

import torch

from quack.gemm import gemm
from blackwell_gemm_tables import make_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--buffers", type=int, default=16)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--routes-per-buffer", type=int, default=2048)
    parser.add_argument("--tokens-per-buffer", type=int, default=2048)
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--cluster-m", type=int, choices=(1, 2), default=2)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--sm-count", type=int, default=None)
    parser.add_argument("--stores", action="store_true", help="also measure bulk copy and scatter-add")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--repetitions", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=100, help="full graph replays")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("run this benchmark on a B200/B300 node")
    if min(args.k, args.n, args.buffers, args.experts, args.routes_per_buffer,
           args.tokens_per_buffer, args.rounds, args.repetitions) <= 0:
        parser.error("dimensions and repetition counts must be positive")
    if args.buffers < 2 and not args.baseline_only:
        parser.error("the multi-buffer comparison requires at least two buffers")
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype, device = torch.bfloat16, torch.device("cuda")
    counts = [[args.routes_per_buffer // args.experts + (e < args.routes_per_buffer % args.experts)
               for e in range(args.experts)] for _ in range(args.buffers)]
    geometry = dict(n=args.n, tile_m=args.tile_m, tile_n=args.tile_n,
                    cluster_m=args.cluster_m, group_size=args.group_size)
    multi_rows, offsets = make_table(counts, **geometry)
    expert_counts = [sum(cs[e] for cs in counts) for e in range(args.experts)]
    single_rows, _ = make_table([expert_counts], **geometry)
    multi_table = torch.tensor(multi_rows, device=device, dtype=torch.int32)
    single_table = torch.tensor(single_rows, device=device, dtype=torch.int32)
    # Identical physical source data across variants avoids giving the two
    # single-buffer variants an L2-sharing advantage over multi-buffer gather.
    # Independent source allocations are exercised by the numerical suite.
    joined_x = torch.randn(args.buffers * args.tokens_per_buffer, args.k,
                           device=device, dtype=dtype)
    xs = tuple(joined_x.split(args.tokens_per_buffer))
    indices = tuple(torch.randint(args.tokens_per_buffer, (sum(cs),), device=device,
                                  dtype=torch.int32) for cs in counts)
    joined_idx = torch.cat([indices[b][offsets[b][e]:offsets[b][e + 1]]
                            + b * args.tokens_per_buffer for e in range(args.experts)
                            for b in range(args.buffers)])
    cu = torch.tensor([0, *expert_counts], device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    w = torch.randn(args.experts, args.n, args.k, device=device, dtype=dtype) / math.sqrt(args.k)
    r = joined_idx.numel()
    reference = torch.empty(r, args.n, device=device, dtype=torch.float32)
    start = 0
    for e, count in enumerate(expert_counts):
        gathered = joined_x[joined_idx[start:start + count].long()]
        reference[start:start + count] = gathered.float() @ w[e].float().T
        start += count
    common = dict(tile_M=args.tile_m, tile_N=args.tile_n, cluster_M=args.cluster_m,
                  cluster_N=1, max_swizzle_size=args.group_size,
                  is_dynamic_persistent=False, sm_count=args.sm_count)
    variants = {}
    outputs = {}

    def add(name, a, idx, *, table=None, store="tma", scatter=None):
        out = torch.zeros(r, args.n, device=device, dtype=dtype)
        outputs[name] = out
        def call():
            gemm(a, w, out, None, None, A_idx=idx,
                 cu_seqlens_m=cu if table is None else None, gather_work_table=table,
                 multi_buffer_gather=isinstance(a, tuple), epilogue_store=store,
                 scatter_table=scatter, **common)
        call()
        expected = reference
        if scatter is not None:
            expected = torch.zeros_like(reference).index_add_(
                0, scatter.long(), reference.to(dtype).float()
            )
        torch.testing.assert_close(out.float(), expected, atol=0.07, rtol=0.035)
        variants[name] = call

    add("indexed baseline", joined_x, joined_idx)
    if not args.baseline_only:
        add("single table", joined_x, joined_idx, table=single_table)
        add("multi table", xs, indices, table=multi_table)
        if args.stores:
            add("bulk rows", joined_x, joined_idx, table=single_table, store="bulk_rows")
            mapping = torch.arange(r, dtype=torch.int32, device=device) // 2
            add("scatter add", joined_x, joined_idx, table=single_table,
                store="bulk_rows_reduce", scatter=mapping)
    torch.cuda.synchronize()
    names = list(variants)
    graphs = []
    for rotation in range(len(names)):
        graph = torch.cuda.CUDAGraph()
        events = []
        order = names[rotation:] + names[:rotation]
        with torch.cuda.graph(graph):
            # Backlog keeps event pairs from measuring host enqueue gaps. Variants
            # alternate at single-launch granularity; rotate L2 predecessor order.
            torch.cuda._sleep(2_000_000)
            for _ in range(args.repetitions):
                for name in order:
                    if name == "scatter add":
                        outputs[name].zero_()  # reset outside the measured interval
                    # External events retain record nodes/timestamps in a captured graph.
                    start = torch.cuda.Event(enable_timing=True, external=True)
                    end = torch.cuda.Event(enable_timing=True, external=True)
                    start.record()
                    variants[name]()
                    end.record()
                    events.append((name, start, end))
        graphs.append((graph, events))
    for i in range(args.warmup):
        graphs[i % len(graphs)][0].replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in names}
    round_baselines = []
    for i in range(args.rounds):
        graph, events = graphs[i % len(graphs)]
        graph.replay()
        torch.cuda.synchronize()
        per_round = []
        for name, start, end in events:
            us = start.elapsed_time(end) * 1000
            samples[name].append(us)
            if name == names[0]:
                per_round.append(us)
        round_baselines.append(statistics.median(per_round))
    base = statistics.median(samples[names[0]])
    print(f"{torch.cuda.get_device_name()}: R={r} N={args.n} K={args.k}, "
          f"MMA tile=({args.tile_m},{args.tile_n}), cluster_m={args.cluster_m}")
    print("variant                 median us    TFLOPS   time/baseline")
    for name, times in samples.items():
        us = statistics.median(times)
        print(f"{name:24s} {us:9.2f} {2 * r * args.n * args.k / us / 1e6:9.1f} {us / base:12.3f}")
    print(f"Baseline round drift (max/min): {max(round_baselines) / min(round_baselines):.3f}; "
          "large drift means the comparison should be repeated on an idle node.")
    print("Packing, metadata construction, and destination zeroing excluded; "
          "these are GEMM timings, not end-to-end proxy throughput.")


if __name__ == "__main__":
    main()
