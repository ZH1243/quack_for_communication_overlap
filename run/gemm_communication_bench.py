#!/usr/bin/env python3
"""Shared Hopper/Blackwell feature comparisons and numerical references.

Use hopper_gemm_communication_bench.py or blackwell_gemm_communication_bench.py
to select the architecture. Inputs, controls, and timing are shared.
"""

import argparse
import math
from contextlib import ExitStack

import torch
import torch.nn.functional as F


def parse_args(argv=None, *, device_capacity=10):
    if device_capacity not in (9, 10):
        raise ValueError("feature comparisons support Hopper (9) and Blackwell (10)")
    hopper = device_capacity == 9
    target = "Hopper H100/H200" if hopper else "Blackwell B200/B300"
    parser = argparse.ArgumentParser(
        description=f"Matched {target} feature comparisons, checked before and after timing."
    )
    parser.set_defaults(device_capacity=device_capacity, pingpong=hopper, tile_k=None)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=1024,
                        help="GEMM projection width; SwiGLU output has N/2 columns")
    parser.add_argument("--buffers", type=int, default=16)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--routes-per-buffer", type=int, default=2048)
    parser.add_argument("--tokens-per-buffer", type=int, default=2048)
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=128)
    if hopper:
        parser.add_argument("--tile-k", type=int, default=None,
                            help="CTA K tile; omitted uses the kernel default")
        parser.add_argument("--pingpong", action=argparse.BooleanOptionalAction, default=True,
                            help="use pingpong for every variant (default: on; required for waves)")
    parser.add_argument("--cluster-m", type=int, choices=(1, 2), default=2)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--sm-count", type=int, default=None)
    parser.add_argument("--packed-baseline", action="store_true",
                        help="also compare a non-gather grouped GEMM; packing is excluded")
    parser.add_argument("--inline-indices", action="store_true",
                        help="compare table order, then embedded indices against the same order")
    parser.add_argument("--reordered-tables", action="store_true",
                        help="compare single/multi-buffer tables in matched expert/M/N order")
    parser.add_argument("--single-source-fast-path", action="store_true",
                        help="experimental Hopper multi-buffer gather optimization; "
                             "implies --reordered-tables")
    parser.add_argument("--hoist-gather-addresses", action="store_true",
                        help="experimental Hopper gather address hoisting; "
                             "implies --reordered-tables")
    parser.add_argument("--stores", action="store_true",
                        help="bulk copy, row scatter, reduction, multi-buffer and compact output")
    parser.add_argument("--readiness", action="store_true",
                        help="table-ready acquire checks with all rows already available")
    parser.add_argument("--waves", action="store_true",
                        help="completion barrier and GPU ready prefix; implies --stores")
    parser.add_argument("--host-ready", action="store_true",
                        help="mapped-host ready-prefix publication; implies --waves")
    parser.add_argument("--weighted-contributions", action="store_true",
                        help="compare fused FP32 route scaling against matching unweighted "
                             "scatter/reduction and enabled completion paths; implies --stores")
    parser.add_argument("--swiglu", action="store_true",
                        help="compare indexed/table fused SwiGLU with store_preact=False")
    parser.add_argument("--baseline-only", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--variants", nargs="+", metavar="NAME",
                           help="time only these enabled variants; first becomes the baseline")
    selection.add_argument("--profile-variant", metavar="NAME",
                           help="check/warm one enabled variant, then launch in NVTX quack_profile")
    if not hopper:
        parser.add_argument("--print-kernel-config", action="store_true",
                            help="print compiled stages/layouts; requires QUACK_CACHE_ENABLED=0")
        parser.add_argument("--bulk-smem-padding", type=int, choices=(0, 16), default=None,
                            help="bulk-row padding in bytes; requires QUACK_CACHE_ENABLED=0")
        parser.add_argument("--bulk-smem-cols", type=int, choices=(0, 64, 128, 256), default=None,
                            help="buffered columns, 0=full; requires QUACK_CACHE_ENABLED=0")
        parser.add_argument("--bulk-smem-buffers", type=int, choices=(1, 2), default=None,
                            help="split columns into reusable windows; requires cache disabled")
        parser.add_argument("--bulk-store-warps", type=int, choices=(1, 2, 4), default=None,
                            help="row-copy issuing epilogue warps; requires cache disabled")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--repetitions", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=100, help="full graph replays")
    args = parser.parse_args(argv)
    if args.single_source_fast_path and not hopper:
        parser.error("--single-source-fast-path requires Hopper (SM90)")
    if args.hoist_gather_addresses and not hopper:
        parser.error("--hoist-gather-addresses requires Hopper (SM90)")
    args.reordered_tables = (
        args.reordered_tables or args.single_source_fast_path or args.hoist_gather_addresses
    )
    args.waves = args.waves or args.host_ready
    args.stores = args.stores or args.waves or args.weighted_contributions
    if min(args.k, args.n, args.buffers, args.experts, args.routes_per_buffer,
           args.tokens_per_buffer, args.tile_m, args.tile_n, args.group_size,
           args.rounds, args.repetitions) <= 0 or args.warmup < 0:
        parser.error("dimensions/counts must be positive and warmup nonnegative")
    if args.buffers < 2 and not args.baseline_only:
        parser.error("the multi-buffer comparison requires at least two buffers")
    if args.baseline_only and (
        args.stores or args.readiness or args.swiglu or args.inline_indices or args.reordered_tables
    ):
        parser.error("--baseline-only cannot be combined with feature comparisons")
    if args.sm_count is not None and (args.sm_count <= 0 or args.sm_count % args.cluster_m):
        parser.error("sm-count must be positive and divisible by cluster-m")
    if args.swiglu and args.n % 2:
        parser.error("SwiGLU requires even N")
    if args.stores and args.n % 8:
        parser.error("bulk stores require BF16 row widths divisible by 8")
    if getattr(args, "bulk_smem_padding", None) is not None and not args.stores:
        parser.error("--bulk-smem-padding requires --stores, --waves, or --host-ready")
    if getattr(args, "bulk_smem_cols", None) is not None and not args.stores:
        parser.error("--bulk-smem-cols requires --stores, --waves, or --host-ready")
    if getattr(args, "bulk_smem_buffers", None) is not None and not args.stores:
        parser.error("--bulk-smem-buffers requires --stores, --waves, or --host-ready")
    if getattr(args, "bulk_store_warps", None) is not None and not args.stores:
        parser.error("--bulk-store-warps requires --stores, --waves, or --host-ready")
    if hopper:
        if args.waves and not args.pingpong:
            parser.error("Hopper wave barriers and ready publication require --pingpong")
        if args.tile_k is not None and (args.tile_k <= 0 or args.tile_k % 16):
            parser.error("BF16 tile-k must be a positive multiple of 16")
        if args.pingpong:
            max_n = {64: 256, 128: 208, 192: 128}.get(args.tile_m)
            if max_n is None or args.tile_n % 16 or args.tile_n > max_n:
                parser.error("Hopper pingpong requires tile-m 64/128/192 and tile-n "
                             "divisible by 16, at most 256/208/128 respectively")
        elif args.tile_m not in (64, 128, 192, 256, 320):
            parser.error("Hopper tile-m must be 64/128/192/256/320")
        elif args.tile_m in (192, 320):
            max_n = 256 if args.tile_m == 192 else 160
            if args.tile_n % 32 or args.tile_n > max_n:
                parser.error(f"Hopper tile-m {args.tile_m} requires tile-n divisible by 32 "
                             f"and at most {max_n}")
        elif not ((args.tile_n % 16 == 0 and args.tile_n <= 256)
                  or (args.tile_n % 32 == 0 and args.tile_n <= 512)):
            parser.error("Hopper tile-n must be divisible by 16 and <=256, "
                         "or divisible by 32 and <=512")
    ntiles = math.ceil(args.n / args.tile_n)
    if ntiles % args.group_size:
        parser.error("N tile count must be divisible by group-size")
    return args


def make_route_weights(mapping, *, seed=5678):
    """Fixed positive FP32 scales normalized over each synthetic destination's routes.

    The benchmark pairs routes into duplicate destinations, rather than using a
    real top-k router. Normalize complete destination groups before splitting
    them into output buffers. A private RNG preserves the existing workload.
    """
    generator = torch.Generator(device=mapping.device).manual_seed(seed)
    weights = torch.rand(mapping.numel(), dtype=torch.float32, device=mapping.device,
                         generator=generator).add_(0.01)
    totals = torch.zeros_like(weights).scatter_add_(0, mapping.long(), weights)
    return (weights / totals[mapping.long()]).contiguous()


def scatter_reference(reference, mapping, dtype, route_weights=None):
    """Round each scaled FP32 contribution before the scatter-add reference."""
    accumulator = reference if route_weights is None else reference * route_weights[:, None]
    return torch.zeros_like(reference).index_add_(0, mapping.long(), accumulator.to(dtype).float())


def reorder_gather_rows(rows, *, multi_buffer=False):
    """Order descriptors by expert, packed output M range, then N group.

    Multi-buffer output offsets are sums of the per-source offsets, as in the
    kernel scheduler. Sorting by those sums matches the single-buffer traversal,
    including tiles spanning source boundaries and sources with empty ranges.
    """
    if multi_buffer:
        return sorted(rows, key=lambda row: (row[0], sum(row[2::2]), sum(row[3::2]), row[1]))
    return sorted(rows)


def make_inline_comparison_rows(single_rows, tokens, *, cluster_rows):
    """Build separate-index and inline tables with identical expert/M/N traversal.

    Inline descriptors require consecutive N-group rows for each M bundle.
    Keep an ordinary four-column control in that order to separate traversal
    effects from the cost of embedding indices. Preparation is outside timing.
    """
    ordered_rows = reorder_gather_rows(single_rows)
    inline_rows = [
        [e, ng, start, end, *tokens[start:end], *([-1] * (cluster_rows - (end - start)))]
        for e, start, end, ng in ordered_rows
    ]
    return ordered_rows, inline_rows


def single_source_cta_coverage(rows, *, cta_rows, cluster_m, group_size=1):
    """Count single-source, mixed-source and empty CTA work tiles from metadata.

    Each descriptor concatenates its source ranges before splitting rows across
    CTAs. A partial CTA is eligible when its valid rows belong to one source.
    N-group expansion repeats that eligibility for every tile in the group.
    """
    single, mixed, empty = 0, 0, 0
    for row in rows:
        lengths = [end - start for start, end in zip(row[2::2], row[3::2])]
        for cta in range(cluster_m):
            lo, hi = cta * cta_rows, (cta + 1) * cta_rows
            prefix, active = 0, 0
            for length in lengths:
                active += max(lo, prefix) < min(hi, prefix + length)
                prefix += length
            single += (active == 1) * group_size
            mixed += (active > 1) * group_size
            empty += (active == 0) * group_size
    return single, mixed, empty


def benchmark(args, resources):
    # Delay CUDA/DSL imports so CLI and reference arithmetic can be tested on CPU.
    device_capacity = args.device_capacity
    target = "H100/H200 (SM90)" if device_capacity == 9 else "B200/B300 (SM10x)"
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != device_capacity:
        raise RuntimeError(f"run this benchmark on an {target} node")
    from quack.cute_dsl_utils import get_max_active_clusters
    from quack.gemm import build_scatter_row_addresses, gemm
    from quack.gemm_config import GemmConfig, cta_tile_shape_m
    from quack.gemm_interface import gemm_act
    if __package__:
        from .gemm_bench_utils import (
            Variant, mapped_host_flag, measure, profile_variant, select_variants,
        )
        from .gemm_tables import make_table
        from .wave_ready import build_wave_ready_prefix
    else:
        from gemm_bench_utils import (
            Variant, mapped_host_flag, measure, profile_variant, select_variants,
        )
        from gemm_tables import make_table
        from wave_ready import build_wave_ready_prefix
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype, device = torch.bfloat16, torch.device("cuda")
    counts = [[args.routes_per_buffer // args.experts + (e < args.routes_per_buffer % args.experts)
               for e in range(args.experts)] for _ in range(args.buffers)]
    geometry = dict(n=args.n, tile_m=args.tile_m, tile_n=args.tile_n,
                    cluster_m=args.cluster_m, group_size=args.group_size,
                    device_capacity=device_capacity)
    multi_rows, offsets = make_table(counts, **geometry)
    expert_counts = [sum(cs[e] for cs in counts) for e in range(args.experts)]
    single_rows, _ = make_table([expert_counts], **geometry)
    multi_table = torch.tensor(multi_rows, device=device, dtype=torch.int32)
    single_table = torch.tensor(single_rows, device=device, dtype=torch.int32)
    # Source views keep physical data identical across the gather variants.
    # Independent source allocations are covered by the numerical suite.
    joined_x = torch.randn(args.buffers * args.tokens_per_buffer, args.k,
                           device=device, dtype=dtype)
    xs = tuple(joined_x.split(args.tokens_per_buffer))
    indices = tuple(torch.randint(args.tokens_per_buffer, (sum(cs),), device=device,
                                  dtype=torch.int32) for cs in counts)
    joined_idx = torch.cat([indices[b][offsets[b][e]:offsets[b][e + 1]]
                            + b * args.tokens_per_buffer for e in range(args.experts)
                            for b in range(args.buffers)])
    cu = torch.tensor([0, *expert_counts], device=device, dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
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
                  tile_K=args.tile_k,
                  persistent=True, is_dynamic_persistent=False, pingpong=args.pingpong,
                  use_tma_gather=False, sm_count=args.sm_count)
    print(f"{torch.cuda.get_device_name()}: dtype={dtype}, R={r} N={args.n} K={args.k}, "
          f"experts={args.experts}, buffers={args.buffers}", flush=True)
    print(f"Routes per expert: {expert_counts}", flush=True)
    cta_m = cta_tile_shape_m(args.tile_m, args.cluster_m, device_capacity)
    print(f"Shared config: tile=({args.tile_m},{args.tile_n}), "
          f"tile_K={args.tile_k if args.tile_k is not None else 'auto'}, "
          f"CTA rows={cta_m}, cluster rows={cta_m * args.cluster_m}, "
          f"cluster=({args.cluster_m},1), group_size={args.group_size}, "
          f"sm_count={args.sm_count if args.sm_count is not None else 'device default'}",
          flush=True)
    print(f"Shared config: static persistent, swap_ab=False, pingpong={args.pingpong}; "
          "gather variants use cp.async and identical source data/weights", flush=True)
    variants = {}

    def check(actual, expected, *, reduce=False):
        baseline_error = (expected.to(dtype).float() - expected).abs().max().item()
        torch.testing.assert_close(actual.float(), expected,
                                   atol=max(0.07 if reduce else 0.035, 2 * baseline_error),
                                   rtol=0.035)

    def add(name, a, idx, *, table=None, store="tma", scatter=None, out=None,
            expected=reference, logical=None, control=None, prepare_extra=lambda: None,
            check_extra=lambda: None, **kwargs):
        if out is None:
            out = torch.full((r, args.n), float("nan"), device=device, dtype=dtype)
        buffers = out if isinstance(out, tuple) else (out,)

        def call():
            gemm(a, w, out, None, None, A_idx=idx,
                 cu_seqlens_m=cu if table is None else None, gather_work_table=table,
                 multi_buffer_gather=isinstance(a, tuple), epilogue_store=store,
                 scatter_table=scatter, **common, **kwargs)

        def prepare():
            if store == "bulk_rows_reduce":
                for buffer in buffers:
                    buffer.zero_()
            prepare_extra()

        def validate():
            check(logical() if logical else out, expected, reduce=store == "bulk_rows_reduce")
            check_extra()

        variants[name] = Variant(call, validate, prepare, control)

    if args.packed_baseline:
        packed = joined_x[joined_idx.long()].contiguous()
        add("packed baseline", packed, None)
    add("indexed baseline", joined_x, joined_idx,
        control="packed baseline" if args.packed_baseline else None)
    if not args.baseline_only:
        add("single table", joined_x, joined_idx, table=single_table, control="indexed baseline")
        add("multi table", xs, indices, table=multi_table, control="single table")

    ordered_table, ordered_multi_table = None, None
    if args.inline_indices or args.reordered_tables:
        ordered_rows = reorder_gather_rows(single_rows)
        ordered_table = torch.tensor(ordered_rows, device=device, dtype=torch.int32)
        add("single table reordered", joined_x, joined_idx, table=ordered_table,
            control="single table")

    if args.reordered_tables:
        ordered_multi_rows = reorder_gather_rows(multi_rows, multi_buffer=True)
        ordered_multi_table = torch.tensor(ordered_multi_rows, device=device, dtype=torch.int32)
        add("multi table reordered", xs, indices, table=ordered_multi_table,
            control="single table reordered")
        print("Reordered controls: single/multi tables traverse expert/M/N; "
              "multi table reordered compares against single table reordered.", flush=True)

    if args.single_source_fast_path:
        add("multi single-source fast", xs, indices, table=ordered_multi_table,
            control="multi table reordered", gather_single_source_fast_path=True)
        single, mixed, empty = single_source_cta_coverage(
            ordered_multi_rows, cta_rows=cta_m, cluster_m=args.cluster_m,
            group_size=args.group_size,
        )
        print(f"Experimental single-source fast path: CTA work tiles from metadata "
              f"(including N groups): {single} eligible, {mixed} mixed, {empty} empty. "
              "Mixed/empty CTAs retain the original gather path.", flush=True)
        print("Fast-path control: identical multi-buffer inputs, table, expert/M/N order "
              "and launch settings; only the single-source specialization is enabled. "
              "Readiness checks are disabled for this pair; SwiGLU variants are unchanged.",
              flush=True)

    if args.hoist_gather_addresses:
        add("multi hoisted addresses", xs, indices, table=ordered_multi_table,
            control="multi table reordered", gather_hoist_addresses=True)
        print("Experimental address hoist: identical multi-buffer inputs, table, expert/M/N "
              "order and launch settings; only gather address hoisting is enabled. "
              "This variant retains the original source selection for mixed/partial CTAs; "
              "single-source optimization and readiness checks are disabled. "
              "SwiGLU variants are unchanged.", flush=True)

    if args.inline_indices:
        _, inline_rows = make_inline_comparison_rows(
            ordered_rows, joined_idx.cpu().tolist(), cluster_rows=cta_m * args.cluster_m,
        )
        inline_table = torch.tensor(inline_rows, device=device, dtype=torch.int32)
        unused_idx = torch.full_like(joined_idx, -1)
        add("inline-index table", joined_x, unused_idx, table=inline_table,
            control="single table reordered")
        print("Inline controls: single table reordered changes only traversal to expert/M/N; "
              "inline-index table embeds indices with that same traversal.", flush=True)

    if args.readiness:
        single_ready = torch.full((1,), len(single_rows), device=device, dtype=torch.int32)
        multi_ready = torch.full((1,), len(multi_rows), device=device, dtype=torch.int32)
        add("single table ready", joined_x, joined_idx, table=single_table,
            gather_work_table_ready=single_ready, control="single table")
        add("multi table ready", xs, indices, table=multi_table,
            gather_work_table_ready=multi_ready, control="multi table")
        if ordered_table is not None:
            add("single reordered ready", joined_x, joined_idx, table=ordered_table,
                gather_work_table_ready=single_ready, control="single table reordered")
        if ordered_multi_table is not None:
            add("multi reordered ready", xs, indices, table=ordered_multi_table,
                gather_work_table_ready=multi_ready, control="multi table reordered")
        print("Ready-check timing: all rows are available; producer delay is excluded.", flush=True)

    if args.stores:
        add("bulk rows", joined_x, joined_idx, table=single_table,
            store="bulk_rows", control="single table")
        permutation = torch.randperm(r, device=device, dtype=torch.int32)
        perm_ref = torch.empty_like(reference)
        perm_ref[permutation.long()] = reference
        add("row scatter", joined_x, joined_idx, table=single_table,
            store="bulk_rows", scatter=permutation, expected=perm_ref, control="bulk rows")
        add("reduce unique", joined_x, joined_idx, table=single_table,
            store="bulk_rows_reduce", scatter=permutation,
            expected=perm_ref.to(dtype).float(), control="row scatter")
        # Pair routes into even destination rows across the full allocation so
        # duplicate reduction exercises every destination buffer at normal sizes.
        mapping = (permutation // 2) * 2
        reduced = scatter_reference(reference, mapping, dtype)
        add("scatter add", joined_x, joined_idx, table=single_table,
            store="bulk_rows_reduce", scatter=mapping, expected=reduced, control="reduce unique")
        route_weights = None
        weighted_reduced = None
        if args.weighted_contributions:
            route_weights = make_route_weights(mapping)
            weighted_reduced = scatter_reference(reference, mapping, dtype, route_weights)
            add("weighted scatter add", joined_x, joined_idx, table=single_table,
                store="bulk_rows_reduce", scatter=mapping, expected=weighted_reduced,
                route_weights=route_weights, control="scatter add")
            print("Routing weights: fixed FP32 positive random values (seed=5678), normalized "
                  "over all routes sharing each synthetic destination. Weight generation and "
                  "packing excluded; scaling occurs before BF16 conversion. Read each weighted "
                  "row's 'vs control' for its added GEMM-path cost.", flush=True)
        # Keep global destination row order identical across single/multi-buffer
        # scatter. Destinations are equal-sized views of one allocation, avoiding
        # an allocation-placement change when measuring address representation.
        rows_per_dest = args.routes_per_buffer
        dest_mapping = torch.stack((mapping // rows_per_dest, mapping % rows_per_dest), dim=1)
        ntiles = math.ceil(args.n / args.tile_n)

        def add_destinations(name, *, compact=False, control, expected=reduced, **kwargs):
            width = ntiles * args.tile_n if compact else args.n
            storage = torch.zeros(r, width, device=device, dtype=dtype)
            buffers = tuple(storage.split(rows_per_dest))
            if compact:
                buffers = tuple(b.view(ntiles, rows_per_dest * args.tile_n) for b in buffers)
            addresses = build_scatter_row_addresses(
                buffers, dest_mapping, reduce=True, tile_n=args.tile_n if compact else None,
            )

            def logical():
                if not compact:
                    return storage
                return torch.cat([
                    b.view(ntiles, rows_per_dest, args.tile_n).permute(1, 0, 2)
                    .reshape(rows_per_dest, width)[:, :args.n] for b in buffers
                ])

            add(name, joined_x, joined_idx, table=single_table, out=buffers,
                store="bulk_rows_reduce", scatter=addresses, expected=expected, logical=logical,
                multi_buffer_scatter=True, n_tile_major_output=compact, control=control, **kwargs)

        add_destinations("multi-dest scatter add", control="scatter add")
        add_destinations("compact scatter add", compact=True, control="multi-dest scatter add")
        if args.weighted_contributions:
            add_destinations("weighted compact scatter", compact=True,
                             control="compact scatter add",
                             expected=weighted_reduced, route_weights=route_weights)

    if args.waves:
        num_work = len(single_rows) * args.group_size
        active = min(get_max_active_clusters(args.cluster_m, sm_count=args.sm_count), num_work)
        num_groups = math.ceil(args.n / args.tile_n) // args.group_size
        prefix = torch.tensor(build_wave_ready_prefix(
            single_rows, group_size=args.group_size, num_groups=num_groups, wave_work_ids=active,
        ), device=device, dtype=torch.int32)
        print(f"Wave geometry: {active} resident clusters, {len(prefix)} waves, "
              f"{num_groups} column groups", flush=True)

        def add_wave(name, *, publish=False, host=False, control, weighted=False):
            counter = torch.zeros(2 if publish else 1, device=device, dtype=torch.int32)
            kwargs = dict(wave_store_counter=counter)
            reset = lambda: None
            if publish:
                kwargs["wave_ready_prefix"] = prefix
            if host:
                address, word, reset = resources.enter_context(mapped_host_flag())
                kwargs["wave_ready_host_address"] = address

            def check_state():
                expected = [num_work * args.cluster_m] + ([num_groups] if publish else [])
                assert counter.cpu().tolist() == expected
                if host:
                    assert word.value == num_groups

            add_destinations(name, compact=True, control=control, prepare_extra=reset,
                             check_extra=check_state,
                             expected=weighted_reduced if weighted else reduced,
                             route_weights=route_weights if weighted else None, **kwargs)

        add_wave("wave barrier", control="compact scatter add")
        add_wave("GPU ready prefix", publish=True, control="wave barrier")
        if args.weighted_contributions:
            add_wave("weighted wave barrier", control="wave barrier", weighted=True)
            add_wave("weighted GPU ready", publish=True, control="GPU ready prefix", weighted=True)
        if args.host_ready:
            add_wave("host ready prefix", publish=True, host=True, control="GPU ready prefix")
            if args.weighted_contributions:
                add_wave("weighted host ready", publish=True, host=True,
                         control="host ready prefix", weighted=True)
        print("Wave timing includes the wrapper's GPU counter reset and cooperative launch; "
              "output and mapped-host flag resets are outside the interval.", flush=True)

    if args.swiglu:
        config = GemmConfig(
            tile_m=args.tile_m, tile_n=args.tile_n, cluster_m=args.cluster_m, cluster_n=1,
            tile_k=args.tile_k,
            max_swizzle_size=args.group_size, is_dynamic_persistent=False,
            pingpong=args.pingpong, swap_ab=False, device_capacity=device_capacity,
        )
        gate, up = reference.chunk(2, dim=-1)
        post_ref = F.silu(gate) * up
        weights_kn = w.transpose(1, 2)

        def add_swiglu(name, a, idx, *, table=None, control):
            post = torch.full((r, args.n // 2), float("nan"), device=device, dtype=dtype)

            def call():
                gemm_act(a, weights_kn, activation="swiglu", A_idx=idx,
                         cu_seqlens_m=cu if table is None else None, gather_work_table=table,
                         multi_buffer_gather=isinstance(a, tuple), postact_out=post,
                         store_preact=False, concat_layout=("B",), tuned=False, config=config,
                         sm_count=args.sm_count)

            variants[name] = Variant(call, lambda: check(post, post_ref), control=control)

        add_swiglu("indexed swiglu", joined_x, joined_idx, control="indexed baseline")
        add_swiglu("single table swiglu", joined_x, joined_idx,
                   table=single_table, control="indexed swiglu")
        add_swiglu("multi table swiglu", xs, indices,
                   table=multi_table, control="single table swiglu")
        print(f"SwiGLU: projection N={args.n}, output width={args.n // 2}, no preact store. "
              "TFLOPS counts GEMM operations only; the timed path includes activation.", flush=True)

    if args.profile_variant is not None:
        selected = select_variants(variants, [args.profile_variant])
        profile_variant(
            selected[args.profile_variant], name=args.profile_variant, warmup=args.warmup
        )
        return
    if args.variants is not None:
        variants = select_variants(variants, args.variants)
    measure(variants, flops=2 * r * args.n * args.k, rounds=args.rounds,
            repetitions=args.repetitions, warmup=args.warmup)
    print("Packing, metadata construction, and destination zeroing excluded; "
          "these are GEMM-path timings, not end-to-end proxy throughput.")


def main(argv=None, *, device_capacity=10):
    args = parse_args(argv, device_capacity=device_capacity)
    with ExitStack() as resources:
        if device_capacity == 10:
            if __package__:
                from . import blackwell_bench_utils as overrides
            else:
                import blackwell_bench_utils as overrides
            for name in ("bulk_smem_padding", "bulk_smem_cols", "bulk_smem_buffers",
                         "bulk_store_warps"):
                value = getattr(args, name)
                if value is not None:
                    resources.enter_context(getattr(overrides, name)(value))
            if args.print_kernel_config:
                resources.enter_context(overrides.kernel_config_trace())
        benchmark(args, resources)
