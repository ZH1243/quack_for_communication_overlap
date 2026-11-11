#!/usr/bin/env python3
"""Matched Blackwell feature comparisons, numerically checked before and after timing."""

import argparse
import math
from contextlib import ExitStack

import torch
import torch.nn.functional as F

from quack.cute_dsl_utils import get_max_active_clusters
from quack.gemm import build_scatter_row_addresses, gemm
from quack.gemm_config import GemmConfig, cta_tile_shape_m
from quack.gemm_interface import gemm_act

from blackwell_bench_utils import (
    Variant, bulk_smem_buffers, bulk_smem_cols, bulk_smem_padding, bulk_store_warps,
    kernel_config_trace, mapped_host_flag, measure, profile_variant, select_variants,
)
from blackwell_gemm_tables import make_table
from wave_ready import build_wave_ready_prefix


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=1024,
                        help="GEMM projection width; SwiGLU output has N/2 columns")
    parser.add_argument("--buffers", type=int, default=16)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--routes-per-buffer", type=int, default=2048)
    parser.add_argument("--tokens-per-buffer", type=int, default=2048)
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--cluster-m", type=int, choices=(1, 2), default=2)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--sm-count", type=int, default=None)
    parser.add_argument("--packed-baseline", action="store_true",
                        help="also compare a non-gather grouped GEMM; packing is excluded")
    parser.add_argument("--inline-indices", action="store_true",
                        help="also compare work tables that embed token indices directly")
    parser.add_argument("--stores", action="store_true",
                        help="bulk copy, row scatter, reduction, multi-buffer and compact output")
    parser.add_argument("--readiness", action="store_true",
                        help="table-ready acquire checks with all rows already available")
    parser.add_argument("--waves", action="store_true",
                        help="completion barrier and GPU ready prefix; implies --stores")
    parser.add_argument("--host-ready", action="store_true",
                        help="mapped-host ready-prefix publication; implies --waves")
    parser.add_argument("--swiglu", action="store_true",
                        help="compare indexed/table fused SwiGLU with store_preact=False")
    parser.add_argument("--baseline-only", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--variants", nargs="+", metavar="NAME",
                           help="time only these enabled variants; first becomes the baseline")
    selection.add_argument("--profile-variant", metavar="NAME",
                           help="check/warm one enabled variant, then launch in NVTX quack_profile")
    parser.add_argument("--print-kernel-config", action="store_true",
                        help="print compiled stages/layouts; requires QUACK_CACHE_ENABLED=0")
    parser.add_argument("--bulk-smem-padding", type=int, choices=(0, 16), default=None,
                        help="bulk-row padding ablation in bytes; requires QUACK_CACHE_ENABLED=0")
    parser.add_argument("--bulk-smem-cols", type=int, choices=(0, 64, 128, 256), default=None,
                        help="total buffered column limit, 0=full; requires QUACK_CACHE_ENABLED=0")
    parser.add_argument("--bulk-smem-buffers", type=int, choices=(1, 2), default=None,
                        help="split the column budget into reusable windows; requires cache disabled")
    parser.add_argument("--bulk-store-warps", type=int, choices=(1, 2, 4), default=None,
                        help="row-copy issuing warps within the epilogue; requires cache disabled")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--repetitions", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=100, help="full graph replays")
    args = parser.parse_args()
    args.waves = args.waves or args.host_ready
    args.stores = args.stores or args.waves
    if min(args.k, args.n, args.buffers, args.experts, args.routes_per_buffer,
           args.tokens_per_buffer, args.tile_m, args.tile_n, args.group_size,
           args.rounds, args.repetitions) <= 0 or args.warmup < 0:
        parser.error("dimensions/counts must be positive and warmup nonnegative")
    if args.buffers < 2 and not args.baseline_only:
        parser.error("the multi-buffer comparison requires at least two buffers")
    if args.baseline_only and (args.stores or args.readiness or args.swiglu or args.inline_indices):
        parser.error("--baseline-only cannot be combined with feature comparisons")
    if args.sm_count is not None and (args.sm_count <= 0 or args.sm_count % args.cluster_m):
        parser.error("sm-count must be positive and divisible by cluster-m")
    if args.swiglu and (args.n % 2 or args.sm_count is not None):
        parser.error("SwiGLU requires even N; gemm_act does not expose an SM-count limit")
    if args.stores and args.n % 8:
        parser.error("bulk stores require BF16 row widths divisible by 8")
    if args.bulk_smem_padding is not None and not args.stores:
        parser.error("--bulk-smem-padding requires --stores, --waves, or --host-ready")
    if args.bulk_smem_cols is not None and not args.stores:
        parser.error("--bulk-smem-cols requires --stores, --waves, or --host-ready")
    if args.bulk_smem_buffers is not None and not args.stores:
        parser.error("--bulk-smem-buffers requires --stores, --waves, or --host-ready")
    if args.bulk_store_warps is not None and not args.stores:
        parser.error("--bulk-store-warps requires --stores, --waves, or --host-ready")
    ntiles = math.ceil(args.n / args.tile_n)
    if ntiles % args.group_size:
        parser.error("N tile count must be divisible by group-size")
    return args


def benchmark(args, resources):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("run this benchmark on a B200/B300 node")
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
                  persistent=True, is_dynamic_persistent=False, pingpong=False,
                  use_tma_gather=False, sm_count=args.sm_count)
    print(f"{torch.cuda.get_device_name()}: dtype={dtype}, R={r} N={args.n} K={args.k}, "
          f"experts={args.experts}, buffers={args.buffers}", flush=True)
    print(f"Routes per expert: {expert_counts}", flush=True)
    print(f"Shared config: MMA tile=({args.tile_m},{args.tile_n}), tile_K=auto, "
          f"cluster=({args.cluster_m},1), group_size={args.group_size}, "
          f"sm_count={args.sm_count if args.sm_count is not None else 'device default'}",
          flush=True)
    print("Shared config: static persistent, swap_ab=False, pingpong=False; "
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

    if args.inline_indices:
        cluster_rows = cta_tile_shape_m(args.tile_m, args.cluster_m, 10) * args.cluster_m
        tokens = joined_idx.cpu().tolist()
        # Indexed descriptors require consecutive N-group rows per M bundle.
        inline_rows = [[e, ng, start, end, *tokens[start:end],
                        *([-1] * (cluster_rows - (end - start)))]
                       for e, start, end, ng in sorted(single_rows)]
        inline_table = torch.tensor(inline_rows, device=device, dtype=torch.int32)
        unused_idx = torch.full_like(joined_idx, -1)
        add("inline-index table", joined_x, unused_idx, table=inline_table, control="single table")

    if args.readiness:
        single_ready = torch.full((1,), len(single_rows), device=device, dtype=torch.int32)
        multi_ready = torch.full((1,), len(multi_rows), device=device, dtype=torch.int32)
        add("single table ready", joined_x, joined_idx, table=single_table,
            gather_work_table_ready=single_ready, control="single table")
        add("multi table ready", xs, indices, table=multi_table,
            gather_work_table_ready=multi_ready, control="multi table")
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
        reduced = torch.zeros_like(reference).index_add_(
            0, mapping.long(), reference.to(dtype).float()
        )
        add("scatter add", joined_x, joined_idx, table=single_table,
            store="bulk_rows_reduce", scatter=mapping, expected=reduced, control="reduce unique")
        # Keep global destination row order identical across single/multi-buffer
        # scatter. Destinations are equal-sized views of one allocation, avoiding
        # an allocation-placement change when measuring address representation.
        rows_per_dest = args.routes_per_buffer
        dest_mapping = torch.stack((mapping // rows_per_dest, mapping % rows_per_dest), dim=1)
        ntiles = math.ceil(args.n / args.tile_n)

        def add_destinations(name, *, compact=False, control, **kwargs):
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
                store="bulk_rows_reduce", scatter=addresses, expected=reduced, logical=logical,
                multi_buffer_scatter=True, n_tile_major_output=compact, control=control, **kwargs)

        add_destinations("multi-dest scatter add", control="scatter add")
        add_destinations("compact scatter add", compact=True, control="multi-dest scatter add")

    if args.waves:
        num_work = len(single_rows) * args.group_size
        active = min(get_max_active_clusters(args.cluster_m, sm_count=args.sm_count), num_work)
        num_groups = math.ceil(args.n / args.tile_n) // args.group_size
        prefix = torch.tensor(build_wave_ready_prefix(
            single_rows, group_size=args.group_size, num_groups=num_groups, wave_work_ids=active,
        ), device=device, dtype=torch.int32)
        print(f"Wave geometry: {active} resident clusters, {len(prefix)} waves, "
              f"{num_groups} column groups", flush=True)

        def add_wave(name, *, publish=False, host=False, control):
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
                             check_extra=check_state, **kwargs)

        add_wave("wave barrier", control="compact scatter add")
        add_wave("GPU ready prefix", publish=True, control="wave barrier")
        if args.host_ready:
            add_wave("host ready prefix", publish=True, host=True, control="GPU ready prefix")
        print("Wave timing includes the wrapper's GPU counter reset and cooperative launch; "
              "output and mapped-host flag resets are outside the interval.", flush=True)

    if args.swiglu:
        config = GemmConfig(
            tile_m=args.tile_m, tile_n=args.tile_n, cluster_m=args.cluster_m, cluster_n=1,
            max_swizzle_size=args.group_size, is_dynamic_persistent=False,
            pingpong=False, swap_ab=False, device_capacity=10,
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
                         store_preact=False, concat_layout=("B",), tuned=False, config=config)

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


def main():
    args = parse_args()
    with ExitStack() as resources:
        if args.bulk_smem_padding is not None:
            resources.enter_context(bulk_smem_padding(args.bulk_smem_padding))
        if args.bulk_smem_cols is not None:
            resources.enter_context(bulk_smem_cols(args.bulk_smem_cols))
        if args.bulk_smem_buffers is not None:
            resources.enter_context(bulk_smem_buffers(args.bulk_smem_buffers))
        if args.bulk_store_warps is not None:
            resources.enter_context(bulk_store_warps(args.bulk_store_warps))
        if args.print_kernel_config:
            resources.enter_context(kernel_config_trace())
        benchmark(args, resources)


if __name__ == "__main__":
    main()
