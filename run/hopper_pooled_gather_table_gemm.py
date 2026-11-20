#!/usr/bin/env python3
"""Time single-table Hopper GEMM on the logical inputs of a multi-buffer run.

Run the default eight-source workload from the repository root.

    python run/hopper_pooled_gather_table_gemm.py

The original runner's --num-input-buffers, --tokens-per-buffer,
--routes-per-buffer and routing flags apply to logical slices of one X tensor.
The single-buffer GEMM consumes pool-relative token IDs and four-column
descriptors. Both are prepared before graph capture.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import torch

if __package__:
    from . import hopper_gather_table_gemm as table
else:
    import hopper_gather_table_gemm as table


def parse_args(argv=None) -> argparse.Namespace:
    parser = table.make_arg_parser(
        description="Time the single-buffer Hopper gather kernel on pooled logical sources. "
                    "Multi-buffer routing is always enabled for input generation.",
        include_activation=False,
    )
    # Accept the original command's --multi-buffer-gather flag for logical input generation.
    parser.set_defaults(
        multi_buffer_gather=True, activation=None, down_projection=False, indexed_gather=False,
        num_input_buffers=8, tokens=2048, routes=4096, hidden=7168, output_dim=2048,
        tile_m=128, tile_n=256, experts=16, warmup=5, iterations=10, timing_samples=5,
    )
    return parser.parse_args(argv)


@torch.inference_mode()
def prepare_pooled_inputs(
    args: argparse.Namespace, device: torch.device
) -> tuple[table.TableGatherInputs, table.TableGatherInputs]:
    """Return the logical reference inputs and the single-buffer launch inputs sharing X and D."""
    tokens = args.tokens_per_buffer or args.tokens
    routes = args.routes_per_buffer or args.routes
    total_tokens = args.num_input_buffers * tokens
    total_routes = args.num_input_buffers * routes
    max_index = torch.iinfo(torch.int32).max
    if total_tokens - 1 > max_index or total_routes > max_index:
        raise ValueError("pooled token IDs and route bounds must fit in int32")
    pool = torch.empty((total_tokens, args.hidden), device=device, dtype=table.DTYPES[args.dtype])
    logical = table.prepare_inputs(args, device, input_pool=pool)

    # One entry per packed output row. Descriptor order can revisit a row for
    # another N group, whereas output_segments lists each route exactly once.
    source_and_token = table.build_original_token_mapping(logical)
    global_indices = (source_and_token[:, 0] * tokens + source_and_token[:, 1]).to(torch.int32)

    # The multi-buffer scheduler computes packed output bounds by summing all
    # source starts/ends, including nonzero starts of empty source ranges.
    multi_table = logical.work_table
    starts = multi_table[:, 2::2].sum(1, dtype=torch.int32)
    ends = multi_table[:, 3::2].sum(1, dtype=torch.int32)
    single_table = torch.stack((multi_table[:, 0], starts, ends, multi_table[:, 1]), dim=1)
    offsets = tuple(sum(bounds) for bounds in zip(*logical.route_offsets))
    pooled = replace(
        logical, X=pool, A_idx=global_indices, work_table=single_table, route_offsets=(offsets,),
        output_segments=tuple(
            (expert, 0, start, end)
            for expert, (start, end) in enumerate(zip(offsets, offsets[1:]))
        ),
    )
    return logical, pooled


def main() -> None:
    args = parse_args()
    table.validate_args(args)
    device = table.hopper_device(args.device)
    torch.manual_seed(args.seed)
    logical, pooled = prepare_pooled_inputs(args, device)
    print(
        f"Using {len(logical.X)} logical slices in one X tensor of shape {tuple(pooled.X.shape)}, "
        f"with {logical.X[0].shape[0]} token rows and {logical.A_idx[0].numel()} routes per slice."
    )
    single_args = argparse.Namespace(**vars(args))
    single_args.multi_buffer_gather = False
    # Check the shared output using the original local token IDs and logical X slices.
    table.benchmark_inputs(single_args, pooled, reference_inputs=logical)


if __name__ == "__main__":
    main()
