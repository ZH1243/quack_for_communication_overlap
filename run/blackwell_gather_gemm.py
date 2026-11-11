#!/usr/bin/env python3
"""Benchmark indexed token-gather GEMM on Blackwell B200/B300 (SM10x).

Reuse the Hopper runner's uniform MoE routing, numerical references and timing
driver. The SM100 kernel supports cp.async gather (default), --use-tma-gather,
fused --activation and optional --down-projection. Dynamic persistent scheduling
is enabled by default; --no-dynamic-persistent selects static scheduling.

Example:
    python run/blackwell_gather_gemm.py \
        --tokens 4096 --hidden 2048 --output-dim 4096 \
        --experts 8 --routes 8192 --use-tma-gather

No gather/scatter work tables, streamed readiness, bulk-row stores, or Hopper
ping-pong scheduling are exposed. Output stays in expert-contiguous route order;
the optional down projection consumes it without gathering again.
"""

if __package__:
    from . import hopper_gather_gemm as shared
else:
    import hopper_gather_gemm as shared


def parse_args(argv=None):
    return shared.parse_args(argv, blackwell=True)


def main() -> None:
    shared.run(parse_args())


if __name__ == "__main__":
    main()
