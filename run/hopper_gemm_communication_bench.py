#!/usr/bin/env python3
"""Matched Hopper H100/H200 feature costs, checked before and after timing."""

if __package__:
    from . import gemm_communication_bench as shared
else:
    import gemm_communication_bench as shared

# Keep the standalone runner's numerical helpers available to existing callers.
benchmark = shared.benchmark
make_route_weights = shared.make_route_weights
scatter_reference = shared.scatter_reference


def parse_args(argv=None):
    return shared.parse_args(argv, device_capacity=9)


def main():
    shared.main(device_capacity=9)


if __name__ == "__main__":
    main()
