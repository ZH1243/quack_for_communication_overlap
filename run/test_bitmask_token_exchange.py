#!/usr/bin/env python3
"""Hopper numerical integration regression; launch with single-node torchrun.

Exercises batch dispatch, variable-sized return DMA, empty pairs/idle ranks,
N-tile-major groups, repeated buffer restoration and unchanged fixed dispatch.
The runner checks dispatch values, return bytes and GEMMs against references.
"""

import os
import sys

import torch

try:
    from run import bitmask_exchange
except ModuleNotFoundError as error:
    if error.name != "run":
        raise
    import bitmask_exchange

from hopper_stream_gather_table_gemm import main


if __name__ == "__main__":
    world = int(os.environ["LOCAL_WORLD_SIZE"])
    for probability, tokens, tile_major, streaming, swizzle in (
        (None, world * 32, False, False, 1),
        (0.25, 257, False, False, 1),
        (0.25, 257, True, True, 1),
        (0.25, 257, True, True, 2),
        (0.1, 3, True, True, 1),
        (0.25, 2, True, True, 1),
        (0.0, 17, True, True, 1),
        (1.0, 17, True, False, 1),
    ):
        sys.argv = [
            sys.argv[0], "--initial-token-exchange", "--return-token-results",
            "--source-tokens", str(tokens), "--multi-buffer-gather",
            "--balanced-multi-buffer-gather", "--round-robin-m-clusters",
            "--num-input-buffers", str(world), "--tokens-per-buffer", "32",
            "--routes-per-buffer", "64", "--hidden", "512", "--output-dim", "128",
            "--experts", "2", "--tile-m", "128", "--tile-n", "128", "--pingpong",
            "--activation", "swiglu", "--proxy-mode", "thread",
            "--flush-entries", "2", "--flush-interval-us", "5",
            "--down-projection2", "--down-combine-in-place",
            "--down-tile-m", "128", "--down-tile-n", "128", "--down-pingpong",
            "--down-gather-table", "--down-gather-table-expert-waves",
            "--down-max-swizzle-size", str(swizzle), "--down-gather-table-wave-barrier",
            "--down-gather-table-publish-ready", "--trace-down-ready",
            "--warmup", "1", "--iterations", "2", "--timing-samples", "1",
        ]
        if probability is not None:
            sys.argv.extend(["--bitmask-token-routing", "--token-route-probability",
                             str(probability), "--local-routes-per-token", "2"])
        if tile_major:
            sys.argv.append("--down-n-tile-major-output")
        if streaming:
            sys.argv.append("--return-token-results-per-group")
        original_generator = bitmask_exchange.make_router_metadata
        if tokens == 2:
            # Only source 1 sends data, to ranks 0 and 1. Both computing ranks
            # have an empty first input/output buffer; ranks >=2 are idle.
            def forced_metadata(tokens, world, probability, seed, device):
                source = seed - 104729  # This regression uses the default seed=0.
                mask = sum(1 << ((d - source - 1) % world) for d in (0, 1))
                return torch.full((tokens,), mask if source == 1 else 0,
                                  dtype=torch.uint8, device=device)
            bitmask_exchange.make_router_metadata = forced_metadata
        try:
            main()
        finally:
            bitmask_exchange.make_router_metadata = original_generator
        # Do not reuse a /dev/shm name until rank zero has unlinked the prior run.
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group("gloo")
        dist.barrier()
    dist.destroy_process_group()
