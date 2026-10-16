#!/usr/bin/env python3
"""Numerical return-copy regression for single-node torchrun on Hopper.

Exercises repeated generations, both layouts, and streaming groups of one/two N tiles. The runner
checks received bytes against peer outputs and GEMM values against PyTorch.
"""

import os
import sys

from hopper_stream_gather_table_gemm import main


if __name__ == "__main__":
    world = int(os.environ["LOCAL_WORLD_SIZE"])
    for tile_major, streaming, swizzle in (
        (False, False, 1), (True, False, 1), (True, True, 1), (True, True, 2)
    ):
        sys.argv = [
            sys.argv[0], "--initial-token-exchange", "--return-token-results",
            "--source-tokens", str(world * 2048), "--multi-buffer-gather",
            "--balanced-multi-buffer-gather", "--round-robin-m-clusters",
            "--num-input-buffers", str(world), "--tokens-per-buffer", "2048",
            "--routes-per-buffer", "4096", "--hidden", "512", "--output-dim", "128",
            "--experts", "2", "--tile-m", "128", "--tile-n", "128", "--pingpong",
            "--activation", "swiglu", "--proxy-mode", "thread",
            "--flush-entries", "1000", "--flush-interval-us", "5",
            "--down-projection2", "--down-combine-in-place",
            "--down-tile-m", "128", "--down-tile-n", "128", "--down-pingpong",
            "--down-gather-table", "--down-gather-table-expert-waves",
            "--down-max-swizzle-size", str(swizzle), "--down-gather-table-wave-barrier",
            "--down-gather-table-publish-ready", "--trace-down-ready",
            "--warmup", "1", "--iterations", "3", "--timing-samples", "1",
        ]
        if tile_major:
            sys.argv.append("--down-n-tile-major-output")
        if streaming:
            sys.argv.append("--return-token-results-per-group")
        main()
