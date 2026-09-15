# Hopper bulk-row epilogue

`hopper_gather_gemm.py --epilogue-store bulk_rows` enables full-tile shared-memory
staging and linear `cp.async.bulk.global.shared::cta.bulk_group` stores. The
default `--epilogue-store tma` retains the existing tensor TMA epilogue.

```bash
python run/hopper_gather_gemm.py --tile-m 128 --tile-n 128 --cluster-m 2 --epilogue-store bulk_rows
python run/hopper_gather_gemm.py --tile-m 128 --tile-n 128 --cluster-m 2 --epilogue-store bulk_rows --pingpong
```

## Implementation

Each CTA stages its full output tile in unswizzled, row-major SMEM. Existing
epilogue arithmetic and register-to-SMEM partitioning still operate on subtiles,
but each subtile writes a different region of the full tile. After one
async-shared fence and epilogue barrier, all 32 lanes of the designated store
warp issue disjoint row copies: lane `i` copies rows `i`, `i+32`, etc. A full
128x128 BF16 tile therefore issues 128 copies of 256 bytes, four per lane.
Each lane commits its own bulk group. There is no elected-lane-only commit/wait
and no per-row wait.

Non-pingpong waits for every issuing lane's SMEM reads before staging the next
tile. Pingpong performs that wait before signaling the next warpgroup's epilogue
barrier. The waits allow source-buffer reuse without waiting for global writes;
all issuing lanes additionally drain their global writes at kernel exit. There
is only one full output buffer per CTA, including in pingpong mode.

The kernel predicates rows against the current expert's length and shortens
copies for the final N tile. The API requires row-major FP16/BF16/FP32 D,
16-byte-aligned base and row/batch strides, and a logical row width divisible
by 16 bytes. Partial N tiles satisfying those requirements are supported.

This first mode supports the primary D output of the low-level `quack.gemm.gemm`
API, including alpha/beta and C. Fused activations/auxiliary outputs, output
quantization, add-to-output, split-K, and gather work tables are not supported;
the runner rejects combining this mode with `--activation`.

Full output SMEM is charged to the A/B stage budget: 32 KiB for 128x128 BF16,
64 KiB for 256x128 BF16. Configurations leaving fewer than two A/B stages are
rejected because the current MMA mainloop waits for its next stage before
releasing the previous one. In particular, use the explicit 128x128 options
above rather than the runner's default 256x256 tile at its default K tile.

## Remote validation

Start with two BF16 numerical cases, then run the new coverage:

```bash
pytest tests/test_gemm_bulk_rows.py -x -k 'gather and bf16 and full_n'
pytest tests/test_gemm_bulk_rows.py -x
```

Tests compare output values with float32 PyTorch ground truth and a same-dtype
rounding baseline, as well as exact equality with the original TMA path. They
cover both pingpong modes, empty/ragged experts, N tails, padded output strides,
persistent SMEM reuse, changed-input CUDA graph replays, output guard regions,
FP32 output, and a batched linear epilogue with C.

For memory/synchronization checks (on a small runner input):

```bash
compute-sanitizer --tool memcheck python run/hopper_gather_gemm.py --tokens 257 --hidden 128 --output-dim 136 --experts 2 --routes 258 --tile-m 128 --tile-n 128 --cluster-m 2 --epilogue-store bulk_rows --iterations 2 --timing-samples 1 --no-cuda-graph
compute-sanitizer --tool racecheck python run/hopper_gather_gemm.py --tokens 257 --hidden 128 --output-dim 136 --experts 2 --routes 258 --tile-m 128 --tile-n 128 --cluster-m 2 --epilogue-store bulk_rows --pingpong --iterations 2 --timing-samples 1 --no-cuda-graph
```

For timing, repeat the first two runner commands with `--epilogue-store tma`,
keeping shapes, dtype, seed, and warmup identical. The runner uses CUDA graphs
by default. Alternate mode order across rounds on shared GPUs; these separate
process runs are a first comparison, not a tightly interleaved performance
measurement. Inspect epilogue duration and A/B stalls in a profiler before
attributing a change in whole-kernel time to the stores alone.

This mode reduces per-subtile synchronization but increases copy instruction
count, introduces unswizzled SMEM accesses, and may reduce A/B stage depth.
It is an experimental path; a speedup has not been established.
