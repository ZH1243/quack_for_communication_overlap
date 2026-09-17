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
quantization, add-to-output, and split-K are not supported;
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


## Bulk-row reduce-add

Use the same full-tile staging with one elementwise reduction per valid row:

```bash
python run/hopper_gather_gemm.py --tile-m 128 --tile-n 128 --cluster-m 2 --epilogue-store bulk_rows_reduce
python run/hopper_gather_gemm.py --tile-m 128 --tile-n 128 --cluster-m 2 --epilogue-store bulk_rows_reduce --pingpong
```

For BF16/FP16 this issues
`cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16` / `.f16`.
The SMEM layout, lane-to-row assignment, boundary predicates, and bulk-group
commit/wait protocol are shared with `bulk_rows`.

At the low-level API, `gemm(..., epilogue_store="bulk_rows_reduce")` adds the
converted epilogue result into the existing D values. It does **not** zero D.
Callers wanting ordinary GEMM results must zero D before each invocation on the
same stream (or establish an equivalent dependency). This mode inherits the
bulk-row restrictions; `add_to_output=True` remains unsupported because the
mode itself specifies accumulation. FP32 output uses `.add.f32`; on CUDA 12.9,
subnormal inputs/results may flush to zero. FP16/BF16 use `.noftz`.

The runner performs the reset automatically before **every** warmup and timed
GEMM, including every iteration within every graph replay. It leaves the final
result available for the reference check. For reduction timing, each iteration
is ordered on one stream as:

```text
zero output -> start event -> GEMM -> end event
```

Each sample averages only the per-GEMM event intervals. Reported effective
TFLOP/s therefore excludes zeroing, without subtracting a separately measured
reset cost. CUDA graph capture uses external timing events so the event-record
nodes run on every replay. Ordinary modes retain their existing batched timing.
Keep CUDA graphs enabled for performance comparisons: `--no-cuda-graph` still
excludes zeroing but can include host enqueue gaps in the per-GEMM intervals.
Zeroing can affect cache residency even though its duration is excluded, and
per-GEMM event instrumentation differs from the original batched timing.

Numerical coverage includes accumulation into nonzero D (to detect an accidental
plain store), zero-before-replay, ragged experts, N tails, padded strides,
FP16/BF16/FP32 output, both pingpong modes, and runner graph/direct timing:

```bash
pytest tests/test_gemm_bulk_rows.py -x -k 'reduce_benchmark'
pytest tests/test_gemm_bulk_rows.py -x
```


## Scattered output rows in the pre-gather runner

`hopper_pregather_gemm.py --scatter-table` creates a seeded CUDA int32 permutation
of the R routed rows, outside timing. Row `i` of pre-gathered A writes to
`output[scatter_table[i]]`. The mapping spans all experts and works both with
`--gather-table` and the default `cu_seqlens_m` scheduler. It requires
`--epilogue-store bulk_rows` or `bulk_rows_reduce`; omitting the flag retains the
original output order and avoids the mapping loads.

```bash
python run/hopper_pregather_gemm.py --gather-table --scatter-table \
    --tile-m 128 --tile-n 128 --epilogue-store bulk_rows --pingpong
python run/hopper_pregather_gemm.py --gather-table --scatter-table \
    --tile-m 128 --tile-n 128 --epilogue-store bulk_rows_reduce --pingpong
```

The store warp reads adjacent scatter entries into registers before epilogue
output staging (four indices per lane for tile M=128, eight for M=256), then
issues the same number and size of bulk operations. No additional shared-memory
buffer, barrier, or per-row wait is introduced. Stores use the original D base
and its actual row stride; predicates still use the source descriptor/expert
bounds. Index reads and destination-address calculation are included in timing.
Random destinations can affect memory locality; compare the same command with
and without `--scatter-table` on an otherwise idle Hopper before drawing
performance conclusions.

The low-level `quack.gemm.gemm(..., scatter_table=...)` argument currently supports
plain grouped GEMM without C/bias, with rank-2 output, and either cu_seqlens_m or
a single-buffer four-column gather table. The caller must supply a contiguous
CUDA int32 permutation of `[0, D.shape[0])` on the output device. Its contents
are not scanned during launches, avoiding a synchronization or validation kernel
in the timed/captured path. Duplicate destinations are outside this contract.
Reduction mode still requires output initialization before every launch; the
runner excludes its zeroing from timing as before.

Start with a small numerical subset, then run the full bulk-row suite:

```bash
pytest tests/test_gemm_bulk_rows.py -x -k 'scatter and table_n_group_major and cooperative and bf16'
pytest tests/test_gemm_bulk_rows.py -x
```

Scatter coverage includes identity/random cross-expert permutations, M/N tails,
empty experts, both scheduling orders, cluster offsets, pingpong, FP16/BF16,
padded output views, reduction into nonzero destinations, persistent reuse,
and mapping changes across launches and graph replays. Comparisons use float32
PyTorch references and the original TMA output path.
