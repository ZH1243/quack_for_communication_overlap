# Hopper gather GEMM runners

This directory contains four standalone benchmark runners for QuACK's grouped
GEMM on NVIDIA Hopper GPUs (SM90). The original pair uses uniform
Mixture-of-Experts (MoE) routing; the table-scheduled runner also supports
uneven expert lengths.

## Scripts

### `hopper_gather_gemm.py`

Benchmarks the fused gather-A path. The GEMM receives the original token tensor
`X` and the routing indices `A_idx`, then gathers rows of `X` while executing the
GEMM through QuACK's `cp.async` gather path.

Tensor shapes are:

- `X`: `[T, K]`
- `W`: `[E, K, N]`
- `A_idx`: `[R]`
- `cu_seqlens_m`: `[E + 1]`
- output: `[R, N]`

### `hopper_pregather_gemm.py`

Benchmarks the non-fused counterpart. Before timing begins, it materializes
`A = X[A_idx]` as a contiguous `[R, K]` tensor. The timed grouped GEMM reads
this tensor through the ordinary TMA A-load path.

The reported time **does not include** the pre-gather operation. Consequently,
this runner measures the GEMM after pre-gathering, not the full pre-gather plus
GEMM pipeline. Keep this distinction in mind when comparing the two scripts.

### `hopper_gather_table_gemm.py`

Benchmarks the new Hopper table-scheduled gather-A path. It passes no
`cu_seqlens_m`; instead, a GPU-resident contiguous int32 `[Q, 4]` table stores
`(expert_id, route_start, route_end, cid_n_base)`. Each row expands to
`x = min(max_swizzle_size, ceil(N / tile_N))` consecutive AlongN work IDs.
The runner distributes routes near-evenly across experts, including true
varlen-M when `R` is not divisible by `E`.

It also has an opt-in multi-buffer mode. Each `X_j` and `A_idx_j` is a
separate CUDA allocation, and the kernel gathers from them directly without
materializing `torch.cat(X_j)`:

```bash
python run/hopper_gather_table_gemm.py --multi-buffer-gather \
    --num-input-buffers 3 --tokens-per-buffer 4096 \
    --routes-per-buffer 8195 --hidden 4096 --output-dim 4096 --experts 8
```

The multi-buffer work table has shape `[Q, 2 + 2*b]`, with rows
`(expert_id, cid_n_base, start_0, end_0, ..., start_b-1, end_b-1)`. Output is
expert-major, with routes ordered by input-buffer index inside each expert.
`--round-robin-m-clusters` changes only the scheduling order: all N-group rows
for an expert/M-cluster stay consecutive, and those bundles are interleaved
across experts. It can be combined independently with
`--balanced-multi-buffer-gather`, which controls route allocation within each
M-cluster rather than bundle ordering.

All runners exclude kernel compilation, warmup, CUDA graph capture, and the
correctness check from the reported kernel time.

### `hopper_stream_gather_table_gemm.py`

Runs the single- or multi-buffer table kernel while `run/cpu_proxy/stream_gather_proxy`
streams the gather table from CUDA-pinned CPU memory into a poisoned HBM table
allocation. The proxy publishes the number of ready table rows after
every copy batch. Before reading row `i`, lane 0 of the scheduler warp waits for
`ready_rows[0] >= i + 1`; it then broadcasts the row through the existing
scheduler path. All GEMM, gather, and epilogue behavior is otherwise unchanged.

To run an independent copy on each of eight local GPUs, use `torchrun`:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
    --log-dir logs/stream_gather --tee 3 \
    run/hopper_stream_gather_table_gemm.py --proxy-mode thread \
    --multi-buffer-gather --balanced-multi-buffer-gather --round-robin-m-clusters \
    --num-input-buffers 16 --tokens-per-buffer 2048 --routes-per-buffer 4096 \
    --hidden 4096 --output-dim 1024 --tile-m 128 --tile-n 128 --experts 8 \
    --warmup 5 --iterations 10 --timing-samples 5 --pingpong --activation swiglu \
    --flush-entries 1000 --flush-interval-us 5 --flag-update-mode memcpy \
    --down-projection2 --down-combine-in-place --down-n-tile-major-output \
    --down-tile-m 128 --down-tile-n 128 --down-pingpong \
    --down-gather-table --down-gather-table-expert-waves --down-max-swizzle-size 1 \
    --down-gather-table-wave-barrier --down-gather-table-publish-ready --trace-down-ready
```

The launcher's `LOCAL_RANK` selects the CUDA device for both the kernel and its
proxy, overriding `--device`. Plain `python` launches still use `--device` (default
0). Device indices refer to the `CUDA_VISIBLE_DEVICES` ordering; set that variable
and reduce `--nproc-per-node` to run on a subset. `--tee 3` labels console output
by worker and saves each worker's stdout/stderr under the log directory.

With `--proxy-mode thread`, this creates eight Python worker processes, each with
its own CPU proxy thread, GPU buffers, correctness checks, and timing results.
All dimensions and the expert count apply **per GPU**. No process group, inter-GPU
data exchange, or iteration barriers are used; workers progress independently,
so the results are not a synchronized eight-GPU throughput measurement.

Build the C++ proxy first:

```bash
cmake -S run/cpu_proxy -B run/cpu_proxy/build
cmake --build run/cpu_proxy/build -j
```

Single-buffer gather is the default and uses `--tokens` and `--routes`:

```bash
python run/hopper_stream_gather_table_gemm.py --proxy-mode thread \
    --tokens 4096 --routes 8195 --hidden 4096 --output-dim 4096 --experts 8 \
    --flush-entries 2 --flush-interval-us 10
```

Both proxy modes also support `--activation` and `--down-projection` in single-buffer
mode. Rebuild both proxy artifacts after updating from a multi-buffer-only version.
The `--num-input-buffers`, `--tokens-per-buffer`, and `--routes-per-buffer` options
apply only with `--multi-buffer-gather`.

For multiple buffers:

```bash
python run/hopper_stream_gather_table_gemm.py --multi-buffer-gather \
    --num-input-buffers 3 --tokens-per-buffer 4096 \
    --routes-per-buffer 8195 --hidden 4096 --output-dim 4096 --experts 8 \
    --flush-entries 2 --flush-interval-us 10 --flag-update-mode memcpy
```

For single-buffer indexed gather, add `--indexed-gather`:

```bash
python run/hopper_stream_gather_table_gemm.py --indexed-gather --proxy-mode thread \
    --tokens 4096 --routes 8195 --hidden 4096 --output-dim 8192 \
    --tile-m 128 --tile-n 128 --pingpong \
    --flush-entries 50 --flush-interval-us 5
```

Indexed mode works with both proxy modes and with activation/down projection.
It streams `[expert, n_base, output_start, output_end, token_0, ...]` rows with
`tile_m * cluster_m` token slots and `-1` tail padding. Output uses the explicit
packed ranges, with no padding between experts. It cannot be combined with
`--multi-buffer-gather`, balanced allocation, or round-robin scheduling.
Rebuild the proxy executable and shared library after updating the sources.

To run the C++ producer on a worker thread in the Python process, add:

```bash
python run/hopper_stream_gather_table_gemm.py --multi-buffer-gather \
    --proxy-mode thread --flush-entries 2 --flush-interval-us 20
```

Thread mode loads `run/cpu_proxy/build/libstream_gather_proxy_thread.so`, uses
the PyTorch table and readiness pointers directly, and creates its private CUDA
stream in PyTorch's primary CUDA context. It therefore needs neither CUDA IPC,
MPS, nor `--dma-kick-bytes` to avoid cross-context scheduling. The shared
library owns one persistent native C++ worker thread, so Python can wait for
the QuACK kernel concurrently.

Use `--flag-update-mode stream-write` to publish the flag with
`cuStreamWriteValue32` instead. The default `memcpy` mode uses
`cudaMemcpyAsync`. In both modes, row copies and flag writes share the proxy's
nonblocking CUDA stream, so observing a prefix also orders the corresponding
table data before the scheduler's system-scope acquire load.

The proxy prepares its pinned table once and stays alive across warmup and
timed launches. Before each launch, Python poisons the old HBM rows, resets the
ready count, and launches the persistent kernel while the proxy is idle. Timing
starts immediately before `GO` (when the proxy is about to issue its first
batch) and ends when the GroupedGEMM finishes. Proxy startup and table
construction are excluded; table transfer, configured inter-batch delays,
scheduler polling, and GEMM are included. CUDA graphs are not used for this
external-producer protocol.

In the default process mode, the proxy is a separate CUDA process. On systems
without CUDA MPS, sufficiently small HtoD copies can take CUDA's front-end
inline path and wait behind the persistent kernel's CUDA context. Use thread
mode or run process mode under MPS for true overlap. As a diagnostic or
fallback, pass `--dma-kick-bytes 65536`; the proxy then issues one 64 KiB scratch
HtoD transfer before each flush sequence. The scratch copy is timed but does
not alter the ready-row prefix.

## Requirements

- An NVIDIA Hopper GPU with compute capability 9.x (for example, H100)
- A working CUDA-enabled PyTorch installation
- QuACK installed from this repository

From the repository root, a development installation can be created with:

```bash
pip install -e '.[dev]'
```

Run the scripts from the repository root so the commands and imports below
resolve consistently.

## Running the benchmarks

Run the fused gather benchmark with its default settings:

```bash
python run/hopper_gather_gemm.py
```

Run the pre-gathered benchmark with the same settings:

```bash
python run/hopper_pregather_gemm.py
```

To use global serpentine column-group traversal with expert work waves:

```bash
python run/hopper_pregather_gemm.py --gather-table --gather-table-expert-waves \
    --tile-m 128 --tile-n 128 --cluster-m 2 --max-swizzle-size 1
```

This mode fills each logical wave with the current expert, borrowing from its
later column groups as necessary. Only after exhausting all of that expert's
work may the next expert share the wave. Already assigned work is skipped on
subsequent serpentine passes. The wave size is the launched persistent-cluster
count, which must be divisible by the effective swizzle size; otherwise the
runner raises an error. Logical waves do not synchronize clusters on the GPU.
The flag takes precedence over `--gather-table-n-group-major` if both are set.
Without the new flag, existing table orderings are unchanged.

To order the output stores of successive waves, add `--gather-table-wave-barrier`:

```bash
python run/hopper_pregather_gemm.py --gather-table --gather-table-expert-waves \
    --gather-table-wave-barrier --pingpong --epilogue-store bulk_rows \
    --tile-m 128 --tile-n 128 --cluster-m 2 --max-swizzle-size 1
```

This requires expert-wave scheduling, pingpong, and `bulk_rows` or
`bulk_rows_reduce`. Each store-warp lane waits for its tile's global writes to
complete. After a warp synchronization, one lane increments a GPU-scope release
counter once per CTA tile, even for an entirely predicated tile. Before the next
wave enters its epilogue, it acquire-waits for all preceding-wave CTA tiles.
Operand loading and MMA may still overlap across waves. Global-memory completion
does not mean a physical flush past L2 or establish an RDMA notification protocol.

The barrier path uses a cooperative CUDA launch with PDL disabled, ensuring that
waiting CTAs do not prevent other participants from running. CUDA rejects a grid
that cannot meet cooperative-launch residency requirements; there is no fallback
to an ordinary spinning launch. The counter is private to the runner's launch
closure and reset on the same stream before **every** GEMM, including each
invocation inside CUDA Graph capture/replay. Counter reset and barrier overhead
are included in the reported time. Output zeroing for `bulk_rows_reduce` remains
excluded as before. Expect reduced overlap and sensitivity to the slowest CTA.

The low-level `quack.gemm.gemm` API enables this behavior with
`wave_store_counter=counter`, where `counter` is a separate contiguous CUDA
`torch.int32` tensor of shape `[1]`. The API resets it on every launch and leaves
the completed CTA-tile count there afterward. Do not share the workspace between
concurrent invocations. This initial implementation accepts only static,
single-buffer, non-streaming four-column gather tables with pingpong bulk stores.
At this level the barrier works for any table order, not only expert-wave order.

Hopper numerical regression checks (including partial waves and graph replay):

```bash
pytest tests/test_gemm_wave_store_barrier.py -x
```

Run the table-scheduled fused gather benchmark:

```bash
python run/hopper_gather_table_gemm.py
python run/hopper_gather_table_gemm.py --pingpong --tile-m 64 --tile-n 256
python run/hopper_stream_gather_table_gemm.py --multi-buffer-gather
```

An explicit example suitable for comparing their reported GEMM times is:

```bash
python run/hopper_gather_gemm.py \
    --tokens 4096 --hidden 4096 --output-dim 4096 \
    --experts 8 --routes 8192 \
    --warmup 5 --iterations 100 --timing-samples 5

python run/hopper_pregather_gemm.py \
    --tokens 4096 --hidden 4096 --output-dim 4096 \
    --experts 8 --routes 8192 \
    --warmup 5 --iterations 100 --timing-samples 5

python run/hopper_gather_table_gemm.py \
    --tokens 4096 --hidden 4096 --output-dim 4096 \
    --experts 8 --routes 8195 \
    --warmup 5 --iterations 100 --timing-samples 5
```

The first run of a new configuration may pause while QuACK compiles a
specialized kernel. Each script prints the median per-kernel time, all timing
samples, effective throughput, and correctness-check result.

Use `--help` to see the command-line interface directly:

```bash
python run/hopper_gather_gemm.py --help
python run/hopper_pregather_gemm.py --help
python run/hopper_gather_table_gemm.py --help
python run/hopper_stream_gather_table_gemm.py --help
```

## Runtime parameters

The original fused and pre-gathered scripts accept the same parameters. The
table runner accepts the applicable static-persistent subset and permits
`--routes` values not divisible by the expert count. Its ping-pong mode uses
the SM90 ping-pong tile limits: tile M must be 64, 128, or 192, and the maximum
tile N is respectively 256, 208, or 128.

| Parameter | Default | Description |
| --- | ---: | --- |
| `--tokens`, `-T` | `4096` | Number of rows in the source token tensor (`T`). |
| `--hidden`, `-K` | `4096` | Input/hidden dimension (`K`). |
| `--output-dim`, `-N` | `4096` | GEMM output dimension (`N`). |
| `--experts`, `-E` | `8` | Number of experts (`E`). |
| `--routes`, `-R` | `8192` | Total routed token assignments (`R`). Must be divisible by `--experts`. |
| `--multi-buffer-gather` | off | Read separate `X_j`/`A_idx_j` allocations through the wider gather table. |
| `--num-input-buffers` | `2` | Number of input buffers in multi-buffer mode. |
| `--tokens-per-buffer` | `--tokens` | Number of rows in each `X_j`. |
| `--routes-per-buffer` | `--routes` | Number of entries in each `A_idx_j`. |
| `--dtype` | `bf16` | Input/output dtype: `bf16` or `fp16`. |
| `--device` | `0` | CUDA device index. |
| `--seed` | `0` | Random seed used to create inputs and routing. |
| `--tile-m` | `256` | GEMM M tile size. |
| `--tile-n` | `256` | GEMM N tile size. |
| `--tile-k` | automatic | GEMM K tile size; QuACK selects it when omitted. |
| `--cluster-m` | `2` | Thread-block cluster extent along M. N and K cluster extents are fixed at 1. |
| `--max-swizzle-size` | `8` | Maximum persistent tile-scheduler swizzle size. |
| `--pingpong` | off | Enable the ping-pong kernel schedule. |
| `--routing-with-replacement` | off | Permit a token to be selected multiple times for one expert. |
| `--warmup` | `5` | Number of untimed warmup launches. |
| `--iterations` | `100` | Number of GEMM launches in each timing sample. |
| `--timing-samples` | `5` | Number of timing samples used to calculate the median. |
| `--no-cuda-graph` | off | Use ordinary batched launches instead of CUDA graph replay. This is less reliable for short kernels. |
| `--skip-check` | off | Skip comparison against the float32 PyTorch reference. |
| `--atol` | `0.03` | Absolute tolerance for the correctness check. |
| `--rtol` | `0.001` | Relative tolerance for the correctness check. |

The streaming runner accepts the table runner's parameters and additionally
accepts `--flush-entries` (default `1`), `--flush-interval-us` (default `10`),
`--flag-update-mode {memcpy,stream-write}`, `--dma-kick-bytes` (default `0`),
`--proxy-mode {process,thread}` (default `process`), `--proxy-binary`, and
`--thread-proxy-library`. It requires `--multi-buffer-gather`.

Routing is uniform: every expert receives `R / E` assignments. Without
`--routing-with-replacement`, `R / E` cannot exceed `T`. A token is still
allowed to appear under different experts, as in top-k MoE routing.

For more stable timing of short kernels, CUDA graphs are enabled by default.
Use identical shapes, dtypes, kernel settings, and timing settings when
comparing the two runners.

### Direct token indices in the Hopper gather table

`hopper_gather_table_gemm.py --indexed-gather` enables a single-buffer table of
shape `int32[Q, 4 + tile_m * cluster_m]`. Each row is
`(expert_id, cid_n_base, output_start, output_end, token_idx_0, ..., token_idx_{C-1})`, where
`C = tile_m * cluster_m`. Token indices directly address `X[T, K]` and may be
nonconsecutive or repeated. With the same `--seed` and other input options,
both modes generate identical X, weights, and per-expert routing. Each indexed
entry embeds exactly `A_idx[route_start:route_end]` from its corresponding legacy
entry; enabling the flag does not resample tokens. Table ordering differs between modes; both have packed output. Valid indices must be in `[0, T)`; partial tiles
have a valid prefix followed by `-1` padding.

All N groups for an M-cluster must appear consecutively, in increasing
`cid_n_base` order (`0, x, 2*x, ...`), where
`x = min(max_swizzle_size, ceil(gemm_N / tile_n))`. These rows must carry the
same expert, output range, and token indices. Each bundle writes only
`[output_start, output_end)`. Experts remain consecutive with no output padding,
including for the optional down projection.
For gated activations, `gemm_N` is the preactivation width, twice `output_dim`.

The Python GEMM APIs select this format by its table width. They still require
an `A_idx` vector with the actual output length for sizing, but the kernel does
not read its values. The runner keeps it for reference checks.
`--indexed-gather` cannot be combined with `--multi-buffer-gather`. Without
this flag, the existing four-column table and packed output behavior remain.

```bash
python run/hopper_gather_table_gemm.py --indexed-gather \
    --tokens 257 --routes 421 --experts 3 --hidden 80 --output-dim 768 \
    --tile-m 64 --tile-n 128 --cluster-m 2 --max-swizzle-size 2 --pingpong

python run/hopper_gather_table_gemm.py --indexed-gather \
    --tokens 257 --routes 421 --experts 3 --hidden 80 --output-dim 768 \
    --tile-m 64 --tile-n 128 --cluster-m 2 --max-swizzle-size 2 --pingpong \
    --activation swiglu --down-projection

pytest tests/test_gemm_gather_table.py -k indexed_gather -x
```

### Gather/activation followed by configurable down projection

`hopper_gather_table_gemm.py --down-projection2` reuses the plain grouped GEMM
setup and launch from `hopper_pregather_gemm.py`. It consumes the contiguous
activated intermediate directly, preserving the actual (possibly unequal)
expert row counts. It is mutually exclusive with `--down-projection` and
requires `--activation`.

All plain-GEMM kernel options from the pregather runner are available with a
`--down-` prefix and independent defaults. For example:

```bash
python run/hopper_gather_table_gemm.py \
    --tokens 4096 --hidden 4096 --output-dim 14336 \
    --experts 8 --routes 8195 --activation swiglu --down-projection2 \
    --down-tile-m 128 --down-tile-n 128 --down-pingpong \
    --down-scatter-table --down-n-tile-major-output \
    --down-gather-table --down-gather-table-expert-waves \
    --down-max-swizzle-size 1 --down-gather-table-wave-barrier \
    --down-epilogue-store bulk_rows
```

Down options include tile/cluster sizes, swizzle, pingpong, both table traversal
orders, wave barriers, all epilogue store modes, single/multiple scatter buffers,
N-tile-major output, used/capacity rows, and scatter with replacement. Existing
compatibility constraints still apply. The down table uses identity input
indices and the final hidden width; it is independent of the up table, including
when up uses indexed or multi-buffer gather.

The scatter destinations are benchmark-generated permutations, or random
indices with `--down-scatter-table-with-replacement`; they do not automatically
implement the original MoE routing or routing weights. Duplicate destinations
require `--down-epilogue-store bulk_rows_reduce`. Reduction outputs are zeroed
on every pipeline invocation, including CUDA graph replay, and this reset is
included in the reported pipeline time. N-tile-major outputs are converted to
logical rows only for correctness checking. Up and down execute sequentially
on the same stream; wave barriers order the down kernel's stores.

Hopper numerical regression coverage (including repeated CUDA graph replay):

```bash
pytest tests/test_gemm_gather_table.py -k down_projection2 -x
```

### In-place original-token local combine

Add `--down-combine-in-place` to a `--multi-buffer-gather --down-projection2`
run to write results back into the **original input allocations**. The mode
selects multi-buffer scatter and `bulk_rows_reduce` automatically; its mapping
comes from original buffer IDs and token indices in packed intermediate order,
including balanced gathering and round-robin scheduling. Repeated routes sum
at the original token destination. Tokens without local routes become zero.
This is an **unweighted** local combine (routing probabilities are not applied).

For example, retain your up-projection arguments and add:

```bash
--down-projection2 --down-combine-in-place \
--down-tile-m 128 --down-tile-n 128 --down-pingpong \
--down-n-tile-major-output \
--down-gather-table --down-gather-table-expert-waves \
--down-max-swizzle-size 1 --down-gather-table-wave-barrier
```

Output buffer count and token-row counts come from the inputs. Do not supply
random-destination or output-row/capacity options. The mode overrides the
ordinary down scatter-buffer count and epilogue-store selections.
N-tile-major output reinterprets the same memory: downstream consumers must
use that layout. It requires `hidden % down_tile_n == 0` to avoid padding beyond
the original allocation. Without that flag, outputs retain row-major layout.

Each invocation restores a saved copy of the original inputs, runs the up
projection, zeros the destination allocations, then runs down projection and
reduction on the same stream. **Reported pipeline timing includes restoration
and zeroing**, including on CUDA graph replay. Saved copies also preserve the
original inputs for reference checking; they require additional GPU memory.
The final contents of the input allocations are the combined outputs.

```bash
pytest tests/test_gemm_gather_table.py -k down_combine_in_place_original_tokens -x
```
