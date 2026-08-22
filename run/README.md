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

Runs the multi-buffer table kernel while `run/cpu_proxy/stream_gather_proxy`
streams the gather table from CUDA-pinned CPU memory into a poisoned HBM table
allocation. The proxy publishes the number of ready table rows after
every copy batch. Before reading row `i`, lane 0 of the scheduler warp waits for
`ready_rows[0] >= i + 1`; it then broadcasts the row through the existing
scheduler path. All GEMM, gather, and epilogue behavior is otherwise unchanged.

Build the C++ proxy first:

```bash
cmake -S run/cpu_proxy -B run/cpu_proxy/build
cmake --build run/cpu_proxy/build -j
```

Then run, for example:

```bash
python run/hopper_stream_gather_table_gemm.py --multi-buffer-gather \
    --num-input-buffers 3 --tokens-per-buffer 4096 \
    --routes-per-buffer 8195 --hidden 4096 --output-dim 4096 --experts 8 \
    --flush-entries 2 --flush-interval-us 10 --flag-update-mode memcpy
```

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

Run the table-scheduled fused gather benchmark:

```bash
python run/hopper_gather_table_gemm.py
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
table runner accepts the applicable subset (it is static-persistent and does
not expose `--pingpong`) and permits `--routes` values not divisible by the
expert count.

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
