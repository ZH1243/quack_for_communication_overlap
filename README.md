# 🦆 QuACK: A Quirky Assortment of CuTe Kernels 🦆

Kernels are written in the [CuTe-DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).

## Installation

``` bash
# For CUDA 12.9:
pip install quack-kernels

# For CUDA 13.x:
pip install 'quack-kernels[cu13]' --extra-index-url https://download.pytorch.org/whl/cu130

# Do not use uv for CUDA 13.x installs yet: it can race/install
# nvidia-cutlass-dsl[cu13] in the wrong order (NVIDIA/cutlass#3259):
# https://github.com/NVIDIA/cutlass/issues/3259

# Optional: install NVIDIA matmul heuristics for better untuned GEMM configs
pip install 'quack-kernels[heuristics]'

# Optional: JAX bindings (pulls in jax and jax-tvm-ffi)
pip install 'quack-kernels[jax]'
```

## Requirements

- H100, B200/B300, or RTX 50 GPU
- CUDA toolkit 12.9+
- Python 3.12

## Kernels 🐥

- 🦆 RMSNorm forward + backward
- 🦆 Softmax forward + backward
- 🦆 Cross entropy forward + backward
- 🦆 Layernorm forward + backward
- 🦆 Hopper gemm + epilogue
- 🦆 Blackwell gemm + epilogue
- 🦆 Blackwell GeForce gemm + epilogue

## Usage

```
from quack import rmsnorm, softmax, cross_entropy
```

JAX bindings are also available for some kernels (see [docs/jax.md](docs/jax.md)):

```
from quack.softmax_jax import softmax
```

## Documentations

- [JAX interface](docs/jax.md) — optional `jax` + `jax-tvm-ffi` bindings, see `quack/softmax_jax.py` for an example.

[2025-07-10] We have a comprehensive
[blogpost](media/2025-07-10-membound-sol.md) on how to get memory-bound kernels
to speed-of-light, right in the comfort of Python thanks to the [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).

## Performance

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  >
</figure>
</div>

See our [blogpost](media/2025-07-10-membound-sol.md) for the details.

## Development

To set up the development environment:

```bash
pip install -e '.[dev]'
pre-commit install

# For CUDA 13.x:
pip install 'quack-kernels[dev,cu13]' --extra-index-url https://download.pytorch.org/whl/cu130

# Do not use uv for CUDA 13.x installs yet; use pip instead.
# See https://github.com/NVIDIA/cutlass/issues/3259
```

## Conda development environment (CUDA 12.9)

Create a clean environment rather than cloning an existing environment, which may bring in
incompatible CUDA or CUTLASS packages:

```bash
conda create --name zh_quack_env312 python=3.12 pip -y
conda activate zh_quack_env312
python -m pip install --upgrade pip
```

From the repository root, install the CUDA 12.9 build of PyTorch first. Using the CUDA-specific
index prevents pip from selecting a CUDA 13 build that requires a newer driver:

```bash
python -m pip install "torch==2.13.0" \
  --index-url https://download.pytorch.org/whl/cu129
python -m pip install \
  "cuda-python==12.9.4" \
  "cuda-bindings==12.9.4"
```

Install QuACK in editable mode so changes to this checkout are used by new Python processes:

```bash
python -m pip install -e '.[dev]'
```

Validate the environment and editable installation:

```bash
python -m pip check
python -c "import quack, torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(quack.__file__)"
```

Finally, run a small numerical GPU check:

```bash
python - <<'PY'
import torch

from quack.softmax import softmax

x = torch.randn(2, 1024, device="cuda", dtype=torch.bfloat16)
y = softmax(x)
torch.testing.assert_close(y, torch.softmax(x, dim=-1), atol=1e-2, rtol=1e-2)
print("QuACK works:", y.shape, torch.cuda.get_device_name())
PY
```

## Bitmask token exchange on Hopper

`run/hopper_stream_gather_table_gemm.py` supports an opt-in, one-time variable-size
initial exchange. Add these options to a single-node `torchrun` command using
`--initial-token-exchange --multi-buffer-gather --proxy-mode thread`:

```bash
--bitmask-token-routing --token-route-probability 0.25 --local-routes-per-token 2
```

Without `--bitmask-token-routing`, the existing equal-chunk exchange is unchanged.
The new mode requires 2–8 GPUs, `--num-input-buffers` equal to the number of GPUs,
and a positive `--source-tokens` (which need not be divisible by the GPU count).
Rebuild the thread proxy with CUDA 12.9 or newer before running:

```bash
cmake -S run/cpu_proxy -B run/cpu_proxy/build
cmake --build run/cpu_proxy/build -j
```

Each rank generates a reproducible `uint8` router vector on its GPU, with one byte
per source token. On source rank `s`, bit `j` selects `(s + j + 1) % world_size`;
only the low `world_size` bits are used, including the bit selecting the local
GPU. Each bit is independently set with probability `--token-route-probability`
(default 0.25). Tokens may therefore have zero or multiple destinations.

Ranks exchange counts before allocating one input buffer per source. Selected
source rows are copied in increasing token-index order with one
`cudaMemcpyBatchAsync` call per nonempty destination. The exchange happens once,
before warmup, and is excluded from GEMM timing.

In this mode, `--tokens-per-buffer` and `--routes-per-buffer` do not determine
allocation or work sizes. A buffer receiving `T` tokens has
`T * --local-routes-per-token` local expert routes (default multiplier 2), with at
least one route for every received token. Per-expert quotas remain near-even.
Without `--routing-with-replacement`, the multiplier cannot exceed the number of
local experts. Empty pairs have zero routes; completely idle ranks skip GEMMs
but still participate in result-return completion barriers.

Both `--return-token-results` and `--return-token-results-per-group` support the
variable sizes, with their existing prerequisite flags. A source allocates its
receive buffer for computing rank `d` using the number of tokens it sent to `d`.
The return preserves packed token order and the selected output layout; it does
not scatter into or reduce across the original source-token tensor. The routing
plan retains the selected source indices (`inputs.routing_plan.indices`), also
exposed as `ReturnExchange.source_token_indices`. Per-group returns use each
peer's own byte count and tile offset. The reported GEMM throughput still excludes
the final wait for return DMA and the rank barrier, although concurrent DMA can
influence GEMM performance.

Add `--final-combine` together with `--bitmask-token-routing --return-token-results`
to reduce the returned results on each source GPU after every iteration. A CuTe
kernel reads the packed buffers (including N-tile-major layout), sums contributions
in FP32, and writes a contiguous `[source_tokens, hidden]` tensor in the received
dtype, available as `ReturnExchange.combined_output`. Tokens sent nowhere receive
zeros. This is an unweighted sum of the already locally combined expert outputs.
Add `--final-combine-ctas 16` to launch exactly 16 persistent CTAs, with 256 threads
per CTA by default. Use `--final-combine-threads-per-cta 512` to double the threads
while keeping the requested CTA count fixed. By default each thread processes
eight adjacent elements using vector loads/stores and eight FP32 accumulators. With 256 threads, each CTA processes 2048-column chunks
in a grid-stride loop. Use `--final-combine-elements-per-thread 1` for the scalar
baseline, or `4`, `16`, `32`, `64`, or `128` to vary the per-thread work. Widths above 8
use multiple 128-bit copies and one FP32 accumulator per element. CTAs span
token boundaries to keep threads active when a token needs fewer than a full
CTA of threads. Larger widths may increase register pressure, cause spills, or exceed launch
resource limits, especially with 512 threads per CTA. Benchmark 64 before 128.
The per-thread width must divide both the hidden dimension and the N tile width
(for tile-major input); otherwise the kernel uses the scalar fallback. Unsupported alignment, row/tile
widths, or dtypes fall back to one element per thread. The launch log reports the
actual vector width and threads per CTA. With 512 threads and eight elements per
thread, each CTA processes 4096 columns per iteration. The default,
`--final-combine-ctas 0`, launches
`ceil(source_tokens * hidden / (threads_per_cta * actual_elements_per_thread))` CTAs.
The CTA count limits the number of blocks, but does not guarantee one CTA per SM
or reserve particular SMs. This option requires `--final-combine`.
Add `--final-combine-column-group-major` with `--down-n-tile-major-output` to
assign work by N tile, then source token, then adjacent columns within the tile.
Each combine group contains `--down-tile-n` columns, independently of the down
GEMM's swizzle size. The default assignment remains row-major. For example, with
`--hidden 4096 --down-tile-n 128 --final-combine-elements-per-thread 32
--final-combine-threads-per-cta 512`, a CTA iteration processes 128 tokens × 128
columns instead of four complete token rows. The output remains contiguous
`[source_tokens, hidden]` by default; CTAs do not synchronize at group boundaries.

Add `--final-combine-column-group-major-output` to change the **physical result
layout**, independently of `--final-combine-column-group-major` traversal. All
three implementations (direct, register prefetch, shared pipeline) support this.
The output group width is `--down-tile-n`; `ReturnExchange.combined_output` is then
a contiguous tensor shaped `[hidden / down_tile_n, source_tokens * down_tile_n]`,
logically `[column group, source token, column within group]`. The kernel writes
this order directly, with no extra transpose launch. The flat element offset is:

```text
((column // group_width) * source_tokens + token) * group_width + column % group_width
```

For column traversal, four-peer register prefetch, and column-group-major output,
append these flags to the distributed command:

```bash
--final-combine-register-prefetch --final-combine-prefetch-peers 4 \
--final-combine-column-group-major --final-combine-column-group-major-output
```

The output flag also works with row traversal and row-major received buffers.
Consumers must use the selected physical layout. To reconstruct a logical
row-major tensor outside the timed kernel (this can allocate and copy):

```python
logical = combined_output.view(hidden // group_width, source_tokens, group_width) \
                         .permute(1, 0, 2).reshape(source_tokens, hidden)
```

The Python API selects this with `FinalCombine(..., output_tile_n=128)` (`0`
keeps row-major output). Input and output group widths may differ. Vector width
must divide the output group width as well as the input group width: the direct
kernel falls back to scalar copies if needed; prefetch kernels reject unsupported
widths. The integration reference check validates the selected physical layout.

Add `--final-combine-pipelined` to opt into an asynchronous peer-prefetch kernel,
intended for a small fixed CTA budget. It issues 128-bit `cp.async` loads into a
shared-memory ring, keeps several peers in flight, and accumulates completed
stages in FP32 in the same ascending peer order as the original kernel. Each
thread owns its shared-memory slots; no CTA barriers or atomics are needed.
Shared memory is interleaved in 16-byte lane stripes. Traversal is row-major by
default, including for N-tile-major inputs; use `--final-combine-column-group-major`
to opt into column traversal. This replaces the pipeline's earlier automatic
column traversal. The original kernel remains the default.

For the 4-CTA, 512-thread, 32-element configuration, append just:

```bash
--final-combine-pipelined
```

The default `--final-combine-pipeline-stages 3` uses 96 KiB of shared memory per
CTA for that configuration. Try `2` (64 KiB) and `4` (128 KiB) as well. The CTA
count and threads per CTA remain exactly as requested. The pipeline requires
SM80 or newer, aligned FP16/BF16 buffers, and an elements-per-thread value of at
least 8 dividing both hidden and the input N tile width. Unsupported configurations
or requests exceeding the device's shared-memory limit fail with an error;
the pipeline never silently falls back to the original kernel. More stages can
hide additional load latency but consume more shared memory. In an H20 run with
the benchmark defaults, direct row traversal took 1018 us, direct column traversal
1246 us, and the shared-memory pipeline with column traversal about 1480 us at all
three stage counts. That run showed a regression, not a pipeline speedup.

The alternative `--final-combine-register-prefetch` uses no shared memory. It
assigns adjacent eight-element stripes to adjacent lanes, then loads a batch of
peers into registers before reducing them in FP32 in ascending peer order. It
keeps the same CTA/thread counts and row-major output. Select it instead of
`--final-combine-pipelined`:

```bash
--final-combine-register-prefetch --final-combine-prefetch-peers 2
```

`--final-combine-prefetch-peers 1` isolates the striped mapping; `2` (default) and
`4` allow more independent peer loads at higher register cost. The compiler
controls instruction scheduling, so this is a prefetch opportunity rather than
an explicit asynchronous-copy pipeline. Wide vectors and deep prefetch may spill
or exceed launch resources. Like the shared pipeline, this path requires aligned
FP16/BF16 buffers and at least eight elements per thread dividing hidden and the
N tile width. The register variant still needs target-GPU validation and timing.

Run the focused numerical regression and standalone benchmark on Hopper:

```bash
python -m unittest run.test_final_combine.PipelinedFinalCombineTest
python -m unittest run.test_final_combine.RegisterFinalCombineTest
python -m unittest run.test_final_combine.FinalCombineOutputLayoutTest
python -m run.benchmark_final_combine \
  --source-tokens 4096 --hidden 4096 --tile-n 128 --peers 8 \
  --route-probability 0.5 --ctas 4 --elements-per-thread 32 --threads-per-cta 512
```

The benchmark checks every variant against a PyTorch FP32 reference, warms the
same inputs, interleaves individual launches, rotates variant order, and queues
a GPU burner before each timed round to avoid host enqueue gaps. It reports
median/minimum/p90 kernel time, output bandwidth, logical input-plus-output
bandwidth, and speedup over the original row-major kernel. It also includes the
original column-group-major kernel and both row and column traversal for every
register prefetch depth and shared pipeline depth. Register depths default to 1
and 2; add `--prefetch-peers 1 2 4` to include depth 4. `register-row-N` is the
previous `register-N` variant; `register-col-N` combines register prefetch with
column-group-major traversal. Column variants require a nonzero `--tile-n`.
Use `--output-tile-n 128` to benchmark column-group-major output for every variant;
the default `--output-tile-n 0` keeps row-major output. The `row`/`col` labels still
describe traversal, and the benchmark logs the output layout separately. For example:

```bash
python -m run.benchmark_final_combine --prefetch-peers 1 2 4 --stages 2 --ctas 4 \
  --output-tile-n 128
```

Use an otherwise idle GPU; these timings reuse resident buffers and do
not reproduce the cache state or contention of the full distributed run. Recheck
the original `torchrun` command with the new flag after the standalone check.

Add `--final-combine-periodic-release` to simulate delayed column-group availability
**after all return copies have completed**. The native CPU proxy helper publishes
one additional ready group using pinned-host-to-HBM `int32` copies every
`--flush-interval-us` (requested host submission interval). The first group is
released after one interval; zero releases all groups without intentional delay.
`--flush-entries` still controls only the up-GEMM table batches.

```bash
# Rebuild on the CUDA host before using the new flag:
cmake -S run/cpu_proxy -B run/cpu_proxy/build
cmake --build run/cpu_proxy/build -j
# Append to the existing streaming command:
# --final-combine-periodic-release
```

This mode requires `--final-combine --down-n-tile-major-output --proxy-mode thread
--flag-update-mode memcpy`. It automatically enables column-group-major traversal
without changing the selected output layout. One group is `--down-tile-n` columns,
independent of down-GEMM swizzle grouping. All three combine implementations poll
with system-scope acquire loads before reading that group's input contributions.
The counter resets to zero before each launch, and publication runs on a separate
nonblocking stream using immutable pinned values. The native helper runs on the
calling CPU thread while the enqueued combine kernel runs on the GPU; the call
returns after the flag copies finish, with GPU combine completion still ordered
on the caller's CUDA stream. This is a synthetic gate, not a return-copy readiness
protocol. Combine timing includes the artificial waits and host publication costs;
microsecond intervals are subject to OS scheduling and CUDA submission latency.

```bash
python -m unittest run.test_final_combine.FinalCombineReadinessTest
```

The combine runs after return-copy completion and finishes before receive buffers
are reused; its execution is excluded from both reported throughput timings.
CUDA events bracket the final-combine launch after return-copy completion. Each
rank reports the mean time per timing sample and the median of those means,
excluding warmup. These event intervals can include GPU idle gaps during host
enqueue and are not a profiler measurement of kernel-only duration.
Each rank also reports the final result size in bytes and effective bandwidth per
sample and at the median time: `result.numel() * result.element_size() / time`.
Bandwidth uses decimal GB/s (1 GB = 10^9 bytes) and counts only final output bytes,
without adding input reads or metadata traffic.
Without this flag, only the separate packed receive buffers are produced.

CPU metadata checks and a Hopper numerical integration regression:

```bash
python -m unittest run.test_token_routing run.test_bitmask_routing
python -m unittest run.test_final_combine  # Requires CUDA and CuTe DSL
torchrun --standalone --nnodes=1 --nproc-per-node=4 run/test_bitmask_token_exchange.py
```

The integration regression includes legacy exchange, unequal counts, sparse and
empty routing, full replication, whole-buffer returns, and one/two-tile return
groups. It validates dispatched token values, returned bytes, and GEMM outputs.
