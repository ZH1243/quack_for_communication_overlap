# Hopper gather GEMM runners

This directory contains two standalone benchmark runners for QuACK's grouped
GEMM on NVIDIA Hopper GPUs (SM90). They use the same uniform Mixture-of-Experts
(MoE) routing setup but differ in where routed token rows are gathered.

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

Both runners exclude kernel compilation, warmup, CUDA graph capture, and the
correctness check from the reported kernel time.

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
```

The first run of a new configuration may pause while QuACK compiles a
specialized kernel. Each script prints the median per-kernel time, all timing
samples, effective throughput, and correctness-check result.

Use `--help` to see the command-line interface directly:

```bash
python run/hopper_gather_gemm.py --help
python run/hopper_pregather_gemm.py --help
```

## Runtime parameters

The two scripts accept the same parameters.

| Parameter | Default | Description |
| --- | ---: | --- |
| `--tokens`, `-T` | `4096` | Number of rows in the source token tensor (`T`). |
| `--hidden`, `-K` | `4096` | Input/hidden dimension (`K`). |
| `--output-dim`, `-N` | `4096` | GEMM output dimension (`N`). |
| `--experts`, `-E` | `8` | Number of experts (`E`). |
| `--routes`, `-R` | `8192` | Total routed token assignments (`R`). Must be divisible by `--experts`. |
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

Routing is uniform: every expert receives `R / E` assignments. Without
`--routing-with-replacement`, `R / E` cannot exceed `T`. A token is still
allowed to appear under different experts, as in top-k MoE routing.

For more stable timing of short kernels, CUDA graphs are enabled by default.
Use identical shapes, dtypes, kernel settings, and timing settings when
comparing the two runners.
