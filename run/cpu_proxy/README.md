# Streaming gather-table CPU proxy

Build the proxy executable and in-process shared library against the CUDA
Toolkit installed on the Hopper host:

```bash
cmake -S run/cpu_proxy -B run/cpu_proxy/build
cmake --build run/cpu_proxy/build -j
```

The build produces two frontends over the same implementation:

- `stream_gather_proxy` is the original executable. Python starts it once,
  and it imports the Python-owned HBM allocation through CUDA IPC.
- `libstream_gather_proxy_thread.so` is loaded with `ctypes` by
  `--proxy-mode thread`. It receives the PyTorch device pointers directly and
  performs each flush on its persistent native C++ worker thread in the Python
  process.

Both frontends construct single- or multi-buffer gather tables directly in
CUDA-pinned host memory. They copy table rows on a private nonblocking CUDA
stream and publish the ready-row prefix after each batch with either
`cudaMemcpyAsync` or `cuStreamWriteValue32` on that same stream.

Small HtoD copies may use CUDA's front-end inline path instead of a copy
engine. In process mode, that path can wait for a context scheduling boundary
because the proxy and persistent kernel occupy separate CUDA contexts. Prefer
`--proxy-mode thread`, or run process mode under CUDA MPS. If neither is
available, `--dma-kick-bytes 65536` puts one copy-engine-sized scratch transfer
before each flush sequence; it does not publish table rows or change readiness
granularity, but its cost is included in the end-to-end measurement.

In process mode, the Python runner keeps the exported tensor storage alive and
releases PyTorch's IPC reference counter after `QUIT`. Both legacy 64-byte CUDA
IPC handles and PyTorch's versioned 66-byte `cudaMalloc` handles are supported.
Expandable-segment handles use a different import API and must be disabled for
process mode. Thread mode does not export or import CUDA IPC memory.

A buffer count of one emits `[expert, route_start, route_end, cid_n_base]` rows.
Larger counts emit `[expert, cid_n_base, start_0, end_0, ...]` rows. The Python
runner selects the format from `--multi-buffer-gather` and passes the effective
buffer count to either frontend.

With `--indexed-gather`, single-buffer rows instead have `4 + tile_m * cluster_m`
columns: `[expert, cid_n_base, output_start, output_end, token_0, ...]`, with unused token slots set to `-1`.
All N groups for an M cluster are consecutive. Python appends the original
unpadded int32 routing vector after the ready flag in the shared device allocation
(`[table | ready flag | routing indices]`). Both frontends copy this vector to the
host once during initialization and construct the pinned table themselves.
This setup copy is outside the timed streaming loop. The four-column range format
and the existing thread API remain unchanged. Rebuild both artifacts to use the
new table format.


## CPU observation of down-output readiness

Append `--trace-down-ready` to the streaming runner with `--proxy-mode thread`,
`--down-projection2`, and `--down-gather-table-publish-ready` (and that flag's
expert-wave/barrier prerequisites). Rebuild the proxy library after updating.

The down kernel additionally publishes increasing column-group prefixes into a
mapped, pinned host int32 flag with system-scope release ordering. The existing
GPU readiness flag and GPU wave counter stay in HBM. After flushing the up table,
the same proxy worker acquire-polls the host flag and records timestamps using
`std::chrono::steady_clock`. It does not print or allocate on each observation.
The host flag is reset before each launch, after the previous worker and GPU have
finished. A 30-second polling timeout reports a missing final publication.

At the end, only the last measured invocation prints its observations and the gaps
between them in microseconds; warmup is excluded. For example (illustrative numbers):

```text
CPU down-ready launch 1: [0,1) at 12.300 us (first); [1,3) at 19.500 us (gap=7.200 us)
```

Ranges are column-group IDs, not column numbers. Multiply by the printed group
width to get logical columns. A prefix jump is a single CPU observation; its
intermediate GPU publication times cannot be recovered. The first timestamp is
relative to the start of polling after table flushing. Subsequent gaps measure
CPU detection latency intervals, not exact GPU execution intervals. Busy polling
uses a CPU core and may perturb timings. No copies of output data are performed.
Without `--trace-down-ready`, no host flag or polling is enabled and the existing
GPU-only publication behavior is retained. Process proxy mode does not support
this option.
