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

Both frontends construct the multi-buffer gather table directly in
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
