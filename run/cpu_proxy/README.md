# Streaming gather-table CPU proxy

Build the proxy against the CUDA Toolkit installed on the Hopper host:

```bash
cmake -S run/cpu_proxy -B run/cpu_proxy/build
cmake --build run/cpu_proxy/build -j
```

`hopper_stream_gather_table_gemm.py` starts this executable once. The proxy
constructs the multi-buffer gather table directly in CUDA-pinned host memory,
imports the Python-owned HBM allocation through CUDA IPC, and then accepts a
`GO` command for every streamed kernel launch. It copies table rows on a
private nonblocking CUDA stream and publishes the ready-row prefix after each
batch with either `cudaMemcpyAsync` or `cuStreamWriteValue32` on that same
stream. `QUIT` releases the imported allocation and pinned buffers.

Small HtoD copies may use CUDA's front-end inline path instead of a copy
engine. When the proxy and persistent kernel occupy separate CUDA contexts,
that path can wait for a context scheduling boundary. Prefer running both
processes under CUDA MPS. If MPS is unavailable, the streaming runner's
`--dma-kick-bytes 65536` option puts one copy-engine-sized scratch transfer
before each flush sequence; it does not publish table rows or change readiness
granularity, but its cost is included in the end-to-end measurement.

The Python runner keeps the exported tensor storage alive and releases
PyTorch's IPC reference counter after `QUIT`. Both legacy 64-byte CUDA IPC
handles and PyTorch's versioned 66-byte `cudaMalloc` handles are supported.
Expandable-segment handles use a different import API and must be disabled for
this runner.
