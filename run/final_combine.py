"""Source-side reduction of packed results returned by bitmask routing."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch


class _CombineKernel:
    def __init__(self, counts, hidden, tile_n, num_ctas):
        self.counts = counts
        self.hidden = hidden
        self.tile_n = tile_n
        self.num_ctas = num_ctas

    @cute.jit
    def __call__(self, received: tuple, rows: cute.Tensor, output: cute.Tensor,
                 stream: cuda.CUstream):
        self.kernel(received, rows, output).launch(
            grid=(self.num_ctas, 1, 1),
            block=(256, 1, 1), stream=stream,
        )

    @cute.kernel
    def kernel(self, received: tuple, rows: cute.Tensor, output: cute.Tensor):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        blocks_per_token = cute.ceil_div(self.hidden, 256)
        for chunk in cutlass.range(block, blocks_per_token * output.shape[0], self.num_ctas):
            token = chunk // blocks_per_token
            column = chunk % blocks_per_token * 256 + thread
            if column < self.hidden:
                total = cutlass.Float32(0.0)
                for peer in cutlass.range_constexpr(len(self.counts)):
                    row = rows[peer, token]
                    if row >= 0:
                        offset = row * self.hidden + column
                        if cutlass.const_expr(self.tile_n > 0):
                            offset = ((column // self.tile_n * self.counts[peer] + row)
                                      * self.tile_n + column % self.tile_n)
                        total += received[peer][offset].to(cutlass.Float32)
                output[token, column] = total.to(output.element_type)


class FinalCombine:
    """Compile once; overwrite every output element on each invocation.

    Inputs are one packed buffer per computing GPU and its unique source-token
    indices. Empty peers are omitted. Reduction uses FP32 in ascending peer order
    with a single conversion to the received dtype; no atomics or reset required.
    The caller must finish all incoming copies before calling this object.
    num_ctas=0 launches one CTA per 256-column chunk; a positive value launches
    exactly that many CTAs, each processing chunks with a grid-stride loop.
    """

    @torch.inference_mode()
    def __init__(self, received, indices, source_tokens, hidden, tile_n=0, num_ctas=0):
        if num_ctas < 0:
            raise ValueError("num_ctas must be nonnegative")
        if tile_n < 0 or (tile_n and hidden % tile_n):
            raise ValueError("hidden must be divisible by the N tile width")
        device, dtype = received[0].device, received[0].dtype
        active = [(buffer, index) for buffer, index in zip(received, indices) if index.numel()]
        self.output = torch.empty((source_tokens, hidden), device=device, dtype=dtype)
        inverse = torch.full((len(active), source_tokens), -1, dtype=torch.int32)
        for peer, (buffer, index) in enumerate(active):
            if not buffer.is_contiguous() or buffer.numel() != index.numel() * hidden:
                raise ValueError("received buffer must be contiguous with count * hidden elements")
            inverse[peer, index.cpu().long()] = torch.arange(index.numel(), dtype=torch.int32)
        # Keep a nonempty allocation even when no tokens were sent anywhere.
        self.rows = inverse.to(device) if active else torch.full(
            (1, source_tokens), -1, dtype=torch.int32, device=device
        )
        self.buffers = tuple(buffer.view(-1) for buffer, _ in active)
        self.arguments = (tuple(from_dlpack(buffer) for buffer in self.buffers),
                          from_dlpack(self.rows), from_dlpack(self.output))
        self.num_ctas = num_ctas or source_tokens * ((hidden + 255) // 256)
        kernel = _CombineKernel(
            tuple(index.numel() for _, index in active), hidden, tile_n, self.num_ctas
        )
        self.compiled = cute.compile(
            kernel, *self.arguments, cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        )

    def __call__(self):
        self.compiled(*self.arguments,
                      cuda.CUstream(torch.cuda.current_stream(self.output.device).cuda_stream))
        return self.output
