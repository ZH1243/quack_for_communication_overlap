"""Source-side reduction of packed results returned by bitmask routing."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch


class _CombineKernel:
    def __init__(self, counts, hidden, tile_n, num_ctas, elements_per_thread, threads_per_cta,
                 column_group_major):
        self.counts = counts
        self.hidden = hidden
        self.tile_n = tile_n
        self.num_ctas = num_ctas
        self.elements_per_thread = elements_per_thread
        self.threads_per_cta = threads_per_cta
        self.column_group_major = column_group_major

    @cute.jit
    def __call__(self, received: tuple, rows: cute.Tensor, output: cute.Tensor,
                 stream: cuda.CUstream):
        self.kernel(received, rows, output).launch(
            grid=(self.num_ctas, 1, 1),
            block=(self.threads_per_cta, 1, 1), stream=stream,
        )

    @cute.kernel
    def kernel(self, received: tuple, rows: cute.Tensor, output: cute.Tensor):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        vec = self.elements_per_thread
        vectors_per_token = self.hidden // vec
        total_vectors = output.shape[0] * vectors_per_token
        num_chunks = cute.ceil_div(total_vectors, self.threads_per_cta)
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), output.element_type,
            # Larger per-thread tiles repeat the 128-bit copy atom.
            num_bits_per_copy=min(128, vec * output.element_type.width),
        )
        values = cute.make_rmem_tensor((vec,), output.element_type)
        total = cute.make_rmem_tensor((vec,), cutlass.Float32)
        for chunk in cutlass.range(block, num_chunks, self.num_ctas):
            vector = chunk * self.threads_per_cta + thread
            if vector < total_vectors:
                token = vector // vectors_per_token
                column = vector % vectors_per_token * vec
                if cutlass.const_expr(self.column_group_major):
                    # Visit every source token within one N tile before the next tile.
                    vectors_per_group_row = self.tile_n // vec
                    vectors_per_group = output.shape[0] * vectors_per_group_row
                    group = vector // vectors_per_group
                    within_group = vector % vectors_per_group
                    token = within_group // vectors_per_group_row
                    column = group * self.tile_n + within_group % vectors_per_group_row * vec
                total.fill(0.0)
                for peer in cutlass.range_constexpr(len(self.counts)):
                    row = rows[peer, token]
                    if row >= 0:
                        offset = row * self.hidden + column
                        if cutlass.const_expr(self.tile_n > 0):
                            offset = ((column // self.tile_n * self.counts[peer] + row)
                                      * self.tile_n + column % self.tile_n)
                        # Host-side divisibility/alignment checks ensure a whole vector
                        # fits within a row and, for tile-major inputs, within an N tile.
                        source = cute.make_tensor(
                            received[peer].iterator + cute.assume(offset, divby=vec),
                            cute.make_layout((vec,)),
                        )
                        cute.copy(copy_atom, source, values)
                        total.store(total.load() + values.load().to(cutlass.Float32))
                values.store(total.load().to(output.element_type))
                destination = cute.make_tensor(
                    output.iterator + cute.assume(token * self.hidden + column, divby=vec),
                    cute.make_layout((vec,)),
                )
                cute.copy(copy_atom, values, destination)


class FinalCombine:
    """Compile once; overwrite every output element on each invocation.

    Inputs are one packed buffer per computing GPU and its unique source-token
    indices. Empty peers are omitted. Reduction uses FP32 in ascending peer order
    with a single conversion to the received dtype; no atomics or reset required.
    The caller must finish all incoming copies before calling this object.
    num_ctas=0 covers the flattened output with threads_per_cta * elements_per_thread
    elements per CTA; CTAs may span token boundaries.
    A positive value launches exactly that many CTAs, using a grid-stride loop.
    column_group_major=True traverses [N tile, source token, vector within tile]
    instead of [source token, vector within row]; requires tile_n > 0.
    Output stays row-major. There is no global barrier between column groups.
    Aligned FP16/BF16 inputs use vector copies and independent FP32 accumulators.
    Unsupported alignment, row widths, tile widths, or dtypes use one element per thread.
    """

    @torch.inference_mode()
    def __init__(self, received, indices, source_tokens, hidden, tile_n=0, num_ctas=0,
                 elements_per_thread=8, threads_per_cta=256, column_group_major=False):
        if num_ctas < 0:
            raise ValueError("num_ctas must be nonnegative")
        if elements_per_thread not in (1, 4, 8, 16, 32, 64, 128):
            raise ValueError("elements_per_thread must be 1, 4, 8, 16, 32, 64, or 128")
        if threads_per_cta not in (256, 512):
            raise ValueError("threads_per_cta must be 256 or 512")
        self.threads_per_cta = threads_per_cta
        if tile_n < 0 or (tile_n and hidden % tile_n):
            raise ValueError("hidden must be divisible by the N tile width")
        if column_group_major and not tile_n:
            raise ValueError("column_group_major requires N-tile-major input (tile_n > 0)")
        self.column_group_major = column_group_major
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
        vec = elements_per_thread
        if (dtype not in (torch.float16, torch.bfloat16) or hidden % vec
                or (tile_n and tile_n % vec)
                or any(buffer.data_ptr() % min(16, vec * buffer.element_size())
                       for buffer in self.buffers)):
            vec = 1
        self.elements_per_thread = vec
        alignment = min(16, vec * self.output.element_size())
        self.arguments = (
            tuple(from_dlpack(buffer, assumed_align=alignment) for buffer in self.buffers),
            from_dlpack(self.rows), from_dlpack(self.output, assumed_align=alignment),
        )
        chunk_width = threads_per_cta * vec
        self.num_ctas = num_ctas or ((source_tokens * hidden + chunk_width - 1) // chunk_width)
        kernel = _CombineKernel(
            tuple(index.numel() for _, index in active), hidden, tile_n, self.num_ctas, vec,
            threads_per_cta, column_group_major,
        )
        self.compiled = cute.compile(
            kernel, *self.arguments, cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        )

    def __call__(self):
        self.compiled(*self.arguments,
                      cuda.CUstream(torch.cuda.current_stream(self.output.device).cuda_stream))
        return self.output
