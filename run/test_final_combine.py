"""GPU numerical regression: python -m unittest run.test_final_combine."""

import unittest
import warnings
from itertools import product

import torch


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and CuTe DSL")
class FinalCombineTest(unittest.TestCase):
    # FinalCombine allocates inference tensors; poison/reuse them in the same mode.
    @torch.inference_mode()
    def test_source_order_sum_and_repeated_overwrite(self):
        from run.final_combine import FinalCombine

        # Unequal peer counts, overlapping tokens, an empty peer and unrouted tokens.
        patterns = (
            ([5, 0, 2], [], [2, 5], [0, 3, 5, 1]),
            ([], [], [], []),
            ([0, 1, 2, 3, 4, 5, 6],) * 8,
        )
        for dtype in (torch.float16, torch.bfloat16):
            for tile_n, hidden in (
                (0, 259), (0, 4096), (128, 384), (128, 4096), (128, 4224), (6, 264)
            ):
                for pattern in patterns:
                    # Serial, two persistent CTAs, uneven stride, default, and idle CTAs.
                    for num_ctas in (1, 2, 3, 0, 129):
                        for elements_per_thread, threads_per_cta, column_group_major in product(
                            (1, 4, 8, 16, 32, 64, 128), (256, 512),
                            (False, True) if tile_n else (False,),
                        ):
                            with self.subTest(
                                dtype=dtype, tile_n=tile_n, pattern=pattern,
                                num_ctas=num_ctas, elements_per_thread=elements_per_thread,
                                threads_per_cta=threads_per_cta,
                                column_group_major=column_group_major,
                            ):
                                indices = tuple(torch.tensor(p, dtype=torch.int64) for p in pattern)
                                received = []
                                for index in indices:
                                    shape = ((hidden // tile_n, index.numel() * tile_n)
                                             if tile_n else (index.numel(), hidden))
                                    received.append(torch.empty(shape, dtype=dtype, device="cuda"))
                                combine = FinalCombine(
                                    tuple(received), indices, 7, hidden, tile_n, num_ctas=num_ctas,
                                    elements_per_thread=elements_per_thread,
                                    threads_per_cta=threads_per_cta,
                                    column_group_major=column_group_major,
                                )
                                for iteration in range(2):
                                    reference = torch.zeros(
                                        (7, hidden), device="cuda", dtype=torch.float32
                                    )
                                    for peer, (buffer, index) in enumerate(zip(received, indices)):
                                        generator = torch.Generator(device="cuda").manual_seed(
                                            100 * iteration + peer
                                        )
                                        logical = torch.randn((index.numel(), hidden), device="cuda",
                                                              dtype=dtype, generator=generator)
                                        physical = (logical.view(-1, hidden // tile_n, tile_n)
                                                    .permute(1, 0, 2).contiguous().view_as(buffer)
                                                    if tile_n else logical)
                                        buffer.copy_(physical)
                                        reference.index_add_(0, index.cuda(), logical.float())
                                    combine.output.fill_(float("nan"))
                                    actual = combine()
                                    torch.cuda.synchronize()
                                    torch.testing.assert_close(
                                        actual, reference.to(dtype), rtol=0, atol=0
                                    )

    @torch.inference_mode()
    def test_column_group_major_persistent_ctas(self):
        from run.final_combine import FinalCombine

        # Multiple chunks per group, a partial final chunk, sparse peer coverage,
        # and 32 elements per thread with 512 threads per CTA. Both assignments must agree.
        source_tokens, hidden, tile_n = 257, 4096, 128
        indices = (torch.arange(0, source_tokens, 2), torch.arange(1, source_tokens, 3))
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                received = tuple(torch.empty(
                    (hidden // tile_n, index.numel() * tile_n), dtype=dtype, device="cuda"
                ) for index in indices)
                combines = [FinalCombine(
                    received, indices, source_tokens, hidden, tile_n, num_ctas=4,
                    elements_per_thread=32, threads_per_cta=512, column_group_major=order,
                ) for order in (False, True)]
                for iteration in range(2):
                    reference = torch.zeros(
                        (source_tokens, hidden), dtype=torch.float32, device="cuda"
                    )
                    for peer, (buffer, index) in enumerate(zip(received, indices)):
                        generator = torch.Generator(device="cuda").manual_seed(
                            100 * iteration + peer
                        )
                        logical = torch.randn(
                            (index.numel(), hidden), dtype=dtype, device="cuda", generator=generator
                        )
                        buffer.copy_(logical.view(-1, hidden // tile_n, tile_n)
                                     .permute(1, 0, 2).reshape_as(buffer))
                        reference.index_add_(0, index.cuda(), logical.float())
                    for combine in combines:
                        combine.output.fill_(float("nan"))
                        actual = combine()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(actual, reference.to(dtype), rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and CuTe DSL")
class PipelinedFinalCombineTest(unittest.TestCase):
    @torch.inference_mode()
    def test_reference_and_direct(self):
        from run.final_combine import FinalCombine

        # Partial chunks, idle CTAs, ring wrap, short pipeline drains, and no peers.
        # Tuple: tokens, hidden, tile_n, vector, threads, CTAs, stages, active peers.
        cases = (
            (1, 128, 0, 8, 256, 1, 2, 1),
            (19, 384, 128, 32, 512, 4, 3, 8),
            (257, 4096, 128, 32, 512, 4, 3, 8),
            (259, 384, 0, 32, 256, 3, 2, 8),
            (17, 256, 128, 8, 256, 0, 4, 2),
            (7, 128, 128, 8, 256, 129, 3, 0),
            (131, 384, 128, 16, 256, 1, 4, 8),
            (67, 256, 128, 64, 256, 2, 2, 3),
            (37, 256, 128, 128, 256, 2, 2, 4),
        )
        for dtype, case in product((torch.float16, torch.bfloat16), cases):
            tokens, hidden, tile_n, vec, threads, ctas, stages, peers = case
            with self.subTest(dtype=dtype, case=case), warnings.catch_warnings():
                warnings.filterwarnings(
                    "error", message=r"Argument received .*cannot be converted to a JitArgument",
                    category=UserWarning,
                )
                generator = torch.Generator().manual_seed(42)
                indices = []
                for peer in range(peers):
                    # Peer zero is dense (except the last token); others have holes
                    # and different packed row orders, independent of token order.
                    dense = tokens == 131
                    index = torch.randperm(tokens if dense else max(1, tokens - 1),
                                           generator=generator)
                    indices.append(index if peer == 0 or dense else index[::1 + peer % 3])
                indices.insert(0, torch.empty(0, dtype=torch.int64))
                received = tuple(torch.empty(
                    (hidden // tile_n, index.numel() * tile_n)
                    if tile_n else (index.numel(), hidden), dtype=dtype, device="cuda",
                ) for index in indices)
                direct = FinalCombine(
                    received, indices, tokens, hidden, tile_n, num_ctas=ctas,
                    elements_per_thread=vec, threads_per_cta=threads,
                )
                pipeline = FinalCombine(
                    received, indices, tokens, hidden, tile_n, num_ctas=ctas,
                    elements_per_thread=vec, threads_per_cta=threads,
                    pipelined=True, pipeline_stages=stages,
                )
                for iteration in range(3):
                    reference = torch.zeros((tokens, hidden), device="cuda", dtype=torch.float32)
                    for peer, (buffer, index) in enumerate(zip(received, indices)):
                        generator = torch.Generator(device="cuda").manual_seed(
                            100 * iteration + peer
                        )
                        logical = torch.randn((index.numel(), hidden), dtype=dtype,
                                              device="cuda", generator=generator)
                        physical = (logical.reshape(index.numel(), hidden // tile_n, tile_n)
                                    .permute(1, 0, 2).reshape_as(buffer) if tile_n else logical)
                        buffer.copy_(physical)
                        reference.index_add_(0, index.cuda(), logical.float())
                    for combine in (direct, pipeline):
                        combine.output.fill_(float("nan"))
                        actual = combine()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(actual, reference.to(dtype), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
