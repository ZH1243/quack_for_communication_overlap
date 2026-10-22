"""GPU numerical regression: python -m unittest run.test_final_combine."""

import ctypes
from pathlib import Path
import time
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
    register_prefetch = False

    @torch.inference_mode()
    def test_reference_and_direct(self):
        from run.final_combine import FinalCombine

        # Partial chunks, idle CTAs, ring wrap, short pipeline drains, and no peers.
        # Tuple: tokens, hidden, tile_n, vector, threads, CTAs, stages, active peers.
        cases = (
            (1, 128, 0, 8, 256, 1, 2, 1),
            (19, 384, 128, 32, 512, 4, 3, 8),
            (257, 4096, 128, 32, 512, 4, 3, 8),
            (257, 4096, 128, 32, 512, 4, 2, 8),
            (19, 4096, 128, 32, 256, 4, 4, 8),
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
                    pipelined=not self.register_prefetch, pipeline_stages=stages,
                    register_prefetch=self.register_prefetch,
                    prefetch_peers=stages if stages != 3 else 1,
                    column_group_major=bool(tile_n) and tokens == 19,
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


class RegisterFinalCombineTest(PipelinedFinalCombineTest):
    """Run the same numerical cases against the register-prefetch implementation."""

    register_prefetch = True


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and CuTe DSL")
class FinalCombineOutputLayoutTest(unittest.TestCase):
    @torch.inference_mode()
    def test_all_kernels_and_traversals(self):
        from run.final_combine import FinalCombine

        # tokens, hidden, input tile, output tile, vector, threads, CTAs, traversal, empty
        # Cover register stripes crossing output groups, different input/output
        # widths, row-major input, tails, idle CTAs, and entirely absent peers.
        cases = (
            (257, 4096, 128, 128, 32, 512, 4, False, False),
            (37, 512, 128, 64, 32, 256, 3, True, False),
            (7, 512, 0, 256, 32, 256, 129, False, False),
            (17, 512, 128, 128, 8, 256, 0, True, True),
        )
        for dtype, case, implementation in product(
            (torch.float16, torch.bfloat16), cases, ("direct", "register", "pipelined"),
        ):
            tokens, hidden, tile_n, output_tile, vec, threads, ctas, traversal, empty = case
            with self.subTest(dtype=dtype, case=case, implementation=implementation):
                # Token tokens-1 is unrouted; packed rows are in reverse token order.
                indices = tuple(torch.arange(peer % 3, tokens - 1, 1 + peer % 3).flip(0)
                                if not empty and peer != 1 else torch.empty(0, dtype=torch.int64)
                                for peer in range(8))
                received = tuple(torch.empty(index.numel() * hidden, dtype=dtype, device="cuda")
                                 for index in indices)
                combines = [FinalCombine(
                    received, indices, tokens, hidden, tile_n, num_ctas=ctas,
                    elements_per_thread=vec, threads_per_cta=threads,
                    column_group_major=traversal, pipelined=implementation == "pipelined",
                    register_prefetch=implementation == "register", prefetch_peers=4,
                    output_tile_n=width,
                ) for width in (0, output_tile)]
                for iteration in range(2):
                    reference = torch.zeros((tokens, hidden), device="cuda", dtype=torch.float32)
                    for peer, (buffer, index) in enumerate(zip(received, indices)):
                        generator = torch.Generator(device="cuda").manual_seed(
                            100 * iteration + peer
                        )
                        logical = torch.randn((index.numel(), hidden), dtype=dtype,
                                              device="cuda", generator=generator)
                        physical = (logical.reshape(index.numel(), hidden // tile_n, tile_n)
                                    .permute(1, 0, 2).reshape(-1) if tile_n else logical.reshape(-1))
                        buffer.copy_(physical)
                        reference.index_add_(0, index.cuda(), logical.float())
                    expected = reference.to(dtype)
                    for combine in combines:
                        combine.output.fill_(float("nan"))
                        actual = combine()
                        torch.cuda.synchronize()
                        physical_expected = (expected.reshape(tokens, hidden // output_tile,
                                                              output_tile)
                                             .permute(1, 0, 2).contiguous().reshape_as(actual)
                                             if combine.output_tile_n else expected)
                        torch.testing.assert_close(actual, physical_expected, rtol=0, atol=0)

    @torch.inference_mode()
    def test_scalar_output_layout(self):
        from run.final_combine import FinalCombine

        # Six-column groups force scalar fallback from the requested vector width.
        indices = (torch.tensor([4, 0, 2]), torch.tensor([0, 3]))
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                received = tuple(torch.randn((i.numel(), 264), device="cuda", dtype=dtype)
                                 for i in indices)
                combine = FinalCombine(received, indices, 7, 264, num_ctas=3,
                                       elements_per_thread=8, output_tile_n=6)
                reference = torch.zeros((7, 264), device="cuda", dtype=torch.float32)
                for buffer, index in zip(received, indices):
                    reference.index_add_(0, index.cuda(), buffer.float())
                combine.output.fill_(float("nan"))
                actual = combine()
                expected = (reference.to(dtype).reshape(7, 44, 6)
                            .permute(1, 0, 2).contiguous().reshape(44, 42))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and CuTe DSL")
class FinalCombineReadinessTest(unittest.TestCase):
    @torch.inference_mode()
    def test_delayed_groups_guard_input_loads(self):
        from run.final_combine import FinalCombine

        # Partial chunks and group boundaries inside a CTA; sparse, empty peers.
        tokens, hidden, tile_n = 65, 384, 128
        indices = (torch.arange(0, tokens, 2), torch.arange(1, tokens, 3), torch.empty(0).long())
        for implementation in ({}, {"register_prefetch": True}, {"pipelined": True}):
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(implementation=implementation, dtype=dtype):
                    received = tuple(torch.zeros((3, index.numel() * tile_n),
                                                 device="cuda", dtype=dtype) for index in indices)
                    ready = torch.zeros(1, device="cuda", dtype=torch.int32)
                    combine = FinalCombine(
                        received, indices, tokens, hidden, tile_n, num_ctas=4,
                        elements_per_thread=32, threads_per_cta=512, ready_groups=ready,
                        **implementation,
                    )
                    publisher = torch.cuda.Stream()
                    prefixes = torch.arange(1, 4, dtype=torch.int32, pin_memory=True)
                    for iteration in range(2):
                        ready.zero_()
                        combine.output.fill_(float("nan"))
                        for buffer in received:
                            buffer.zero_()
                        # Inputs are deliberately withheld until each flag update:
                        # removing a readiness wait must produce the wrong sum.
                        reference = torch.zeros((tokens, hidden), dtype=torch.float32)
                        host_buffers = []
                        for peer, index in enumerate(indices):
                            generator = torch.Generator().manual_seed(iteration * 10 + peer)
                            logical = torch.randn((index.numel(), hidden), generator=generator,
                                                  dtype=dtype)
                            reference.index_add_(0, index, logical.float())
                            physical = (logical.reshape(index.numel(), 3, tile_n).permute(1, 0, 2)
                                        .contiguous().reshape(3, index.numel() * tile_n).pin_memory())
                            host_buffers.append(physical)
                        torch.cuda.synchronize()
                        actual = combine()
                        try:
                            for group in range(3):
                                time.sleep(0.02)
                                with torch.cuda.stream(publisher):
                                    for buffer, host in zip(received, host_buffers):
                                        if buffer.numel():
                                            buffer[group].copy_(host[group], non_blocking=True)
                                    ready.copy_(prefixes[group:group + 1], non_blocking=True)
                        finally:
                            # Unblock the GPU even if a host-side assertion/copy raises.
                            with torch.cuda.stream(publisher):
                                ready.copy_(prefixes[-1:], non_blocking=True)
                            publisher.synchronize()
                            torch.cuda.synchronize()
                        torch.testing.assert_close(actual.cpu(), reference.to(dtype), rtol=0, atol=0)

    @torch.inference_mode()
    def test_native_periodic_release_repeated_runs(self):
        from run.final_combine import FinalCombine

        path = Path(__file__).parent / "cpu_proxy/build/libstream_gather_proxy_thread.so"
        if not path.exists():
            self.skipTest("build run/cpu_proxy to test native periodic publication")
        library = ctypes.CDLL(str(path.resolve()))
        indices = (torch.tensor([0, 2, 4]), torch.tensor([1, 2]))
        for implementation in ({}, {"register_prefetch": True}, {"pipelined": True}):
            with self.subTest(implementation=implementation):
                received = tuple(torch.empty((3, index.numel() * 128), device="cuda",
                                             dtype=torch.bfloat16) for index in indices)
                combine = FinalCombine(
                    received, indices, 5, 384, 128, num_ctas=4, elements_per_thread=32,
                    threads_per_cta=512, release_interval_us=1000, release_library=library,
                    **implementation,
                )
                for iteration in range(2):
                    reference = torch.zeros((5, 384), dtype=torch.float32, device="cuda")
                    for peer, (buffer, index) in enumerate(zip(received, indices)):
                        generator = torch.Generator(device="cuda").manual_seed(iteration * 10 + peer)
                        logical = torch.randn((index.numel(), 384), device="cuda",
                                              dtype=torch.bfloat16, generator=generator)
                        buffer.copy_(logical.reshape(-1, 3, 128).permute(1, 0, 2).reshape_as(buffer))
                        reference.index_add_(0, index.cuda(), logical.float())
                    combine.output.fill_(float("nan"))
                    actual = combine()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(actual, reference.bfloat16(), rtol=0, atol=0)
                    self.assertEqual(combine.ready_groups.item(), 3)


if __name__ == "__main__":
    unittest.main()
