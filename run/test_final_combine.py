"""GPU numerical regression: python -m unittest run.test_final_combine."""

import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and CuTe DSL")
class FinalCombineTest(unittest.TestCase):
    def test_source_order_sum_and_repeated_overwrite(self):
        from run.final_combine import FinalCombine

        # Unequal peer counts, overlapping tokens, an empty peer and unrouted tokens.
        patterns = (
            ([5, 0, 2], [], [2, 5], [0, 3, 5, 1]),
            ([], [], [], []),
            ([0, 1, 2, 3, 4, 5, 6],) * 4,
        )
        for dtype in (torch.float16, torch.bfloat16):
            for tile_n in (0, 128):
                hidden = 259 if tile_n == 0 else 384
                for pattern in patterns:
                    with self.subTest(dtype=dtype, tile_n=tile_n, pattern=pattern):
                        indices = tuple(torch.tensor(p, dtype=torch.int64) for p in pattern)
                        received = []
                        for index in indices:
                            shape = ((hidden // tile_n, index.numel() * tile_n)
                                     if tile_n else (index.numel(), hidden))
                            received.append(torch.empty(shape, dtype=dtype, device="cuda"))
                        combine = FinalCombine(tuple(received), indices, 7, hidden, tile_n)
                        for iteration in range(2):
                            reference = torch.zeros((7, hidden), device="cuda", dtype=torch.float32)
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
                            torch.testing.assert_close(actual, reference.to(dtype), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
