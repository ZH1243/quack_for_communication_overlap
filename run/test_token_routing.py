"""CPU regression checks: python -m unittest run.test_token_routing."""

import unittest

import torch

from run.token_routing import covered_token_routes


class TokenRoutingTest(unittest.TestCase):
    def test_coverage_and_numerical_combine(self):
        cases = [(4096, [1024] * 8), (17, [5, 4, 4, 4]),
                 (7, [0, 2, 7, 1]), (3, [1, 1, 1, 0, 0]), (1, [1])]
        for tokens, counts in cases:
            for replacement in (False, True):
                for seed in (0, 17, 42):
                    with self.subTest(tokens=tokens, counts=counts,
                                      replacement=replacement, seed=seed):
                        torch.manual_seed(seed)
                        indices = covered_token_routes(
                            tokens, counts, replacement=replacement, device="cpu"
                        ).long()
                        visits = torch.bincount(indices, minlength=tokens)
                        self.assertTrue(bool((visits > 0).all()))
                        self.assertEqual(indices.numel(), sum(counts))
                        self.assertTrue(bool(((indices >= 0) & (indices < tokens)).all()))
                        # Weighted gather/combine versus an explicit per-token reference.
                        x = torch.arange(1, tokens + 1, dtype=torch.float32)[:, None]
                        actual = torch.zeros_like(x)
                        reference = torch.zeros_like(x)
                        offset = 0
                        for expert, count in enumerate(counts):
                            selected = indices[offset:offset + count]
                            if not replacement:
                                self.assertEqual(selected.unique().numel(), count)
                            actual.index_add_(0, selected, x[selected] * (expert + 1))
                            for token in selected.tolist():
                                reference[token] += x[token] * (expert + 1)
                            offset += count
                        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                        self.assertTrue(bool((actual > 0).all()))

    def test_replacement_and_invalid_quotas(self):
        indices = covered_token_routes(2, [5], replacement=True, device="cpu").long()
        torch.testing.assert_close(indices.unique(), torch.arange(2), rtol=0, atol=0)
        for tokens, counts, replacement in ((4, [1, 2], False), (2, [5], False)):
            with self.assertRaises(ValueError):
                covered_token_routes(tokens, counts, replacement=replacement, device="cpu")


if __name__ == "__main__":
    unittest.main()
