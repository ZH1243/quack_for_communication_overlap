"""CPU numerical checks for benchmark data and FP32 scaling/reference ordering.

Run without CUDA/CuTe: python -m unittest discover -s tests \
    -p test_blackwell_weighted_bench.py -v
"""

import contextlib
import io
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run"))
import blackwell_gemm_communication_bench as bench


class WeightedBenchmarkTests(unittest.TestCase):
    def test_weights_normalize_complete_destination_groups_without_changing_rng(self):
        # Nonadjacent routes and multiple destination buffers must share one
        # normalization domain per destination, not per expert or source list.
        mapping = torch.tensor([4, 0, 4, 0, 1], dtype=torch.int32)
        state = torch.random.get_rng_state().clone()
        weights = bench.make_route_weights(mapping)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
        torch.testing.assert_close(weights, bench.make_route_weights(mapping), rtol=0, atol=0)
        self.assertFalse(torch.equal(weights, bench.make_route_weights(mapping, seed=5679)))
        self.assertTrue(torch.isfinite(weights).all() and (weights > 0).all())
        self.assertEqual(weights.dtype, torch.float32)
        totals = {}
        for row, weight in zip(mapping.tolist(), weights.tolist()):
            totals[row] = totals.get(row, 0.0) + weight
        for total in totals.values():
            self.assertAlmostEqual(total, 1.0, places=6)
        self.assertEqual(weights[-1].item(), 1.0)
        self.assertTrue((weights[:-1] < 1).all())

    def test_scatter_reference_distinct_zero_and_unit_weights(self):
        reference = torch.tensor([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0], [14.0, 16.0]])
        mapping = torch.tensor([2, 0, 2, 0], dtype=torch.int32)
        weights = torch.tensor([0.25, 0.5, 0.75, 0.0])
        expected = torch.tensor([[3.0, 4.0], [0.0, 0.0], [8.0, 10.0], [0.0, 0.0]])
        for dtype in (torch.bfloat16, torch.float16):
            actual = bench.scatter_reference(reference, mapping, dtype, weights)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                bench.scatter_reference(reference, mapping, dtype, torch.ones(4)),
                bench.scatter_reference(reference, mapping, dtype), rtol=0, atol=0,
            )
            torch.testing.assert_close(
                bench.scatter_reference(reference, mapping, dtype, torch.zeros(4)),
                torch.zeros_like(reference), rtol=0, atol=0,
            )

    def test_scaling_precedes_output_conversion(self):
        mapping = torch.tensor([0, 0], dtype=torch.int32)
        reference = torch.tensor([[65536.0], [8.0]])
        actual = bench.scatter_reference(reference, mapping, torch.float16,
                                         torch.tensor([0.5, 0.25]))
        torch.testing.assert_close(actual, torch.tensor([[32770.0], [0.0]]), rtol=0, atol=0)
        self.assertFalse(torch.isfinite(reference.to(torch.float16)).all())
        # Distinguishes early BF16 rounding even without overflow.
        actual = bench.scatter_reference(torch.tensor([[1.00390625]]),
                                         torch.tensor([0]), torch.bfloat16, torch.tensor([1.5]))
        torch.testing.assert_close(actual, torch.tensor([[1.5078125]]), rtol=0, atol=0)

    def test_weighted_cli_preserves_geometry_and_enables_stores(self):
        args = bench.parse_args(["--weighted-contributions", "--k", "1024", "--n", "7168",
                                 "--tile-m", "256", "--tile-n", "512", "--cluster-m", "2"])
        self.assertEqual((args.k, args.n, args.tile_m, args.tile_n, args.cluster_m),
                         (1024, 7168, 256, 512, 2))
        self.assertTrue(args.stores)
        self.assertFalse(args.waves)
        args = bench.parse_args(["--host-ready", "--weighted-contributions"])
        self.assertTrue(args.stores and args.waves and args.host_ready)
        self.assertFalse(bench.parse_args([]).weighted_contributions)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bench.parse_args(["--baseline-only", "--weighted-contributions"])


if __name__ == "__main__":
    unittest.main()
