"""Numerical coverage for the matched Hopper feature benchmark.

CPU metadata/reference checks and optional H100/H200 kernel checks:
    python -m unittest discover -s tests -p test_hopper_communication_bench.py -v
"""

from __future__ import annotations

import __future__
from contextlib import ExitStack
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import torch

from run import hopper_gemm_communication_bench as bench
from run import gemm_communication_bench as shared_bench


def load_cpu_tables():
    """Load the real leaf modules without initializing the CUDA/CuTe package.

    The temporary module registrations are restored before any GPU test runs.
    Postponed annotations also allow these CPU checks in an older Python env.
    """
    root = Path(__file__).resolve().parents[1]
    config = ModuleType("_communication_bench_gemm_config")
    tables = ModuleType("_communication_bench_gemm_tables")
    with patch.dict(sys.modules, {config.__name__: config}):
        path = root / "quack" / "gemm_config.py"
        exec(compile(path.read_text(), str(path), "exec", __future__.annotations.compiler_flag),
             config.__dict__)
        with patch.dict(sys.modules, {"quack.gemm_config": config}):
            path = root / "run" / "gemm_tables.py"
            exec(compile(path.read_text(), str(path), "exec"), tables.__dict__)
    return tables


class HopperBenchmarkMetadataTests(unittest.TestCase):
    def test_inline_indices_match_reordered_control_work_and_values(self):
        """Inline embedding must keep the control's exact expert/M/N traversal."""
        tables = load_cpu_tables()
        counts, tile_m, tile_n, n, k = [271, 0, 131], 128, 128, 400, 8
        source = torch.arange(31 * k).reshape(31, k).float() % 23 / 16
        tokens = ((torch.arange(sum(counts)) * 7 + 3) % 31).tolist()
        weights = torch.arange(3 * n * k).reshape(3, n, k).float() % 19 / 32
        parts, start = [], 0
        for expert, count in enumerate(counts):
            parts.append(source[tokens[start:start + count]] @ weights[expert].T)
            start += count
        reference = torch.cat(parts)
        for capacity in (9, 10):
            for cluster_m in (1, 2):
                for group_size in (1, 2):
                    with self.subTest(capacity=capacity, cluster_m=cluster_m,
                                      group_size=group_size):
                        cta_rows = tables.cta_tile_shape_m(tile_m, cluster_m, capacity)
                        rows, _ = tables.make_table(
                            [counts], device_capacity=capacity, tile_m=tile_m,
                            cluster_m=cluster_m, tile_n=tile_n, n=n, group_size=group_size,
                        )
                        ordered, inline = shared_bench.make_inline_comparison_rows(
                            rows, tokens, cluster_rows=cta_rows * cluster_m,
                        )

                        def execute(output, row, embedded=False):
                            if embedded:
                                expert, n_base, begin, end = row[:4]
                                indices = row[4:4 + end - begin]
                                self.assertEqual(row[4 + end - begin:],
                                                 [-1] * (cta_rows * cluster_m - end + begin))
                            else:
                                expert, begin, end, n_base = row
                                indices = tokens[begin:end]
                            gathered = source[indices]
                            work = []
                            for n_offset in range(group_size):
                                col = (n_base + n_offset) * tile_n
                                col_end = min(col + tile_n, n)
                                for cta in range(cluster_m):
                                    lo = min(cta * cta_rows, len(gathered))
                                    hi = min((cta + 1) * cta_rows, len(gathered))
                                    work.append((expert, begin + lo, begin + hi, col, col_end))
                                    if lo < hi:
                                        output[begin + lo:begin + hi, col:col_end] = (
                                            gathered[lo:hi] @ weights[expert, col:col_end].T
                                        )
                            return work

                        default_output = torch.full_like(reference, float("nan"))
                        default_work = []
                        for row in rows:
                            default_work.extend(execute(default_output, row))
                        torch.testing.assert_close(default_output, reference)
                        control_output = torch.full_like(reference, float("nan"))
                        inline_output = torch.full_like(reference, float("nan"))
                        control_work, inline_work = [], []
                        self.assertEqual(len(ordered), len(inline))
                        for control_row, inline_row in zip(ordered, inline):
                            control_work.extend(execute(control_output, control_row))
                            inline_work.extend(execute(inline_output, inline_row, embedded=True))
                            # Compare partial outputs, not only final coverage: an
                            # unsorted control produces the same final GEMM but
                            # changes which values have been written at each step.
                            torch.testing.assert_close(
                                control_output, inline_output, equal_nan=True,
                            )
                        self.assertEqual(control_work, inline_work)
                        self.assertNotEqual(control_work, default_work)
                        torch.testing.assert_close(control_output, reference)
                        torch.testing.assert_close(inline_output, reference)

    def test_table_cta_expansion_matches_grouped_matmul(self):
        """Check full SM90 cluster spans, source transitions, empty experts and tails."""
        make_table = load_cpu_tables().make_table
        counts = [[259, 0, 3], [0, 0, 0], [7, 0, 263]]
        sources = [
            (torch.arange(17 * 8).reshape(17, 8).float() + 3 * b) / 64
            for b in range(len(counts))
        ]
        indices = [
            (torch.arange(sum(cs), dtype=torch.int32) * 7 + 3 * b) % 17
            for b, cs in enumerate(counts)
        ]
        # Construct the independent expert-major/source-major reference.
        offsets = [[0] for _ in counts]
        for b, cs in enumerate(counts):
            for count in cs:
                offsets[b].append(offsets[b][-1] + count)
        for cluster_m, n, group_size in ((1, 272, 1), (2, 400, 2)):
            with self.subTest(cluster_m=cluster_m, n=n, group_size=group_size):
                weights = torch.arange(3 * n * 8).reshape(3, n, 8).float() % 19 / 32
                packed_parts, reference_parts = [], []
                for expert in range(3):
                    for b, source in enumerate(sources):
                        begin, end = offsets[b][expert:expert + 2]
                        gathered = source[indices[b][begin:end].long()]
                        packed_parts.append(gathered)
                        reference_parts.append(gathered @ weights[expert].T)
                packed = torch.cat(packed_parts)
                reference = torch.cat(reference_parts)
                expert_counts = [sum(cs[e] for cs in counts) for e in range(3)]
                for multi in (False, True):
                    with self.subTest(multi=multi):
                        rows, _ = make_table(
                            counts if multi else [expert_counts], device_capacity=9,
                            tile_m=128, cluster_m=cluster_m, tile_n=128,
                            n=n, group_size=group_size,
                        )
                        output = torch.full_like(reference, float("nan"))

                        def execute(row):
                            expert = row[0]
                            if multi:
                                n_base, ranges = row[1], list(zip(row[2::2], row[3::2]))
                                start = sum(lo for lo, _ in ranges)
                                a = torch.cat([
                                    source[index[lo:hi].long()]
                                    for source, index, (lo, hi) in zip(sources, indices, ranges)
                                ])
                            else:
                                _, start, end, n_base = row
                                a = packed[start:end]
                            for n_offset in range(group_size):
                                col = (n_base + n_offset) * 128
                                col_end = min(col + 128, n)
                                for cta in range(cluster_m):
                                    lo, hi = cta * 128, min((cta + 1) * 128, len(a))
                                    if lo < hi:
                                        output[start + lo:start + hi, col:col_end] = (
                                            a[lo:hi] @ weights[expert, col:col_end].T
                                        )

                        # The first full cluster must compute 128 rows per CTA.
                        # Using SM100's half-M geometry would leave NaNs here.
                        execute(rows[0])
                        span = 128 * cluster_m
                        columns = 128 * group_size
                        torch.testing.assert_close(
                            output[:span, :columns], reference[:span, :columns],
                        )
                        for row in rows[1:]:
                            execute(row)
                        torch.testing.assert_close(output, reference)

    def test_weighted_reference_and_hopper_launch_options(self):
        args = bench.parse_args([
            "--host-ready", "--weighted-contributions", "--swiglu", "--sm-count", "4",
            "--tile-k", "64",
        ])
        self.assertEqual((args.device_capacity, args.pingpong, args.tile_k, args.sm_count),
                         (9, True, 64, 4))
        self.assertTrue(args.stores and args.waves and args.host_ready)
        values = torch.tensor([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0], [14.0, 16.0]])
        mapping = torch.tensor([2, 0, 2, 0], dtype=torch.int32)
        scales = torch.tensor([0.25, 0.5, 0.75, 0.0])
        expected = torch.tensor([[3.0, 4.0], [0.0, 0.0], [8.0, 10.0], [0.0, 0.0]])
        torch.testing.assert_close(
            bench.scatter_reference(values, mapping, torch.bfloat16, scales), expected,
            rtol=0, atol=0,
        )
        generated = bench.make_route_weights(mapping)
        totals = torch.zeros(4).index_add_(0, mapping.long(), generated)
        torch.testing.assert_close(totals, torch.tensor([1.0, 0.0, 1.0, 0.0]))


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
    "requires an H100/H200 GPU",
)
class HopperBenchmarkGpuTests(unittest.TestCase):
    def run_benchmark(self, features):
        args = bench.parse_args([
            "--k", "64", "--n", "272", "--buffers", "3", "--experts", "3",
            "--routes-per-buffer", "129", "--tokens-per-buffer", "67",
            "--tile-m", "128", "--tile-n", "128", "--cluster-m", "2",
            "--sm-count", "4", "--packed-baseline", "--inline-indices", "--readiness",
            "--warmup", "1", "--rounds", "1", "--repetitions", "1", *features,
        ])
        with ExitStack() as resources:
            # The benchmark compares every output with an independent FP32
            # reference twice eagerly and again after captured graph replay.
            bench.benchmark(args, resources)

    @torch.inference_mode()
    def test_gather_and_stores_without_pingpong(self):
        self.run_benchmark(["--no-pingpong", "--stores"])

    @torch.inference_mode()
    def test_all_features_with_pingpong_and_partial_waves(self):
        self.run_benchmark(["--host-ready", "--weighted-contributions", "--swiglu"])


if __name__ == "__main__":
    unittest.main()
