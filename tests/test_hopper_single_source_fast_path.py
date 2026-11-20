"""Numerical tests for Hopper gather source selection, address hoisting and guards.

Run CPU fixture checks and optional Hopper kernel checks from the repository root:
    python -m unittest discover -s tests -p test_hopper_single_source_fast_path.py -v
"""

from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import torch

from run.gemm_communication_bench import single_source_cta_coverage


def make_case(tile_m, cluster_m, device="cpu", *, num_buffers=3):
    """Alternate full/tail single-source tiles with mixed and empty CTA ranges."""
    span, k, n = tile_m * cluster_m, 80, 136
    # Each expert occupies one M cluster. Nonzero source IDs deliberately
    # dominate the fast path; expert 4 crosses a source boundary in both CTAs.
    counts = [
        [0, span, 0],
        [tile_m // 2, tile_m // 2, span - tile_m],
        [0, 0, min(span, tile_m + 7)],
        [0, 7, 0],
        [tile_m - 3, min(8, span - tile_m + 3), max(0, span - tile_m - 5)],
        [0, 0, span],
    ]
    if num_buffers != 3:
        expanded = []
        for expert in counts:
            row = [0] * num_buffers
            for source, count in zip((0, num_buffers // 2, num_buffers - 1), expert):
                row[source] = count
            expanded.append(row)
        # A genuine 16-source workload: every source contributes to this
        # expert, with eight sources per CTA for the 128x2 configuration.
        expanded.append([
            span // num_buffers + (source < span % num_buffers)
            for source in range(num_buffers)
        ])
        counts = expanded
    generator = torch.Generator().manual_seed(9173)
    # Separate allocations with unequal token extents and identical row strides.
    # Correct source-pointer selection cannot be inferred from buffer 0's extent.
    sources = tuple(
        (torch.randn(tokens, k, generator=generator) + b).to(device=device, dtype=torch.bfloat16)
        for b, tokens in enumerate(17 + 12 * b + 2 * (b >= 2) for b in range(num_buffers))
    )
    offsets = [[0] for _ in sources]
    for expert in counts:
        for b, count in enumerate(expert):
            offsets[b].append(offsets[b][-1] + count)
    indices = []
    for b, source in enumerate(sources):
        valid = offsets[b][-1]
        index = torch.full((valid + 7,), -12345, dtype=torch.int32)
        index[:valid] = (torch.arange(valid) * 7 + b * 11) % len(source)
        indices.append(index.to(device))
    weights = (torch.randn(len(counts), n, k, generator=generator) / math.sqrt(k)).to(
        device=device, dtype=torch.bfloat16,
    )
    rows = [
        [expert, ng, *[bound for source in offsets for bound in source[expert:expert + 2]]]
        for expert in range(len(counts)) for ng in range(2)
    ]
    return SimpleNamespace(
        sources=sources, indices=tuple(indices), weights=weights, rows=rows,
        offsets=offsets, counts=counts, tile_m=tile_m, cluster_m=cluster_m, n=n,
        routes=sum(map(sum, counts)), capacity=sum(i.numel() for i in indices),
    )


def reference(case, *, same_dtype=False):
    """Compute directly from original source/expert assignments, without the table."""
    parts = []
    for expert, counts in enumerate(case.counts):
        for b, count in enumerate(counts):
            start = sum(previous[b] for previous in case.counts[:expert])
            gathered = case.sources[b][case.indices[b][start:start + count].long()]
            weight = case.weights[expert]
            if not same_dtype:
                gathered, weight = gathered.float(), weight.float()
            parts.append((gathered @ weight.T).float())
    return torch.cat(parts)


class SingleSourceFixtureTests(unittest.TestCase):
    def test_fixture_values_and_cta_branch_coverage(self):
        configurations = (
            (128, 2, 3, (16, 6, 2)), (64, 1, 3, (8, 4, 0)), (128, 2, 16, (16, 10, 2)),
        )
        for tile_m, cluster_m, num_buffers, expected in configurations:
            with self.subTest(tile_m=tile_m, cluster_m=cluster_m, num_buffers=num_buffers):
                case = make_case(tile_m, cluster_m, num_buffers=num_buffers)
                expected_values = reference(case)
                output = torch.full_like(expected_values, float("nan"))
                self.assertEqual(
                    single_source_cta_coverage(
                        case.rows, cta_rows=tile_m, cluster_m=cluster_m,
                    ), expected,
                )
                for expert, ng, *ranges in case.rows:
                    packed = torch.cat([
                        source[index[begin:end].long()].float()
                        for source, index, begin, end in zip(
                            case.sources, case.indices, ranges[::2], ranges[1::2],
                        )
                    ])
                    start, col = sum(ranges[::2]), ng * 128
                    col_end = min(col + 128, case.n)
                    for cta in range(cluster_m):
                        lo, hi = cta * tile_m, min((cta + 1) * tile_m, len(packed))
                        if lo < hi:
                            output[start + lo:start + hi, col:col_end] = (
                                packed[lo:hi] @ case.weights[expert, col:col_end].float().T
                            )
                torch.testing.assert_close(output, expected_values)
                self.assertEqual(len({source.data_ptr() for source in case.sources}), num_buffers)
                if num_buffers == 16:
                    self.assertEqual(case.counts[-1], [16] * 16)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
    "requires an H100/H200 GPU",
)
class SingleSourceKernelTests(unittest.TestCase):
    @torch.inference_mode()
    def check_case(self, tile_m, cluster_m, pingpong, *, num_buffers=3):
        from quack.gemm import gemm

        self.addCleanup(
            setattr, torch.backends.cuda.matmul, "allow_tf32",
            torch.backends.cuda.matmul.allow_tf32,
        )
        torch.backends.cuda.matmul.allow_tf32 = False
        case = make_case(tile_m, cluster_m, "cuda", num_buffers=num_buffers)
        table = torch.tensor(case.rows, device="cuda", dtype=torch.int32)
        # (single-source fast path, hoisted addresses, guarded copies).
        # Guarded copies require hoisting; test them with both source selectors.
        variants = (
            (False, False, False), (True, False, False),
            (False, True, False), (True, True, False),
            (False, True, True), (True, True, True),
        )
        outputs = {
            flags: torch.full(
                (case.capacity, case.n), float("nan"), device="cuda", dtype=torch.bfloat16,
            )
            for flags in variants
        }

        def launch(flags):
            fast, hoist, guard = flags
            output = outputs[flags]
            output.fill_(float("nan"))
            gemm(
                case.sources, case.weights, output, None, None, A_idx=case.indices,
                gather_work_table=table, multi_buffer_gather=True,
                gather_single_source_fast_path=fast,
                gather_hoist_addresses=hoist,
                gather_guarded_copies=guard,
                tile_M=tile_m, tile_N=128, tile_K=64,
                cluster_M=cluster_m, cluster_N=1, max_swizzle_size=1,
                persistent=True, is_dynamic_persistent=False, pingpong=pingpong,
                # One resident cluster forces every CTA to reuse its state
                # across fast, mixed, tail and empty work in the same launch.
                sm_count=cluster_m,
            )

        def check():
            expected = reference(case)
            baseline = reference(case, same_dtype=True)
            atol = max(0.035, 2 * (baseline - expected).abs().max().item())
            for output in outputs.values():
                torch.testing.assert_close(
                    output[:case.routes].float(), expected, atol=atol, rtol=0.003,
                )
                self.assertTrue(torch.isnan(output[case.routes:]).all().item())
                torch.testing.assert_close(
                    output[:case.routes], outputs[(False, False, False)][:case.routes],
                    atol=0, rtol=0,
                )

        for _ in range(2):
            for flags in variants:
                launch(flags)
            torch.cuda.synchronize()
            check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for flags in variants:
                launch(flags)
        for _ in range(2):
            graph.replay()
            torch.cuda.synchronize()
            check()
        # Reuse the same compiled plans and captured graph after changing both
        # values and source-1 routing; indices must be read on every invocation.
        case.sources[-1].add_(0.25)
        valid = case.offsets[1][-1]
        case.indices[1][:valid].add_(1).remainder_(len(case.sources[1]))
        graph.replay()
        torch.cuda.synchronize()
        check()

    def test_cluster_two_pingpong(self):
        self.check_case(128, 2, True)

    def test_cluster_two_cooperative(self):
        self.check_case(128, 2, False)

    def test_tile_64_single_cta(self):
        self.check_case(64, 1, True)

    def test_sixteen_sources_pingpong(self):
        self.check_case(128, 2, True, num_buffers=16)


if __name__ == "__main__":
    unittest.main()
