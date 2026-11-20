"""Numerical tests for Hopper gather source selection and copy-path experiments.

Run CPU fixture checks and optional Hopper kernel checks from the repository root:
    python -m unittest discover -s tests -p test_hopper_single_source_fast_path.py -v
"""

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
    def test_explicit_copy_skips_rows_and_zero_fills_k_tails(self):
        """Observe skipped versus zero-filled shared vectors before any MMA masking."""
        import cutlass
        import cutlass.cute as cute
        from cutlass import BFloat16, Int32
        from cutlass.cute.runtime import from_dlpack
        from cutlass.utils import SmemAllocator
        from cuda.bindings import driver as cuda
        from quack.copy_utils import _cpasync_gather_row

        @cute.kernel
        def probe(source: cute.Tensor, output: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            storage = SmemAllocator().allocate_tensor(
                BFloat16, cute.make_layout((32, 8, 2), stride=(8, 1, 256)), byte_alignment=128,
            )
            for col in cutlass.range_constexpr(8):
                storage[tidx, col, 0] = BFloat16(-13)
                storage[tidx, col, 1] = BFloat16(-13)
            cute.arch.barrier()
            cute.arch.fence_view_async_shared()
            row_valid = tidx % 2 == 0
            k_valid = tidx % 4 < 2
            _cpasync_gather_row(
                storage[tidx, None, 0].iterator, source[tidx, None].iterator, row_valid,
            )
            _cpasync_gather_row(
                storage[tidx, None, 1].iterator, source[tidx, None].iterator, row_valid,
                src_bytes=Int32(k_valid) * 16,
            )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            for col in cutlass.range_constexpr(8):
                output[0, tidx, col] = storage[tidx, col, 0]
                output[1, tidx, col] = storage[tidx, col, 1]

        @cute.jit
        def launch(source: cute.Tensor, output: cute.Tensor, stream: cuda.CUstream):
            probe(source, output).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

        source = torch.arange(1, 257).reshape(32, 8).to(device="cuda", dtype=torch.bfloat16)
        output = torch.full((2, 32, 8), float("nan"), device="cuda", dtype=source.dtype)
        src_arg, dst_arg = from_dlpack(source), from_dlpack(output)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = cute.compile(launch, src_arg, dst_arg, stream)
        compiled(src_arg, dst_arg, stream)
        torch.cuda.synchronize()
        expected = torch.full_like(output, -13)
        expected[0, ::2] = source[::2]
        expected[1, ::4] = source[::4]
        expected[1, 2::4] = 0
        torch.testing.assert_close(output, expected, atol=0, rtol=0)

    @torch.inference_mode()
    def test_packed_validity_crosses_mask_word_boundary(self):
        """Pack per-thread rows, then observe copies and unused upper mask bits."""
        import cutlass
        import cutlass.cute as cute
        from cutlass import BFloat16, Boolean, Int32, Int64, Uint32
        from cutlass.cute.runtime import from_dlpack
        from cutlass.utils import SmemAllocator
        from cuda.bindings import driver as cuda
        from quack.copy_utils import _cpasync_gather_row, _pack_gather_row_validity

        @cute.kernel
        def probe(
            source: cute.Tensor, flags: cute.Tensor, output: cute.Tensor, masks: cute.Tensor,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            storage = SmemAllocator().allocate_tensor(
                BFloat16,
                cute.make_layout((32, 40, 8, 2), stride=(320, 8, 1, 10240)),
                byte_alignment=128,
            )
            row_valid = cute.make_rmem_tensor(40, Boolean)
            row_masks = cute.make_rmem_tensor(2, Uint32)
            for word in cutlass.range_constexpr(2):
                row_masks[word] = Uint32(0xFFFFFFFF)
            for row in cutlass.range_constexpr(40):
                row_valid[row] = flags[0, tidx, row] != 0
                for col in cutlass.range_constexpr(8):
                    storage[tidx, row, col, 0] = BFloat16(-13)
                    storage[tidx, row, col, 1] = BFloat16(-13)
            _pack_gather_row_validity(row_valid, row_masks)
            for word in cutlass.range_constexpr(2):
                masks[tidx, word] = Int64(row_masks[word])
            cute.arch.barrier()
            cute.arch.fence_view_async_shared()
            for row in cutlass.range_constexpr(40):
                _cpasync_gather_row(
                    storage[tidx, row, None, 0].iterator, source[tidx, row, None].iterator,
                    row_mask=row_masks[row // 32], row_bit=row % 32,
                )
                _cpasync_gather_row(
                    storage[tidx, row, None, 1].iterator, source[tidx, row, None].iterator,
                    src_bytes=Int32(flags[1, tidx, row]) * 16,
                    row_mask=row_masks[row // 32], row_bit=row % 32,
                )
                # Keep the probe independent of outstanding cp.async capacity.
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            for row in cutlass.range_constexpr(40):
                for col in cutlass.range_constexpr(8):
                    output[0, tidx, row, col] = storage[tidx, row, col, 0]
                    output[1, tidx, row, col] = storage[tidx, row, col, 1]

        @cute.jit
        def launch(
            source: cute.Tensor, flags: cute.Tensor, output: cute.Tensor, masks: cute.Tensor,
            stream: cuda.CUstream,
        ):
            probe(source, flags, output, masks).launch(
                grid=(1, 1, 1), block=(32, 1, 1), stream=stream,
            )

        source = (torch.arange(32 * 40 * 8).reshape(32, 40, 8) % 127 + 1).to(
            device="cuda", dtype=torch.bfloat16,
        )
        thread, row = torch.arange(32)[:, None], torch.arange(40)[None, :]
        valid = torch.stack(((3 * thread + row) % 5 != 0, (thread + 2 * row) % 3 != 0))
        flags = valid.to(device="cuda", dtype=torch.int32)
        output = torch.full((2, 32, 40, 8), float("nan"), device="cuda", dtype=source.dtype)
        masks = torch.full((32, 2), -1, device="cuda", dtype=torch.int64)
        args = tuple(from_dlpack(tensor) for tensor in (source, flags, output, masks))
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = cute.compile(launch, *args, stream)
        compiled(*args, stream)
        torch.cuda.synchronize()

        row_valid, k_valid = valid.to(device="cuda").unbind()
        expected = torch.full_like(output, -13)
        expected[0] = torch.where(row_valid[..., None], source, expected[0])
        tail_source = torch.where(k_valid[..., None], source, torch.zeros_like(source))
        expected[1] = torch.where(row_valid[..., None], tail_source, expected[1])
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        # Independent integer arithmetic also checks bit 31, bit 32, and that
        # the last word clears the poisoned 24 bits above the final eight rows.
        expected_masks = torch.stack([
            (valid[0, :, start:start + width].long() * (2 ** torch.arange(width))).sum(1)
            for start, width in ((0, 32), (32, 8))
        ], dim=1)
        torch.testing.assert_close(masks.cpu(), expected_masks, atol=0, rtol=0)

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
        # (single-source fast path, hoisted addresses, guarded copies,
        #  explicit copy, packed validity). Test each step with both selectors.
        variants = (
            (False, False, False, False, False), (True, False, False, False, False),
            (False, True, False, False, False), (True, True, False, False, False),
            (False, True, True, False, False), (True, True, True, False, False),
            (False, True, True, True, False), (True, True, True, True, False),
            (False, True, True, True, True), (True, True, True, True, True),
        )
        outputs = {
            flags: torch.full(
                (case.capacity, case.n), float("nan"), device="cuda", dtype=torch.bfloat16,
            )
            for flags in variants
        }

        def launch(flags):
            fast, hoist, guard, explicit, packed = flags
            output = outputs[flags]
            output.fill_(float("nan"))
            gemm(
                case.sources, case.weights, output, None, None, A_idx=case.indices,
                gather_work_table=table, multi_buffer_gather=True,
                gather_single_source_fast_path=fast,
                gather_hoist_addresses=hoist,
                gather_guarded_copies=guard,
                gather_explicit_copy=explicit,
                gather_packed_validity=packed,
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
                    output[:case.routes], outputs[variants[0]][:case.routes],
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
