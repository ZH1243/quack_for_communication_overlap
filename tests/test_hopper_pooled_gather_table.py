"""Numerical coverage for the single-buffer Hopper benchmark with logical sources."""

from importlib import import_module
from importlib.util import find_spec
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import torch


def load_runners():
    def load():
        return (
            import_module("run.hopper_gather_table_gemm"),
            import_module("run.hopper_pooled_gather_table_gemm"),
        )

    if find_spec("cutlass") is not None:
        return load()
    # On a CPU-only install, the two unrelated eager imports need placeholders.
    # A temporary run package also keeps its imported submodules out of later tests.
    import run

    package = ModuleType("run")
    package.__path__ = run.__path__
    epilogues = ModuleType("quack.gemm_interface")
    epilogues.act_to_pytorch_fn_map = {}
    epilogues.gated_to_pytorch_fn_map = {}
    with patch.dict(sys.modules, {
        "run": package,
        "run.hopper_pregather_gemm": ModuleType("run.hopper_pregather_gemm"),
        "quack.gemm_interface": epilogues,
    }):
        return load()


class PooledGatherTests(unittest.TestCase):
    def test_same_seed_preserves_each_work_tile_and_its_gemm_values(self):
        base, runner = load_runners()
        cases = [
            # Keep the requested routing, M geometry, and eight N tiles; scale K/N for CPU.
            (2048, 4096, 16, 128, 16, 8, []),
            # Multiple N groups, mixed/partial M clusters, and reordered work.
            (11, 19, 7, 2, 11, 2, ["--balanced-multi-buffer-gather", "--round-robin-m-clusters"]),
            # Some expert/source pairs have zero routes.
            (7, 9, 16, 2, 11, 2, ["--routing-with-replacement"]),
        ]
        for tokens, routes, experts, tile_m, n, x, flags in cases:
            with self.subTest(tokens=tokens, routes=routes, flags=flags):
                args = runner.parse_args([
                    "--num-input-buffers", "8", "--tokens-per-buffer", str(tokens),
                    "--routes-per-buffer", str(routes), "--experts", str(experts),
                    "--hidden", "8", "--output-dim", str(n), "--tile-m", str(tile_m),
                    "--tile-n", "2", "--cluster-m", "2", "--max-swizzle-size", str(x), *flags,
                ])
                device = torch.device("cpu")
                torch.manual_seed(args.seed)
                native = base.prepare_inputs(args, device)
                torch.manual_seed(args.seed)
                logical, pooled = runner.prepare_pooled_inputs(args, device)
                self.assertIs(logical.up_output, pooled.up_output)
                self.assertEqual(pooled.work_group_size, native.work_group_size)
                self.assertEqual(pooled.X.shape, (8 * tokens, args.hidden))
                self.assertEqual(pooled.A_idx.shape, (8 * routes,))
                self.assertEqual(pooled.A_idx.dtype, torch.int32)
                self.assertEqual(pooled.work_table.shape, (len(native.work_table), 4))
                torch.testing.assert_close(pooled.W, native.W, atol=0, rtol=0)
                for b, source in enumerate(native.X):
                    self.assertEqual(logical.X[b].data_ptr(), pooled.X[b * tokens].data_ptr())
                    torch.testing.assert_close(logical.X[b], source, atol=0, rtol=0)
                    torch.testing.assert_close(logical.A_idx[b], native.A_idx[b])

                for original, single in zip(native.work_table.tolist(), pooled.work_table.tolist()):
                    expert, n_base, *bounds = original
                    start, end = sum(bounds[::2]), sum(bounds[1::2])
                    self.assertEqual(single, [expert, start, end, n_base])
                    chunks = [
                        (b, native.A_idx[b][lo:hi].long())
                        for b, (lo, hi) in enumerate(zip(bounds[::2], bounds[1::2])) if lo < hi
                    ]
                    ids = pooled.A_idx[start:end].long()
                    torch.testing.assert_close(ids, torch.cat([b * tokens + v for b, v in chunks]))
                    rows = torch.cat([native.X[b][v] for b, v in chunks]).float()
                    cols = slice(n_base * args.tile_n, (n_base + x) * args.tile_n)
                    reference = rows @ native.W[expert, :, cols].float()
                    actual = pooled.X[ids].float() @ pooled.W[expert, :, cols].float()
                    torch.testing.assert_close(actual, reference, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
                         "requires an H100/H200 GPU")
    @torch.inference_mode()
    def test_single_table_graph_against_native_multi_buffer(self):
        base, runner = load_runners()
        args = runner.parse_args([
            "--multi-buffer-gather", "--num-input-buffers", "8", "--tokens-per-buffer", "37",
            "--routes-per-buffer", "101", "--experts", "3", "--hidden", "80", "--output-dim", "144",
            "--tile-m", "128", "--tile-n", "128", "--tile-k", "64", "--max-swizzle-size", "1",
            "--warmup", "1", "--iterations", "2", "--timing-samples", "2",
        ])
        # Exercise the same CLI path, including warmup, graph replay, and validation
        # against the logical input slices. Retain its output for a native-kernel check.
        with patch.object(runner, "parse_args", return_value=args), patch.object(
            base, "benchmark_inputs", wraps=base.benchmark_inputs,
        ) as benchmark:
            runner.main()
        single_args, pooled = benchmark.call_args.args
        self.assertFalse(single_args.multi_buffer_gather)
        torch.manual_seed(args.seed)
        native = base.prepare_inputs(args, pooled.X.device)
        base.make_launch(args, native)()
        torch.testing.assert_close(pooled.up_output, native.up_output)


if __name__ == "__main__":
    unittest.main()
