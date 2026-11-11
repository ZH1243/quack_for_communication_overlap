"""Opt-in tracing for the numerical SwiGLU repro; never used by normal launches.

Enable with QUACK_DEBUG_GATED_STORE=1 QUACK_CACHE_ENABLED=0 and pytest -s.
The fixture restores every patched method after the test.
"""

import cutlass
import cutlass.cute as cute

from quack.epilogue.ops import TileStore
from quack.gemm_sm100 import GemmSm100


_acc_load = GemmSm100.epi_load_acc_subtile
_store_setup = TileStore.store_setup
_store_r2s = TileStore.store_r2s


@cute.jit
def _trace_values(label: cutlass.Constexpr[str], tensor: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    bx, by, bz = cute.arch.block_idx()
    if bx == 0 and by == 0 and bz == 0 and tid % 32 == 0:
        for i in cutlass.range_constexpr(min(4, cute.size(tensor))):
            cute.printf(label + " tid=%d i=%d value=%f", tid, i, tensor[i].to(cutlass.Float32))


@cute.jit
def _trace_coords(source: cute.Tensor, target: cute.Tensor, tidx: cutlass.Int32):
    bx, by, bz = cute.arch.block_idx()
    if bx == 0 and by == 0 and bz == 0 and tidx % 32 == 0:
        for i in cutlass.range_constexpr(min(4, cute.size(target))):
            gate, up, dst = source[2 * i], source[2 * i + 1], target[i]
            cute.printf(
                "gated coords tid=%d i=%d gate=(%d,%d) up=(%d,%d) dst=(%d,%d)",
                tidx, i, gate[0], gate[1], up[0], up[1], dst[0], dst[1],
            )


@cute.jit
def _traced_acc_load(
    self, tiled_copy_t2r, tiled_copy_r2s, tTR_tAcc, tTR_rAcc, tRS_rD,
    epi_coord, acc_pipeline, acc_consumer_state, acc_release_idx,
    clear_acc: cutlass.Boolean = False, no_release: cutlass.Constexpr[bool] = False,
):
    _acc_load(
        self, tiled_copy_t2r, tiled_copy_r2s, tTR_tAcc, tTR_rAcc, tRS_rD,
        epi_coord, acc_pipeline, acc_consumer_state, acc_release_idx,
        clear_acc=clear_acc, no_release=no_release,
    )
    print("gated trace TMEM load:", tiled_copy_t2r)
    print("gated trace accumulator registers:", tRS_rD.layout)
    if epi_coord[0] == 0 and epi_coord[1] == 0:
        _trace_values("gated accumulator", tRS_rD)


def _traced_store_setup(
    self, gemm, params, smem_tensor, tiled_copy_r2s, tiled_copy_t2r,
    tile_coord_mnkl, varlen_manager, tidx,
):
    result = _store_setup(
        self, gemm, params, smem_tensor, tiled_copy_r2s, tiled_copy_t2r,
        tile_coord_mnkl, varlen_manager, tidx,
    )
    if self.gated and gemm.arch == 100:
        aux_copy, partition, _, _ = result
        print("gated trace full-width copy:", tiled_copy_r2s)
        print("gated trace half-width copy:", aux_copy)
        print("gated trace output epi tile:", getattr(params, self._epi_tile_key()))
        print("gated trace shared tensor:", smem_tensor)
        print("gated trace shared partition:", partition)
        m, n = cute.size(smem_tensor, mode=[0]), cute.size(smem_tensor, mode=[1])
        source = tiled_copy_r2s.get_slice(tidx).partition_D(cute.make_identity_tensor((m, 2 * n)))
        target = aux_copy.get_slice(tidx).partition_D(cute.make_identity_tensor((m, n)))
        _trace_coords(source, target, tidx)
    return result


@cute.jit
def _traced_store_r2s(self, gemm, tiled_copy, frag_out, tRS_s_stage, tidx):
    if cutlass.const_expr(self.gated and gemm.arch == 100):
        retiled = tiled_copy.retile(frag_out).contiguous()
        print("gated trace converted registers:", frag_out.layout)
        print("gated trace retiled registers:", retiled.layout)
        print("gated trace store destination:", tRS_s_stage)
        _trace_values("gated converted", frag_out)
        _trace_values("gated retiled", retiled)
    _store_r2s(self, gemm, tiled_copy, frag_out, tRS_s_stage, tidx)
    if cutlass.const_expr(self.gated and gemm.arch == 100):
        # Read only this thread's destinations; no cross-warp barrier is added.
        _trace_values("gated shared readback", tRS_s_stage)


def install_gated_store_trace(monkeypatch):
    monkeypatch.setattr(GemmSm100, "epi_load_acc_subtile", _traced_acc_load)
    monkeypatch.setattr(TileStore, "store_setup", _traced_store_setup)
    monkeypatch.setattr(TileStore, "store_r2s", _traced_store_r2s)
