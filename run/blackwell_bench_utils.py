"""Blackwell-only benchmark overrides and shared timing helper compatibility exports."""

import math
from contextlib import contextmanager

if __package__:
    from .gemm_bench_utils import (
        Variant as Variant,
        mapped_host_flag as mapped_host_flag,
        measure as measure,
        profile_variant as profile_variant,
        select_variants as select_variants,
    )
else:
    from gemm_bench_utils import (
        Variant as Variant,
        mapped_host_flag as mapped_host_flag,
        measure as measure,
        profile_variant as profile_variant,
        select_variants as select_variants,
    )

__all__ = [
    "Variant", "mapped_host_flag", "measure", "profile_variant", "select_variants",
    "bulk_smem_padding", "bulk_smem_cols", "bulk_smem_buffers", "bulk_store_warps",
    "kernel_config_trace",
]


@contextmanager
def bulk_smem_padding(bytes_per_row):
    """Process-local padding ablation; set before compiling any GEMM plans."""
    from quack.cache import CACHE_ENABLED
    from quack.gemm_sm100 import GemmSm100

    if CACHE_ENABLED:
        raise ValueError("--bulk-smem-padding requires QUACK_CACHE_ENABLED=0 before Python starts")
    previous = GemmSm100.bulk_store_smem_padding_bytes
    GemmSm100.bulk_store_smem_padding_bytes = bytes_per_row
    try:
        print(f"Bulk SMEM padding requested: {bytes_per_row} bytes/row; "
              "applied only if A/B stage count is preserved", flush=True)
        yield
    finally:
        GemmSm100.bulk_store_smem_padding_bytes = previous


@contextmanager
def bulk_smem_cols(max_cols):
    """Opt-in column-window experiment, selected before compiling any plans."""
    from quack.cache import CACHE_ENABLED
    from quack.gemm_sm100 import GemmSm100

    if CACHE_ENABLED:
        raise ValueError(
            "bulk SMEM window overrides require QUACK_CACHE_ENABLED=0 before Python starts"
        )
    if max_cols not in (0, 64, 128, 256):
        raise ValueError("bulk SMEM column limit must be 0 (full tile), 64, 128, or 256")
    previous = GemmSm100.bulk_store_smem_max_cols
    GemmSm100.bulk_store_smem_max_cols = max_cols
    try:
        print(f"Bulk SMEM column limit: {max_cols or 'full tile'}", flush=True)
        yield
    finally:
        GemmSm100.bulk_store_smem_max_cols = previous


@contextmanager
def bulk_smem_buffers(count):
    """Split the column budget into one or two reusable output buffers."""
    from quack.cache import CACHE_ENABLED
    from quack.gemm_sm100 import GemmSm100

    if CACHE_ENABLED:
        raise ValueError("bulk SMEM buffer overrides require QUACK_CACHE_ENABLED=0")
    if count not in (1, 2):
        raise ValueError("bulk SMEM buffer count must be 1 or 2")
    previous = GemmSm100.bulk_store_buffer_count
    GemmSm100.bulk_store_buffer_count = count
    try:
        print(f"Bulk SMEM buffers: {count} within the total column budget", flush=True)
        yield
    finally:
        GemmSm100.bulk_store_buffer_count = previous


@contextmanager
def bulk_store_warps(count):
    """Distribute the same row copies among existing SM100 epilogue warps."""
    from quack.cache import CACHE_ENABLED
    from quack.gemm_sm100 import GemmSm100

    if CACHE_ENABLED:
        raise ValueError("bulk store warp overrides require QUACK_CACHE_ENABLED=0")
    if count not in (1, 2, 4):
        raise ValueError("bulk store warp count must be 1, 2, or 4")
    previous = GemmSm100.bulk_store_num_warps
    GemmSm100.bulk_store_num_warps = count
    try:
        print(f"Bulk row-copy issuing warps: {count} (existing epilogue warps)", flush=True)
        yield
    finally:
        GemmSm100.bulk_store_num_warps = previous


@contextmanager
def kernel_config_trace():
    """Print specialization/layout metadata during compilation; no device tracing."""
    import cutlass.cute as cute
    from quack.cache import CACHE_ENABLED
    from quack.gemm_sm100 import GemmSm100
    from cutlass.cute.nvgpu import tcgen05
    from quack.sm100_utils import make_tmem_acc_mn_view

    if CACHE_ENABLED:
        raise ValueError(
            "--print-kernel-config requires QUACK_CACHE_ENABLED=0 before Python starts"
        )
    setup = GemmSm100._setup_attributes
    partition = GemmSm100.epilog_smem_store_and_partition
    tmem_partition = GemmSm100.epilog_tmem_copy_and_partition
    make_tmem_copy = tcgen05.make_tmem_copy

    def traced_setup(self, epilogue_args, varlen_args):
        result = setup(self, epilogue_args, varlen_args)
        a_bytes = cute.size_in_bytes(self.a_smem_dtype, self.a_smem_layout_staged)
        b_bytes = cute.size_in_bytes(self.b_smem_dtype, self.b_smem_layout_staged)
        d_bytes = (cute.size_in_bytes(self.d_dtype, self.epi_smem_layout_staged)
                   if self.epi_smem_layout_staged is not None else 0)
        print(f"kernel config: {type(self).__name__}, gather={self.gather_A}, "
              f"table={self.gather_table}, bulk={self.bulk_row_store}, "
              f"reduce={self.bulk_row_reduce}", flush=True)
        print(f"  MMA instruction={self.mma_inst_shape_mnk}, "
              f"MMA tile={self.mma_tiler}, CTA tile={self.cta_tile_shape_mnk}, "
              f"A/B stages={self.ab_stage}, accumulator stages={self.num_acc_stage}, "
              f"epilogue stages={self.epi_stage}, TMEM columns={self.num_tmem_alloc_cols}",
              flush=True)
        print(f"  staged layout bytes: A={a_bytes}, B={b_bytes}, D={d_bytes}; "
              "excludes other storage, metadata and alignment padding", flush=True)
        print(f"  epilogue tile={self.epi_tile}", flush=True)
        print(f"  D shared layout={self.epi_smem_layout_staged}", flush=True)
        if self.bulk_row_store:
            print(f"  bulk issuing warps={self.bulk_store_num_warps}, "
                  f"rows/issuing thread="
                  f"{math.ceil(self.cta_tile_shape_mnk[0] / (32 * self.bulk_store_num_warps))}",
                  flush=True)
            print(f"  bulk row padding={self.bulk_store_smem_padding_bytes_used} bytes, "
                  f"SMEM row stride={self.bulk_store_smem_row_stride} elements", flush=True)
            print(f"  bulk buffered columns={self.bulk_store_smem_cols}, "
                  f"buffers={self.bulk_store_buffer_count}, "
                  f"columns/window={self.bulk_store_window_cols}, "
                  f"windows/tile={self.cta_tile_shape_mnk[1] // self.bulk_store_window_cols}, "
                  f"stripes/window={self.bulk_store_stripes if self.bulk_store_chunked else 1}",
                  flush=True)
        return result

    def traced_partition(self, tiled_copy_t2r, d_layout, dtype, tTR_rD, sD, tidx):
        result = partition(self, tiled_copy_t2r, d_layout, dtype, tTR_rD, sD, tidx)
        print(f"kernel R2S (bulk={self.bulk_row_store}): {result[0]}", flush=True)
        if result[2] is not None:
            print(f"  shared partition layout={result[2].layout}", flush=True)
        return result

    def traced_tmem_partition(self, tidx, tAcc, epi_tile, use_2cta_instrs):
        # Flush before the native layout builders: SIGFPE cannot be caught as
        # a Python exception, and pytest capture would otherwise hide this data.
        print(f"kernel T2R: CTA={self.cta_tile_shape_mnk}, 2CTA={use_2cta_instrs}, "
              f"bulk={self.bulk_row_store}, windowed={self.bulk_store_chunked}", flush=True)
        print(f"  accumulator layout={tAcc.layout}", flush=True)
        print(f"  accumulator mode sizes={cute.product_each(tAcc.shape)}", flush=True)
        print(f"  epilogue tiler={epi_tile}", flush=True)
        # Show precisely the source view currently used by the kernel.
        view = make_tmem_acc_mn_view(tAcc)
        print(f"  selected accumulator view={view.layout}", flush=True)
        print(f"  selected M/N sizes={cute.product_each(view.shape)[:2]}", flush=True)
        result = tmem_partition(self, tidx, tAcc, epi_tile, use_2cta_instrs)
        print(f"  partition_S completed: {result[1].layout}", flush=True)
        return result

    def traced_make_tmem_copy(atom, tensor, *args, **kwargs):
        # Observe the actual constructor call rather than reconstructing its
        # arguments in the diagnostic (which could obscure the original bug).
        print(f"  TMEM load atom={atom}", flush=True)
        print(f"  make_tmem_copy source={tensor.layout}", flush=True)
        result = make_tmem_copy(atom, tensor, *args, **kwargs)
        print(f"  TMEM tiled copy before partition_S: {result}", flush=True)
        return result

    GemmSm100._setup_attributes = traced_setup
    GemmSm100.epilog_smem_store_and_partition = traced_partition
    GemmSm100.epilog_tmem_copy_and_partition = traced_tmem_partition
    tcgen05.make_tmem_copy = traced_make_tmem_copy
    try:
        yield
    finally:
        GemmSm100._setup_attributes = setup
        GemmSm100.epilog_smem_store_and_partition = partition
        GemmSm100.epilog_tmem_copy_and_partition = tmem_partition
        tcgen05.make_tmem_copy = make_tmem_copy
