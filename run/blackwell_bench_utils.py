"""Timing and mapped-host flag helpers for the Blackwell feature benchmark."""

import ctypes
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Callable

import torch
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda, runtime as cudart


@dataclass
class Variant:
    call: Callable
    check: Callable
    prepare: Callable = lambda: None
    control: str | None = None


def select_variants(variants, names):
    """Use an explicit matched subset, with its first entry as the baseline."""
    if len(set(names)) != len(names):
        raise ValueError("variant selection contains duplicates")
    unknown = [name for name in names if name not in variants]
    if unknown:
        raise ValueError(f"Unknown/disabled variants: {unknown}; available: {list(variants)}")
    selected = {name: variants[name] for name in names}
    selected[names[0]] = replace(selected[names[0]], control=None)
    for name, variant in selected.items():
        if variant.control is not None and variant.control not in selected:
            raise ValueError(f"{name!r} requires control {variant.control!r} in --variants")
    return selected


@contextmanager
def kernel_config_trace():
    """Print specialization/layout metadata during compilation; no device tracing."""
    from quack.cache import CACHE_ENABLED
    from quack.gemm_sm100 import GemmSm100

    if CACHE_ENABLED:
        raise ValueError(
            "--print-kernel-config requires QUACK_CACHE_ENABLED=0 before Python starts"
        )
    setup = GemmSm100._setup_attributes
    partition = GemmSm100.epilog_smem_store_and_partition

    def traced_setup(self, epilogue_args, varlen_args):
        result = setup(self, epilogue_args, varlen_args)
        a_bytes = cute.size_in_bytes(self.a_smem_dtype, self.a_smem_layout_staged)
        b_bytes = cute.size_in_bytes(self.b_smem_dtype, self.b_smem_layout_staged)
        d_bytes = (cute.size_in_bytes(self.d_dtype, self.epi_smem_layout_staged)
                   if self.epi_smem_layout_staged is not None else 0)
        print(f"kernel config: {type(self).__name__}, gather={self.gather_A}, "
              f"table={self.gather_table}, bulk={self.bulk_row_store}, "
              f"reduce={self.bulk_row_reduce}", flush=True)
        print(f"  MMA tile={self.mma_tiler}, CTA tile={self.cta_tile_shape_mnk}, "
              f"A/B stages={self.ab_stage}, accumulator stages={self.num_acc_stage}, "
              f"epilogue stages={self.epi_stage}, TMEM columns={self.num_tmem_alloc_cols}",
              flush=True)
        print(f"  staged layout bytes: A={a_bytes}, B={b_bytes}, D={d_bytes}; "
              "excludes other storage, metadata and alignment padding", flush=True)
        print(f"  epilogue tile={self.epi_tile}", flush=True)
        print(f"  D shared layout={self.epi_smem_layout_staged}", flush=True)
        return result

    def traced_partition(self, tiled_copy_t2r, d_layout, dtype, tTR_rD, sD, tidx):
        result = partition(self, tiled_copy_t2r, d_layout, dtype, tTR_rD, sD, tidx)
        print(f"kernel R2S (bulk={self.bulk_row_store}): {result[0]}", flush=True)
        if result[2] is not None:
            print(f"  shared partition layout={result[2].layout}", flush=True)
        return result

    GemmSm100._setup_attributes = traced_setup
    GemmSm100.epilog_smem_store_and_partition = traced_partition
    try:
        yield
    finally:
        GemmSm100._setup_attributes = setup
        GemmSm100.epilog_smem_store_and_partition = partition


def profile_variant(variant, *, name, warmup):
    """Expose one warmed eager call in the quack_profile NVTX push/pop range."""
    print(f"Checking profiling target: {name}", flush=True)
    for _ in range(2):
        variant.prepare()
        variant.call()
        torch.cuda.synchronize()
        variant.check()
    for _ in range(warmup):
        variant.prepare()
        variant.call()
    torch.cuda.synchronize()
    # Resets and metadata work must not be included in the NVTX profiling range.
    # Wave calls still include their mandatory internal GPU-counter reset.
    variant.prepare()
    torch.cuda.synchronize()
    print(f"Profiling {name} in NVTX range quack_profile", flush=True)
    torch.cuda.nvtx.range_push("quack_profile")
    try:
        variant.call()
    finally:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    variant.check()
    print("Profiling launch passed numerical checks; use graph mode for TFLOPS.", flush=True)


@cute.kernel
def _reset_host_flag_kernel(address: cute.Tensor):
    ptr = cute.make_ptr(Int32, address[0], cute.AddressSpace.gmem, assumed_align=4)
    cute.arch.store(ptr.llvm_ptr, Int32(0), sem="release", scope="sys")


@cute.jit
def _reset_host_flag(address: cute.Tensor, stream: cuda.CUstream):
    _reset_host_flag_kernel(address).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


@contextmanager
def mapped_host_flag():
    """Keep the allocation alive through graph replay, with a capturable reset."""
    err, host = cudart.cudaHostAlloc(4, cudart.cudaHostAllocMapped)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaHostAlloc failed: {err}")
    try:
        err, device_ptr = cudart.cudaHostGetDevicePointer(host, 0)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaHostGetDevicePointer failed: {err}")
        word = ctypes.c_int32.from_address(int(host))
        word.value = 0
        address = torch.tensor([int(device_ptr)], device="cuda", dtype=torch.int64)
        arg = from_dlpack(address)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        reset_fn = cute.compile(_reset_host_flag, arg, stream)

        def reset():
            reset_fn(arg, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

        # Compile and numerically verify before any graph capture.
        word.value = -1
        reset()
        torch.cuda.synchronize()
        assert word.value == 0
        yield address, word, reset
    finally:
        torch.cuda.synchronize()
        (err,) = cudart.cudaFreeHost(host)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaFreeHost failed: {err}")


def measure(variants, *, flops, rounds, repetitions, warmup):
    """Interleave individual launches, rotating order to vary L2 predecessors."""
    for name, variant in variants.items():
        print(f"Checking {name}...", flush=True)
        # Repeat to expose missing resets in reductions and completion counters.
        for _ in range(2):
            variant.prepare()
            variant.call()
            torch.cuda.synchronize()
            variant.check()
    # Load the delay kernel before capture (and before any waiting consumer).
    torch.cuda._sleep(2_000_000)
    torch.cuda.synchronize()
    names = list(variants)
    graphs = []
    for rotation in range(len(names)):
        graph = torch.cuda.CUDAGraph()
        events = []
        order = names[rotation:] + names[:rotation]
        with torch.cuda.graph(graph):
            # GPU backlog prevents event pairs from measuring host enqueue gaps.
            torch.cuda._sleep(2_000_000)
            for _ in range(repetitions):
                for name in order:
                    variant = variants[name]
                    variant.prepare()  # Destination/host-flag resets are excluded.
                    start = torch.cuda.Event(enable_timing=True, external=True)
                    end = torch.cuda.Event(enable_timing=True, external=True)
                    start.record()
                    variant.call()
                    end.record()
                    events.append((name, start, end))
        graphs.append((graph, events))
    print(f"All {len(names)} variants passed; warming and timing graphs...", flush=True)
    for i in range(warmup):
        graphs[i % len(graphs)][0].replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in names}
    round_baselines = []
    for i in range(rounds):
        graph, events = graphs[i % len(graphs)]
        graph.replay()
        torch.cuda.synchronize()
        per_round = []
        for name, start, end in events:
            us = start.elapsed_time(end) * 1000
            samples[name].append(us)
            if name == names[0]:
                per_round.append(us)
        round_baselines.append(statistics.median(per_round))
    # Validate captured replays too, including additive outputs and ready flags.
    for variant in variants.values():
        variant.check()
    medians = {name: statistics.median(times) for name, times in samples.items()}
    base = medians[names[0]]
    print(f"Overall baseline: {names[0]}")
    print("variant                    median us    TFLOPS   time/base   vs control   control")
    for name, us in medians.items():
        control = variants[name].control
        delta = f"{100 * (us / medians[control] - 1):+.2f}%" if control else "-"
        print(f"{name:27s} {us:9.2f} {flops / us / 1e6:9.1f} {us / base:11.3f} "
              f"{delta:>12s}   {control or '-'}")
    print(f"Baseline round drift (max/min): {max(round_baselines) / min(round_baselines):.3f}; "
          "repeat on an idle node if drift is large.")
