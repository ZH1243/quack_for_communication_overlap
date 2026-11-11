"""Timing and mapped-host flag helpers for the Blackwell feature benchmark."""

import ctypes
import statistics
from contextlib import contextmanager
from dataclasses import dataclass
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
