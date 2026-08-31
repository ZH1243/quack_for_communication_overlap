#!/usr/bin/env python3
"""Run Hopper multi-buffer GroupedGEMM while its gather table streams to HBM.

The C++ proxy constructs the table in pinned host memory, copies table rows in
batches, and publishes a ready-row prefix after every batch. It can run either
as a separate CUDA-IPC process or as a worker thread sharing PyTorch's CUDA
context. The persistent scheduler warp waits for the prefix before reading a
row.

Fuse a SwiGLU up-projection while streaming the 2N-wide work table:

    python run/hopper_stream_gather_table_gemm.py --multi-buffer-gather \
        --num-input-buffers 3 --output-dim 14336 --activation swiglu \
        --down-projection
"""

from __future__ import annotations

import argparse
import ctypes
import math
import selectors
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    from run.hopper_gather_table_gemm import (
        DTYPES,
        GATED_ACTIVATIONS,
        TableGatherInputs,
        balanced_buffer_allocations,
        check_correctness,
        check_down_correctness,
        make_arg_parser,
        multi_buffer_route_counts,
        sequential_buffer_allocations,
        validate_args,
    )
except ModuleNotFoundError as error:
    if error.name != "run":
        raise
    from hopper_gather_table_gemm import (  # type: ignore[no-redef]
        DTYPES,
        GATED_ACTIVATIONS,
        TableGatherInputs,
        balanced_buffer_allocations,
        check_correctness,
        check_down_correctness,
        make_arg_parser,
        multi_buffer_route_counts,
        sequential_buffer_allocations,
        validate_args,
    )


DEFAULT_PROXY = Path(__file__).resolve().parent / "cpu_proxy" / "build" / "stream_gather_proxy"
DEFAULT_THREAD_PROXY = DEFAULT_PROXY.with_name("libstream_gather_proxy_thread.so")


def parse_args() -> argparse.Namespace:
    parser = make_arg_parser(
        "Benchmark QuACK's SM90 multi-buffer GroupedGEMM while a CPU proxy streams "
        "its gather table into HBM.",
        include_down_projection=True,
    )
    parser.add_argument(
        "--flush-entries",
        type=int,
        default=1,
        help="Number of complete gather-table rows copied in each proxy batch",
    )
    parser.add_argument(
        "--flush-interval-us",
        type=int,
        default=10,
        help="Delay in microseconds between proxy batches (the first batch is immediate)",
    )
    parser.add_argument(
        "--flag-update-mode",
        choices=("memcpy", "stream-write"),
        default="memcpy",
        help="Publish ready rows with cudaMemcpyAsync or cuStreamWriteValue32",
    )
    parser.add_argument(
        "--dma-kick-bytes",
        type=int,
        default=0,
        help=(
            "Issue an HtoD copy of this many bytes before each flush sequence. This is an "
            "opt-in workaround for CUDA selecting a delayed inline path for small copies from "
            "a separate context; try 65536 when MPS is unavailable"
        ),
    )
    parser.add_argument(
        "--proxy-binary",
        type=Path,
        default=DEFAULT_PROXY,
        help="Path to the built stream_gather_proxy executable",
    )
    parser.add_argument(
        "--proxy-mode",
        choices=("process", "thread"),
        default="process",
        help="Run the proxy as a CUDA-IPC subprocess or an in-process worker thread",
    )
    parser.add_argument(
        "--thread-proxy-library",
        type=Path,
        default=DEFAULT_THREAD_PROXY,
        help="Path to the in-process proxy shared library",
    )
    return parser.parse_args()


def validate_stream_args(args: argparse.Namespace) -> None:
    validate_args(args)
    if not args.multi_buffer_gather:
        raise ValueError("hopper_stream_gather_table_gemm.py requires --multi-buffer-gather")
    if args.flush_entries <= 0:
        raise ValueError(f"flush-entries must be positive, got {args.flush_entries}")
    if args.flush_interval_us < 0:
        raise ValueError(
            f"flush-interval-us must be nonnegative, got {args.flush_interval_us}"
        )
    if args.dma_kick_bytes < 0:
        raise ValueError(f"dma-kick-bytes must be nonnegative, got {args.dma_kick_bytes}")
    proxy_path = args.proxy_binary if args.proxy_mode == "process" else args.thread_proxy_library
    if not proxy_path.is_file():
        artifact = "CPU proxy" if args.proxy_mode == "process" else "thread proxy library"
        raise FileNotFoundError(
            f"{artifact} not found at {proxy_path}. Build it with:\n"
            "  cmake -S run/cpu_proxy -B run/cpu_proxy/build\n"
            "  cmake --build run/cpu_proxy/build -j"
        )


def build_stream_metadata(
    counts_by_buffer: list[list[int]],
    *,
    output_dim: int,
    tile_m: int,
    tile_n: int,
    cluster_m: int,
    max_swizzle_size: int,
    balance_buffers: bool,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, int, int, int], ...],
    int,
    int,
]:
    """Derive allocation metadata and Q without constructing any table rows."""
    num_buffers = len(counts_by_buffer)
    experts = len(counts_by_buffer[0])
    route_offsets = []
    for counts in counts_by_buffer:
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        route_offsets.append(offsets)

    clusters_n = math.ceil(output_dim / tile_n)
    group_size = min(max_swizzle_size, clusters_n)
    assert clusters_n % group_size == 0
    num_n_groups = clusters_n // group_size
    cluster_rows = tile_m * cluster_m
    output_segments = []
    total_m_clusters = 0
    for expert in range(experts):
        expert_counts = [counts_by_buffer[j][expert] for j in range(num_buffers)]
        consumed = [0] * num_buffers
        while sum(consumed) < sum(expert_counts):
            remaining = [count - cursor for count, cursor in zip(expert_counts, consumed)]
            capacity = min(cluster_rows, sum(remaining))
            allocations = (
                balanced_buffer_allocations(remaining, capacity)
                if balance_buffers
                else sequential_buffer_allocations(remaining, capacity)
            )
            for buffer_idx, allocation in enumerate(allocations):
                start = route_offsets[buffer_idx][expert] + consumed[buffer_idx]
                end = start + allocation
                if allocation:
                    output_segments.append((expert, buffer_idx, start, end))
                consumed[buffer_idx] += allocation
            total_m_clusters += 1

    return (
        tuple(tuple(offsets) for offsets in route_offsets),
        tuple(output_segments),
        group_size,
        total_m_clusters * num_n_groups,
    )


@torch.inference_mode()
def prepare_inputs(
    args: argparse.Namespace, device: torch.device
) -> tuple[TableGatherInputs, torch.Tensor, torch.Tensor]:
    """Prepare operands and empty HBM table storage; the proxy owns table contents."""
    dtype = DTYPES[args.dtype]
    tokens = args.tokens_per_buffer or args.tokens
    routes = args.routes_per_buffer or args.routes
    counts_by_buffer = multi_buffer_route_counts(
        routes, args.experts, args.num_input_buffers
    )
    gated = args.activation in GATED_ACTIVATIONS
    gemm_output_dim = args.output_dim * (2 if gated else 1)
    route_offsets, output_segments, group_size, table_rows = build_stream_metadata(
        counts_by_buffer,
        output_dim=gemm_output_dim,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        cluster_m=args.cluster_m,
        max_swizzle_size=args.max_swizzle_size,
        balance_buffers=args.balanced_multi_buffer_gather,
    )

    if gated:
        W = torch.randn(
            (args.experts, gemm_output_dim, args.hidden), dtype=dtype, device=device
        ).transpose(1, 2)
    else:
        W = torch.randn(
            (args.experts, args.hidden, gemm_output_dim), dtype=dtype, device=device
        )
    W.mul_(1.0 / math.sqrt(args.hidden))
    W_down = None
    if args.down_projection:
        W_down = torch.randn(
            (args.experts, args.output_dim, args.hidden), dtype=dtype, device=device
        )
        W_down.mul_(1.0 / math.sqrt(args.output_dim))
    X = tuple(
        torch.randn((tokens, args.hidden), dtype=dtype, device=device)
        for _ in range(args.num_input_buffers)
    )
    idx_buffers = []
    for counts in counts_by_buffer:
        A_idx_j = torch.empty(routes, dtype=torch.int32, device=device)
        offset = 0
        for count in counts:
            if args.routing_with_replacement:
                indices = torch.randint(tokens, (count,), dtype=torch.int32, device=device)
            else:
                indices = torch.randperm(tokens, dtype=torch.int32, device=device)[:count]
            A_idx_j[offset : offset + count].copy_(indices)
            offset += count
        idx_buffers.append(A_idx_j)
    A_idx = tuple(idx_buffers)

    table_width = 2 + 2 * args.num_input_buffers
    # A single allocation avoids opening the same CUDA IPC allocation twice if
    # PyTorch suballocates the table and flag from one caching-allocator segment.
    table_elements = table_rows * table_width
    ipc_backing = torch.empty(table_elements + 1, dtype=torch.int32, device=device)
    work_table = ipc_backing[:table_elements].view(table_rows, table_width)
    ready_rows = ipc_backing[table_elements:]
    ready_rows.zero_()
    total_routes = args.num_input_buffers * routes
    up_output = torch.empty(
        (args.num_input_buffers * routes, args.output_dim), dtype=dtype, device=device
    )
    output = (
        torch.empty((total_routes, args.hidden), dtype=dtype, device=device)
        if args.down_projection
        else up_output
    )
    cu_seqlens_m = None
    if args.down_projection:
        cumulative_routes = [0]
        for expert in range(args.experts):
            count = sum(counts[expert] for counts in counts_by_buffer)
            cumulative_routes.append(cumulative_routes[-1] + count)
        assert cumulative_routes[-1] == total_routes
        cu_seqlens_m = torch.tensor(cumulative_routes, dtype=torch.int32, device=device)
    inputs = TableGatherInputs(
        X,
        W,
        A_idx,
        work_table,
        route_offsets,
        output_segments,
        output,
        group_size,
        W_down=W_down,
        up_output=up_output,
        cu_seqlens_m=cu_seqlens_m,
    )
    return inputs, ready_rows, ipc_backing


def make_launch(
    args: argparse.Namespace, inputs: TableGatherInputs, ready_rows: torch.Tensor
):
    from quack.gemm import gemm as quack_gemm
    from quack.gemm_config import GemmConfig
    from quack.gemm_interface import gemm_act

    B = inputs.W.transpose(1, 2)
    B_down = inputs.W_down.transpose(1, 2) if inputs.W_down is not None else None
    up_output = inputs.up_output if inputs.up_output is not None else inputs.output

    if args.activation is not None:
        config = GemmConfig(
            tile_m=args.tile_m,
            tile_n=args.tile_n,
            tile_k=args.tile_k,
            pingpong=args.pingpong,
            is_dynamic_persistent=False,
            cluster_m=args.cluster_m,
            cluster_n=1,
            cluster_k=1,
            max_swizzle_size=args.max_swizzle_size,
            device_capacity=9,
            use_tma_gather=False,
        )

    def launch_down() -> None:
        assert B_down is not None and inputs.cu_seqlens_m is not None
        quack_gemm(
            up_output,
            B_down,
            inputs.output,
            C=None,
            tile_count_semaphore=None,
            tile_M=args.tile_m,
            tile_N=args.tile_n,
            tile_K=args.tile_k,
            cluster_M=args.cluster_m,
            cluster_N=1,
            cluster_K=1,
            pingpong=args.pingpong,
            persistent=True,
            is_dynamic_persistent=False,
            max_swizzle_size=args.max_swizzle_size,
            cu_seqlens_m=inputs.cu_seqlens_m,
            A_idx=None,
            use_tma_gather=False,
        )

    def launch() -> None:
        if args.activation is not None:
            gemm_act(
                inputs.X,
                inputs.W,
                activation=args.activation,
                postact_out=up_output,
                A_idx=inputs.A_idx,
                gather_work_table=inputs.work_table,
                gather_work_table_ready=ready_rows,
                multi_buffer_gather=True,
                store_preact=False,
                dynamic_scheduler=False,
                tuned=False,
                config=config,
                concat_layout=("B",) if args.activation in GATED_ACTIVATIONS else None,
            )
        else:
            quack_gemm(
                inputs.X,
                B,
                up_output,
                C=None,
                tile_count_semaphore=None,
                tile_M=args.tile_m,
                tile_N=args.tile_n,
                tile_K=args.tile_k,
                cluster_M=args.cluster_m,
                cluster_N=1,
                cluster_K=1,
                pingpong=args.pingpong,
                persistent=True,
                is_dynamic_persistent=False,
                max_swizzle_size=args.max_swizzle_size,
                cu_seqlens_m=None,
                A_idx=inputs.A_idx,
                gather_work_table=inputs.work_table,
                gather_work_table_ready=ready_rows,
                multi_buffer_gather=True,
                use_tma_gather=False,
            )

        if args.down_projection:
            # This launch is ordered after the readiness-gated up GEMM on the
            # same stream and consumes its expert-contiguous activated output.
            launch_down()

    return launch, launch_down if args.down_projection else None


@dataclass
class CudaIpcExport:
    handle_hex: str
    allocation_offset_bytes: int
    storage: object
    storage_device: object
    ref_counter_handle: object
    ref_counter_offset: int
    released: bool = False

    def release(self) -> None:
        if not self.released:
            type(self.storage)._release_ipc_counter(
                self.ref_counter_handle,
                self.ref_counter_offset,
                device=self.storage_device,
            )
            self.released = True


def raw_cuda_ipc_handle(handle: bytes) -> bytes:
    """Extract cudaIpcMemHandle_t bytes from a PyTorch IPC encoding.

    Recent PyTorch versions serialize a one-byte version and a one-byte
    allocation type before the raw cudaIpcMemHandle_t. Expandable allocations
    use a different payload and cannot be opened with cudaIpcOpenMemHandle.
    """
    if len(handle) == 64:
        return handle
    if len(handle) >= 2 and handle[1] == ord("e"):
        raise RuntimeError(
            "the exported CUDA allocation uses PyTorch expandable segments, which the "
            "CPU proxy cannot import with cudaIpcOpenMemHandle; disable expandable segments "
            "for this process"
        )
    if len(handle) == 66 and handle[1] == ord("c"):
        return handle[2:]
    raise RuntimeError(
        f"unexpected CUDA IPC handle encoding ({len(handle)} bytes); expected a legacy "
        "64-byte handle or a versioned 66-byte cudaMalloc handle"
    )


def export_cuda_ipc_allocation(backing: torch.Tensor) -> CudaIpcExport:
    """Export the caching allocation and retain the matching IPC ref counter."""
    try:
        storage = backing._typed_storage()
        shared = storage._share_cuda_()
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError(
            "PyTorch could not export the table allocation through CUDA IPC. "
            "Disable expandable CUDA allocator segments and use a CUDA build that supports "
            "Storage._share_cuda_()."
        ) from error
    if len(shared) != 8:
        raise RuntimeError(f"unexpected CUDA IPC metadata tuple of length {len(shared)}")
    handle = bytes(shared[1])
    storage_offset_bytes = int(shared[3])
    tensor_offset_bytes = backing.storage_offset() * backing.element_size()
    ipc_export = CudaIpcExport(
        "",
        storage_offset_bytes + tensor_offset_bytes,
        storage,
        shared[0],
        shared[4],
        int(shared[5]),
    )
    try:
        ipc_export.handle_hex = raw_cuda_ipc_handle(handle).hex()
    except Exception:
        # _share_cuda_ increments this counter even if validation fails.
        ipc_export.release()
        raise
    return ipc_export


def proxy_command(
    args: argparse.Namespace,
    inputs: TableGatherInputs,
    ipc_handle: str,
    allocation_offset_bytes: int,
) -> list[str]:
    table_bytes = inputs.work_table.numel() * inputs.work_table.element_size()
    routes = args.routes_per_buffer or args.routes
    return [
        str(args.proxy_binary.resolve()),
        "--device",
        str(args.device),
        "--ipc-handle",
        ipc_handle,
        "--table-offset-bytes",
        str(allocation_offset_bytes),
        "--ready-offset-bytes",
        str(allocation_offset_bytes + table_bytes),
        "--table-rows",
        str(inputs.work_table.shape[0]),
        "--table-width",
        str(inputs.work_table.shape[1]),
        "--experts",
        str(args.experts),
        "--routes-per-buffer",
        str(routes),
        "--num-input-buffers",
        str(args.num_input_buffers),
        "--output-dim",
        str(inputs.W.shape[-1]),
        "--tile-m",
        str(args.tile_m),
        "--tile-n",
        str(args.tile_n),
        "--cluster-m",
        str(args.cluster_m),
        "--max-swizzle-size",
        str(args.max_swizzle_size),
        "--entries-per-flush",
        str(args.flush_entries),
        "--interval-us",
        str(args.flush_interval_us),
        "--dma-kick-bytes",
        str(args.dma_kick_bytes),
        "--balanced",
        str(int(args.balanced_multi_buffer_gather)),
        "--round-robin",
        str(int(args.round_robin_m_clusters)),
        "--flag-mode",
        args.flag_update_mode,
    ]


def read_proxy_line(process: subprocess.Popen[str], timeout_s: float = 30.0) -> str:
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        if not selector.select(timeout_s):
            raise TimeoutError(f"CPU proxy produced no status for {timeout_s:g} seconds")
        line = process.stdout.readline().strip()
    finally:
        selector.close()
    if line:
        return line
    stderr = ""
    if process.poll() is not None and process.stderr is not None:
        stderr = process.stderr.read().strip()
    raise RuntimeError(f"CPU proxy exited before sending status: {stderr or 'no error text'}")


@torch.inference_mode()
def run_stream_once(
    proxy,
    launch,
    work_table: torch.Tensor,
    ready_rows: torch.Tensor,
    device: torch.device,
) -> float:
    work_table.fill_(-1)
    ready_rows.zero_()
    torch.cuda.synchronize(device)
    # Queue the persistent kernel while every table row is still poisoned
    # and ready_rows is zero. Its scheduler warp starts by polling row 0.
    launch()
    start_ns = time.perf_counter_ns()
    proxy.go()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
    proxy.wait()
    return elapsed_ms


def stop_proxy(process: subprocess.Popen[str]) -> None:
    return_code = process.poll()
    if return_code is None:
        assert process.stdin is not None
        process.stdin.write("QUIT\n")
        process.stdin.flush()
        try:
            return_code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                return_code = process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait()
    if return_code != 0:
        assert process.stderr is not None
        raise RuntimeError(process.stderr.read().strip() or f"proxy exited {return_code}")


class ProcessProxy:
    def __init__(self, command: list[str]):
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            status = read_proxy_line(self.process)
            if not status.startswith("READY "):
                raise RuntimeError(f"unexpected CPU proxy status: {status}")
        except Exception:
            stop_proxy(self.process)
            raise

    def go(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write("GO\n")
        self.process.stdin.flush()

    def wait(self) -> None:
        status = read_proxy_line(self.process)
        if not status.startswith("DONE "):
            raise RuntimeError(f"unexpected CPU proxy completion status: {status}")

    def close(self) -> None:
        stop_proxy(self.process)


class ThreadProxy:
    """Run the C++ producer on one native worker thread in this process."""

    ERROR_CAPACITY = 4096

    def __init__(
        self,
        library_path: Path,
        args: argparse.Namespace,
        inputs: TableGatherInputs,
        ready_rows: torch.Tensor,
    ):
        self.library = ctypes.CDLL(str(library_path.resolve()))
        char_pointer = ctypes.POINTER(ctypes.c_char)
        try:
            create = self.library.stream_gather_thread_proxy_create
            start = self.library.stream_gather_thread_proxy_start
            wait = self.library.stream_gather_thread_proxy_wait
            destroy = self.library.stream_gather_thread_proxy_destroy
        except AttributeError as error:
            raise RuntimeError(
                f"{library_path} does not expose the thread-proxy API; rebuild run/cpu_proxy"
            ) from error
        create.argtypes = [
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            char_pointer,
            ctypes.c_size_t,
        ]
        create.restype = ctypes.c_void_p
        start.argtypes = [
            ctypes.c_void_p,
            char_pointer,
            ctypes.c_size_t,
        ]
        start.restype = ctypes.c_int
        wait.argtypes = [
            ctypes.c_void_p,
            char_pointer,
            ctypes.c_size_t,
        ]
        wait.restype = ctypes.c_int
        destroy.argtypes = [ctypes.c_void_p]
        destroy.restype = None

        routes = args.routes_per_buffer or args.routes
        error = ctypes.create_string_buffer(self.ERROR_CAPACITY)
        handle = create(
            args.device,
            inputs.work_table.data_ptr(),
            ready_rows.data_ptr(),
            inputs.work_table.shape[0],
            inputs.work_table.shape[1],
            args.experts,
            routes,
            args.num_input_buffers,
            inputs.W.shape[-1],
            args.tile_m,
            args.tile_n,
            args.cluster_m,
            args.max_swizzle_size,
            args.flush_entries,
            args.flush_interval_us,
            args.dma_kick_bytes,
            int(args.balanced_multi_buffer_gather),
            int(args.round_robin_m_clusters),
            int(args.flag_update_mode == "stream-write"),
            error,
            len(error),
        )
        if not handle:
            message = error.value.decode(errors="replace") or "unknown initialization error"
            raise RuntimeError(f"thread proxy initialization failed: {message}")
        self.handle = ctypes.c_void_p(handle)
        self.in_flight = False

    def go(self) -> None:
        if self.in_flight:
            raise RuntimeError("thread proxy already has a flush in flight")
        error = ctypes.create_string_buffer(self.ERROR_CAPACITY)
        result = self.library.stream_gather_thread_proxy_start(self.handle, error, len(error))
        if result != 0:
            message = error.value.decode(errors="replace") or "unknown start error"
            raise RuntimeError(f"thread proxy failed to start: {message}")
        self.in_flight = True

    def wait(self) -> None:
        if not self.in_flight:
            raise RuntimeError("thread proxy has no flush in flight")
        error = ctypes.create_string_buffer(self.ERROR_CAPACITY)
        result = self.library.stream_gather_thread_proxy_wait(self.handle, error, len(error))
        self.in_flight = False
        if result != 0:
            message = error.value.decode(errors="replace") or "unknown streaming error"
            raise RuntimeError(f"thread proxy failed: {message}")

    def close(self) -> None:
        try:
            if self.in_flight:
                self.wait()
        finally:
            if self.handle:
                self.library.stream_gather_thread_proxy_destroy(self.handle)
                self.handle = None


def main() -> None:
    args = parse_args()
    validate_stream_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 9:
        raise RuntimeError(f"this runner requires Hopper SM90, got capability {capability}")

    torch.manual_seed(args.seed)
    inputs, ready_rows, ipc_backing = prepare_inputs(args, device)
    torch.cuda.synchronize(device)
    launch, precompile_down = make_launch(args, inputs, ready_rows)

    counts = [
        [b - a for a, b in zip(offsets[:-1], offsets[1:])]
        for offsets in inputs.route_offsets
    ]
    print(f"Device: {torch.cuda.get_device_name(device)} (SM{capability[0]}{capability[1]})")
    print(
        f"X buffers: {len(inputs.X)} x {tuple(inputs.X[0].shape)}, "
        f"W_up: {tuple(inputs.W.shape)}, dtype: {inputs.X[0].dtype}"
    )
    if inputs.W_down is not None:
        assert inputs.up_output is not None
        print(
            f"W_down: {tuple(inputs.W_down.shape)}, up output: {tuple(inputs.up_output.shape)}, "
            f"final output: {tuple(inputs.output.shape)}"
        )
    else:
        print(f"Output: {tuple(inputs.output.shape)}, down projection: disabled")
    print(f"Routes per expert by buffer: {counts}")
    print(
        f"Work table: {tuple(inputs.work_table.shape)}, x={inputs.work_group_size}, "
        f"expanded work IDs={inputs.work_table.shape[0] * inputs.work_group_size}"
    )
    print(
        f"Streaming: {args.flush_entries} rows/batch, {args.flush_interval_us} us interval, "
        f"flag={args.flag_update_mode}, proxy={args.proxy_mode}, "
        f"DMA-kick={args.dma_kick_bytes} bytes; balanced-buffers="
        f"{args.balanced_multi_buffer_gather}, round-robin-m-clusters="
        f"{args.round_robin_m_clusters}"
    )
    print(f"Fused activation: {args.activation or 'disabled'}")
    if not args.no_cuda_graph:
        print("CUDA graph capture is disabled for externally produced streaming table data.")
    compile_target = "up and down kernels" if args.down_projection else "kernel"
    print(f"Compiling and warming up the readiness-gated QuACK {compile_target}...")

    # Compile the down kernel before any readiness-gated up kernel is queued.
    # Otherwise its first host-side compilation would delay proxy.go() while
    # the already-launched up kernel spins waiting for table row zero.
    if precompile_down is not None:
        assert inputs.up_output is not None
        with torch.inference_mode():
            inputs.up_output.zero_()
            precompile_down()
        torch.cuda.synchronize(device)

    proxy = None
    ipc_export = None
    try:
        if args.proxy_mode == "process":
            ipc_export = export_cuda_ipc_allocation(ipc_backing)
            command = proxy_command(
                args,
                inputs,
                ipc_export.handle_hex,
                ipc_export.allocation_offset_bytes,
            )
            proxy = ProcessProxy(command)
        else:
            proxy = ThreadProxy(args.thread_proxy_library, args, inputs, ready_rows)
        for _ in range(max(1, args.warmup)):
            run_stream_once(proxy, launch, inputs.work_table, ready_rows, device)

        timings_ms = []
        for _ in range(args.timing_samples):
            launches_ms = [
                run_stream_once(proxy, launch, inputs.work_table, ready_rows, device)
                for _ in range(args.iterations)
            ]
            timings_ms.append(statistics.mean(launches_ms))
        median_ms = statistics.median(timings_ms)
        total_routes = inputs.output.shape[0]
        up_flops = 2 * total_routes * args.hidden * inputs.W.shape[-1]
        down_flops = (
            2 * total_routes * args.output_dim * args.hidden if args.down_projection else 0
        )
        tflops = (up_flops + down_flops) / (median_ms * 1e9)
        operation_name = (
            "proxy-flush + two-GEMM MLP"
            if args.down_projection
            else "proxy-flush + up GEMM"
        )
        print(f"Per-launch end-to-end samples (ms): {[round(value, 4) for value in timings_ms]}")
        print(f"Median {operation_name} time: {median_ms:.4f} ms")
        print(f"Effective throughput: {tflops:.2f} TFLOP/s")

        if not args.skip_check:
            max_abs_error, max_allowed_error = check_correctness(
                inputs, activation=args.activation, atol=args.atol, rtol=args.rtol
            )
            if args.down_projection:
                down_error, down_allowed = check_down_correctness(
                    inputs, atol=args.atol, rtol=args.rtol
                )
                print(
                    "Reference check: PASSED "
                    f"(up max error {max_abs_error:.6g}, permitted {max_allowed_error:.6g}; "
                    f"down max error {down_error:.6g}, permitted {down_allowed:.6g})"
                )
            else:
                print(
                    "Reference check: PASSED "
                    f"(max absolute error {max_abs_error:.6g}, "
                    f"permitted bound {max_allowed_error:.6g})"
                )
    finally:
        try:
            if proxy is not None:
                proxy.close()
        finally:
            if ipc_export is not None:
                ipc_export.release()


if __name__ == "__main__":
    main()
