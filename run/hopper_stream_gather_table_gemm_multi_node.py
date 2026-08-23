#!/usr/bin/env python3
"""Run QuACK Grouped_GEMM_Stream_Gather with the in-process multi-node Proxy_C.

Launch one process per GPU with ``torchrun``. Router metadata and same-node token
payloads are exchanged by Python; Proxy_C receives borrowed HBM pointers and owns only
RDMA/NVLink progress, completion processing, and gather-table publication.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


CODE_ROOT = Path(__file__).resolve().parents[2]
SMEP_ROOT = CODE_ROOT / "SM_Free_EP"
DEFAULT_LIBRARY = SMEP_ROOT / "build" / "libsm_free_ep_proxy.so"
if str(SMEP_ROOT) not in sys.path:
    sys.path.insert(0, str(SMEP_ROOT))

from kernels.router_metadata import load_extension, process_router_output  # noqa: E402


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


@dataclass(frozen=True)
class DeviceBinding:
    local_gpu_index: int
    cuda_device_id: int
    rdma_device_name: str


def parse_device_map(value: str) -> dict[int, DeviceBinding]:
    """Parse ``local_gpu_index:cuda_device_id:rdma_device_name`` entries."""
    if not value.strip():
        return {}
    bindings: dict[int, DeviceBinding] = {}
    for raw_entry in value.split(","):
        entry = raw_entry.strip()
        fields = entry.split(":")
        if len(fields) != 3:
            raise ValueError(
                "each --device-map entry must be "
                "local_gpu_index:cuda_device_id:rdma_device_name"
            )
        try:
            local_gpu_index = int(fields[0])
            cuda_device_id = int(fields[1])
        except ValueError as error:
            raise ValueError(f"invalid integer in --device-map entry: {entry}") from error
        rdma_device_name = fields[2].strip()
        if local_gpu_index < 0 or cuda_device_id < 0 or not rdma_device_name:
            raise ValueError(f"invalid --device-map entry: {entry}")
        if local_gpu_index in bindings:
            raise ValueError(f"duplicate local GPU index in --device-map: {local_gpu_index}")
        bindings[local_gpu_index] = DeviceBinding(
            local_gpu_index, cuda_device_id, rdma_device_name
        )
    cuda_ids = [binding.cuda_device_id for binding in bindings.values()]
    if len(cuda_ids) != len(set(cuda_ids)):
        raise ValueError("--device-map cannot assign one CUDA device to multiple local ranks")
    return bindings


def validate_device_map_topology(
    bindings: dict[int, DeviceBinding], gpus_per_node: int, visible_cuda_devices: int
) -> None:
    if not bindings:
        return
    if set(bindings) != set(range(gpus_per_node)):
        missing = sorted(set(range(gpus_per_node)) - set(bindings))
        extra = sorted(set(bindings) - set(range(gpus_per_node)))
        raise ValueError(
            "--device-map must contain exactly one entry for every local GPU index; "
            f"missing={missing}, extra={extra}"
        )
    invalid_cuda_ids = sorted(
        binding.cuda_device_id
        for binding in bindings.values()
        if binding.cuda_device_id >= visible_cuda_devices
    )
    if invalid_cuda_ids:
        raise ValueError(
            "--device-map references CUDA devices outside the PyTorch-visible range: "
            f"{invalid_cuda_ids}; visible device count={visible_cuda_devices}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", "-T", type=int, default=4096)
    parser.add_argument("--hidden", "-K", type=int, default=4096)
    parser.add_argument("--output-dim", "-N", type=int, default=4096)
    parser.add_argument("--experts", "-E", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument(
        "--peer-hosts",
        default=os.environ.get("SMEP_PEER_HOSTS", ""),
        help="Comma-separated node hosts in node-rank order (one entry per node)",
    )
    parser.add_argument("--rdma-device", default=os.environ.get("SMEP_RDMA_DEVICE", ""))
    parser.add_argument(
        "--device-map",
        default=os.environ.get("SMEP_DEVICE_MAP", ""),
        help=(
            "Per-local-rank CUDA/RDMA bindings as comma-separated "
            "local_gpu_index:cuda_device_id:rdma_device_name entries"
        ),
    )
    parser.add_argument("--rdma-port", type=int, default=1)
    parser.add_argument("--gid-index", type=int, default=-1)
    parser.add_argument("--control-port-base", type=int, default=18515)
    parser.add_argument("--num-qps-per-peer", type=int, default=8)
    parser.add_argument("--tokens-per-chunk", type=int, default=32)
    parser.add_argument("--forwarding-batch-tokens", type=int, default=128)
    parser.add_argument("--completion-poll-batch", type=int, default=16)
    parser.add_argument("--max-in-flight-chunks", type=int, default=4)
    parser.add_argument("--send-queue-depth", type=int, default=128)
    parser.add_argument("--recv-queue-depth", type=int, default=128)
    parser.add_argument("--cq-depth", type=int, default=256)
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--tile-m", type=int, default=256)
    parser.add_argument("--tile-n", type=int, default=256)
    parser.add_argument("--tile-k", type=int, default=None)
    parser.add_argument("--cluster-m", type=int, default=2)
    parser.add_argument("--max-swizzle-size", type=int, default=8)
    parser.add_argument("--flag-update-mode", choices=("memcpy", "stream-write"), default="memcpy")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--proxy-library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--exchange-dir", default="/tmp/sm_free_ep")
    parser.add_argument("--run-id", default=os.environ.get("SMEP_RUN_ID", ""))
    parser.add_argument(
        "--proxy-mode",
        choices=("thread",),
        default="thread",
        help="Proxy_C is intentionally in-process so it shares PyTorch's CUDA context",
    )
    parser.add_argument("--mock-mode", action="store_true", help="CPU-only native plumbing mode")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, world_size: int) -> None:
    positive = (
        "tokens",
        "hidden",
        "output_dim",
        "experts",
        "top_k",
        "gpus_per_node",
        "num_qps_per_peer",
        "tokens_per_chunk",
        "forwarding_batch_tokens",
        "tile_m",
        "tile_n",
        "cluster_m",
        "max_swizzle_size",
        "iterations",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if world_size < 2 or world_size % args.gpus_per_node:
        raise ValueError("WORLD_SIZE must be a multiple of --gpus-per-node and at least 2")
    if args.experts % world_size:
        raise ValueError("--experts must divide evenly across all GPUs")
    if args.top_k > args.experts:
        raise ValueError("--top-k cannot exceed --experts")
    if not 2 <= args.gpus_per_node <= 8:
        raise ValueError("--gpus-per-node must be in [2, 8]")
    if args.forwarding_batch_tokens % args.tokens_per_chunk:
        raise ValueError("--forwarding-batch-tokens must be a multiple of --tokens-per-chunk")
    clusters_n = math.ceil(args.output_dim / args.tile_n)
    group_size = min(args.max_swizzle_size, clusters_n)
    if clusters_n % group_size:
        raise ValueError("ceil(output_dim / tile_n) must be divisible by the work group size")
    if args.warmup < 0:
        raise ValueError("--warmup must be nonnegative")
    if not args.mock_mode and not args.proxy_library.is_file():
        raise FileNotFoundError(
            f"Proxy_C library not found at {args.proxy_library}. Build it with:\n"
            f"  cmake -S {SMEP_ROOT} -B {SMEP_ROOT / 'build'}\n"
            f"  cmake --build {SMEP_ROOT / 'build'} -j"
        )


def distributed_run_id(requested: str, rank: int) -> str:
    value = requested if rank == 0 else ""
    if rank == 0 and not value:
        value = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    objects = [value]
    dist.broadcast_object_list(objects, src=0)
    return str(objects[0])


def cpu_packet(rank: int, metadata) -> dict[str, Any]:
    return {
        "rank": rank,
        "node_offsets": metadata.node_offsets.cpu().tolist(),
        "node_indices": metadata.node_token_indices.cpu().tolist(),
        "node_masks": metadata.node_token_masks.cpu().tolist(),
        "expert_offsets": metadata.expert_offsets.cpu().tolist(),
        "expert_indices": metadata.expert_token_indices.cpu().tolist(),
    }


def destination_expert_metadata(
    packets: list[dict[str, Any]], destination_rank: int, experts_per_gpu: int
) -> tuple[list[list[int]], list[list[int]]]:
    first_expert = destination_rank * experts_per_gpu
    all_indices: list[list[int]] = []
    all_offsets: list[list[int]] = []
    for packet in packets:
        offsets = packet["expert_offsets"]
        begin = offsets[first_expert]
        end = offsets[first_expert + experts_per_gpu]
        all_indices.append(packet["expert_indices"][begin:end])
        all_offsets.append(
            [offsets[first_expert + expert] - begin for expert in range(experts_per_gpu + 1)]
        )
    return all_indices, all_offsets


def local_node_groups(num_nodes: int, gpus_per_node: int):
    groups = []
    for node in range(num_nodes):
        ranks = list(range(node * gpus_per_node, (node + 1) * gpus_per_node))
        groups.append(dist.new_group(ranks=ranks, backend="nccl"))
    return groups


@torch.inference_mode()
def prime_same_node_buffers(
    *,
    X: torch.Tensor,
    X_buffers: list[torch.Tensor],
    packets: list[dict[str, Any]],
    node_rank: int,
    local_gpu_index: int,
    gpus_per_node: int,
    local_group,
    expected_tokens: list[int],
) -> None:
    gathered = [torch.empty_like(X) for _ in range(gpus_per_node)]
    dist.all_gather(gathered, X, group=local_group)
    for source_gpu, source_x in enumerate(gathered):
        source_rank = node_rank * gpus_per_node + source_gpu
        packet = packets[source_rank]
        node_begin = packet["node_offsets"][node_rank]
        node_end = packet["node_offsets"][node_rank + 1]
        indices = torch.tensor(
            packet["node_indices"][node_begin:node_end], dtype=torch.int64, device=X.device
        )
        masks = torch.tensor(
            packet["node_masks"][node_begin:node_end], dtype=torch.uint8, device=X.device
        )
        bit = (source_gpu - local_gpu_index) % gpus_per_node
        selected = indices[(masks & (1 << bit)) != 0]
        packed = source_x.index_select(0, selected)
        if packed.shape[0] != expected_tokens[source_rank]:
            raise RuntimeError(
                f"same-node compact count mismatch for source {source_rank}: "
                f"payload={packed.shape[0]} metadata={expected_tokens[source_rank]}"
            )
        X_buffers[source_rank][: packed.shape[0]].copy_(packed)


def build_output_segments(
    expert_offsets: list[list[int]], cluster_rows: int
) -> tuple[tuple[int, int, int, int], ...]:
    """Reference ordering matching Proxy_C's sequential per-buffer allocation."""
    world = len(expert_offsets)
    experts = len(expert_offsets[0]) - 1
    segments: list[tuple[int, int, int, int]] = []
    for expert in range(experts):
        consumed = [0] * world
        counts = [offsets[expert + 1] - offsets[expert] for offsets in expert_offsets]
        while sum(consumed) < sum(counts):
            capacity = min(cluster_rows, sum(counts) - sum(consumed))
            for source in range(world):
                take = min(counts[source] - consumed[source], capacity)
                if take:
                    start = expert_offsets[source][expert] + consumed[source]
                    segments.append((expert, source, start, start + take))
                    consumed[source] += take
                    capacity -= take
                if capacity == 0:
                    break
    return tuple(segments)


@dataclass
class PreparedInputs:
    X: torch.Tensor
    W: torch.Tensor
    R: torch.Tensor
    X_buffers: tuple[torch.Tensor, ...]
    A_idx: tuple[torch.Tensor, ...]
    a_idx_counts: list[int]
    expert_offsets: list[list[int]]
    work_table: torch.Tensor
    ready_rows: torch.Tensor
    output: torch.Tensor
    table_rows: int
    table_width: int
    group_size: int
    output_segments: tuple[tuple[int, int, int, int], ...]
    rdma_recv: tuple[torch.Tensor, ...]
    outgoing_offsets: list[int]
    outgoing_indices: list[int]
    incoming_offsets: list[int]
    incoming_masks: list[int]
    host_a_idx_offsets: list[int]
    host_a_idx: list[int]


@dataclass
class CudaIpcExport:
    handle: bytes
    offset_bytes: int
    storage: object
    storage_device: object
    ref_counter_handle: object
    ref_counter_offset: int
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        type(self.storage)._release_ipc_counter(
            self.ref_counter_handle,
            self.ref_counter_offset,
            device=self.storage_device,
        )
        self.released = True


def export_cuda_ipc_tensor(tensor: torch.Tensor) -> CudaIpcExport:
    """Export PyTorch's allocation base plus this tensor's byte offset."""
    try:
        storage = tensor._typed_storage()
        shared = storage._share_cuda_()
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError(
            "PyTorch could not export a CUDA IPC allocation. Disable expandable CUDA "
            "allocator segments and use a build that supports Storage._share_cuda_()."
        ) from error
    if len(shared) != 8:
        raise RuntimeError(f"unexpected CUDA IPC metadata tuple of length {len(shared)}")
    encoded = bytes(shared[1])
    if len(encoded) == 64:
        handle = encoded
    elif len(encoded) == 66 and encoded[1] == ord("c"):
        handle = encoded[2:]
    elif len(encoded) >= 2 and encoded[1] == ord("e"):
        type(storage)._release_ipc_counter(shared[4], int(shared[5]), device=shared[0])
        raise RuntimeError(
            "expandable CUDA allocator segments cannot be imported by Proxy_C; "
            "disable expandable_segments"
        )
    else:
        type(storage)._release_ipc_counter(shared[4], int(shared[5]), device=shared[0])
        raise RuntimeError(f"unsupported CUDA IPC handle encoding ({len(encoded)} bytes)")
    return CudaIpcExport(
        handle=handle,
        offset_bytes=int(shared[3]) + tensor.storage_offset() * tensor.element_size(),
        storage=storage,
        storage_device=shared[0],
        ref_counter_handle=shared[4],
        ref_counter_offset=int(shared[5]),
    )


@torch.inference_mode()
def prepare_inputs(
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    node_rank: int,
    local_gpu_index: int,
    device: torch.device,
    local_group,
) -> PreparedInputs:
    dtype = DTYPES[args.dtype]
    experts_per_gpu = args.experts // world_size
    torch.manual_seed(args.seed + rank)
    X = torch.randn((args.tokens, args.hidden), dtype=dtype, device=device)
    W = torch.randn((experts_per_gpu, args.hidden, args.output_dim), dtype=dtype, device=device)
    W.mul_(1.0 / math.sqrt(args.hidden))
    R = torch.empty((args.tokens, args.top_k), dtype=torch.int32, device=device)
    load_extension().generate_r_(R, args.experts, args.seed + rank)
    metadata = process_router_output(
        R,
        num_experts=args.experts,
        experts_per_gpu=experts_per_gpu,
        gpus_per_node=args.gpus_per_node,
        local_gpu_index=local_gpu_index,
    )
    torch.cuda.synchronize(device)
    packet = cpu_packet(rank, metadata)
    packets: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(packets, packet)
    packets = [item for item in packets if item is not None]
    packets.sort(key=lambda item: item["rank"])
    if [item["rank"] for item in packets] != list(range(world_size)):
        raise RuntimeError("router metadata all-gather returned duplicate or missing ranks")

    source_indices, source_offsets = destination_expert_metadata(
        packets, rank, experts_per_gpu
    )
    meaningful_counts = [len(indices) for indices in source_indices]
    index_capacity = max(1, *meaningful_counts)
    A_idx = []
    for indices in source_indices:
        tensor = torch.zeros(index_capacity, dtype=torch.int32, device=device)
        if indices:
            tensor[: len(indices)].copy_(torch.tensor(indices, dtype=torch.int32, device=device))
        A_idx.append(tensor)

    expected_tokens = [max(indices, default=-1) + 1 for indices in source_indices]
    # Every buffer is a distinct allocation: Proxy_C exports each one with cudaIpcGetMemHandle.
    X_buffers = [
        torch.empty((args.tokens, args.hidden), dtype=dtype, device=device)
        for _ in range(world_size)
    ]
    prime_same_node_buffers(
        X=X,
        X_buffers=X_buffers,
        packets=packets,
        node_rank=node_rank,
        local_gpu_index=local_gpu_index,
        gpus_per_node=args.gpus_per_node,
        local_group=local_group,
        expected_tokens=expected_tokens,
    )

    num_nodes = world_size // args.gpus_per_node
    rdma_recv = tuple(
        torch.empty((args.tokens, args.hidden), dtype=dtype, device=device)
        for _ in range(num_nodes)
    )
    local_packet = packets[rank]
    outgoing_offsets = list(local_packet["node_offsets"])
    outgoing_indices = list(local_packet["node_indices"])
    incoming_offsets = [0]
    incoming_masks: list[int] = []
    for source_node in range(num_nodes):
        source_rank = source_node * args.gpus_per_node + local_gpu_index
        source_packet = packets[source_rank]
        begin = source_packet["node_offsets"][node_rank]
        end = source_packet["node_offsets"][node_rank + 1]
        incoming_masks.extend(source_packet["node_masks"][begin:end])
        incoming_offsets.append(len(incoming_masks))

    host_a_idx_offsets = [0]
    host_a_idx: list[int] = []
    for indices in source_indices:
        host_a_idx.extend(indices)
        host_a_idx_offsets.append(len(host_a_idx))

    cluster_rows = args.tile_m * args.cluster_m
    total_per_expert = [
        sum(offsets[expert + 1] - offsets[expert] for offsets in source_offsets)
        for expert in range(experts_per_gpu)
    ]
    clusters_n = math.ceil(args.output_dim / args.tile_n)
    group_size = min(args.max_swizzle_size, clusters_n)
    n_groups = clusters_n // group_size
    table_rows = sum(math.ceil(count / cluster_rows) for count in total_per_expert) * n_groups
    if table_rows == 0:
        raise RuntimeError(
            "this rank received no routed tokens; zero-row gather tables are unsupported"
        )
    table_width = 2 + 2 * world_size
    work_table = torch.empty((table_rows, table_width), dtype=torch.int32, device=device)
    ready_rows = torch.zeros(1, dtype=torch.int32, device=device)
    output = torch.empty((world_size * index_capacity, args.output_dim), dtype=dtype, device=device)
    torch.cuda.synchronize(device)
    return PreparedInputs(
        X,
        W,
        R,
        tuple(X_buffers),
        tuple(A_idx),
        meaningful_counts,
        source_offsets,
        work_table,
        ready_rows,
        output,
        table_rows,
        table_width,
        group_size,
        build_output_segments(source_offsets, cluster_rows),
        rdma_recv,
        outgoing_offsets,
        outgoing_indices,
        incoming_offsets,
        incoming_masks,
        host_a_idx_offsets,
        host_a_idx,
    )


class CProxyConfig(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_int32),
        ("node_rank", ctypes.c_int32),
        ("num_nodes", ctypes.c_int32),
        ("local_gpu_index", ctypes.c_int32),
        ("gpus_per_node", ctypes.c_int32),
        ("cuda_device", ctypes.c_int32),
        ("num_experts", ctypes.c_int32),
        ("local_experts", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("dtype_bytes", ctypes.c_int32),
        ("num_qps_per_peer", ctypes.c_int32),
        ("tokens_per_chunk", ctypes.c_int32),
        ("forwarding_batch_tokens", ctypes.c_int32),
        ("completion_poll_batch", ctypes.c_int32),
        ("max_in_flight_chunks", ctypes.c_int32),
        ("send_queue_depth", ctypes.c_int32),
        ("recv_queue_depth", ctypes.c_int32),
        ("cq_depth", ctypes.c_int32),
        ("table_rows", ctypes.c_int32),
        ("table_width", ctypes.c_int32),
        ("gemm_tile_m", ctypes.c_int32),
        ("gemm_tile_n", ctypes.c_int32),
        ("gemm_cluster_m", ctypes.c_int32),
        ("gemm_output_dim", ctypes.c_int32),
        ("gemm_max_swizzle", ctypes.c_int32),
        ("flag_update_mode", ctypes.c_int32),
        ("mock_mode", ctypes.c_int32),
        ("num_tokens", ctypes.c_uint64),
        ("hidden_dim", ctypes.c_uint64),
        ("timeout_ms", ctypes.c_uint64),
        ("control_port_base", ctypes.c_uint16),
        ("rdma_port", ctypes.c_uint8),
        ("gid_index", ctypes.c_int32),
        ("rdma_device", ctypes.c_char_p),
        ("peer_hosts_csv", ctypes.c_char_p),
        ("exchange_dir", ctypes.c_char_p),
        ("run_id", ctypes.c_char_p),
        ("local_x", ctypes.c_uint64),
        ("local_x_bytes", ctypes.c_uint64),
        ("rdma_recv", ctypes.POINTER(ctypes.c_uint64)),
        ("rdma_recv_bytes", ctypes.POINTER(ctypes.c_uint64)),
        ("nvlink_recv", ctypes.POINTER(ctypes.c_uint64)),
        ("nvlink_recv_bytes", ctypes.POINTER(ctypes.c_uint64)),
        ("nvlink_ipc_handles", ctypes.POINTER(ctypes.c_uint8)),
        ("nvlink_ipc_offsets", ctypes.POINTER(ctypes.c_uint64)),
        ("a_idx", ctypes.POINTER(ctypes.c_uint64)),
        ("a_idx_counts", ctypes.POINTER(ctypes.c_int32)),
        ("work_table", ctypes.c_uint64),
        ("ready_rows", ctypes.c_uint64),
        ("outgoing_node_offsets", ctypes.POINTER(ctypes.c_int32)),
        ("outgoing_token_indices", ctypes.POINTER(ctypes.c_int32)),
        ("incoming_node_offsets", ctypes.POINTER(ctypes.c_int32)),
        ("incoming_token_masks", ctypes.POINTER(ctypes.c_uint8)),
        ("expert_offsets", ctypes.POINTER(ctypes.c_int32)),
        ("host_a_idx_offsets", ctypes.POINTER(ctypes.c_int32)),
        ("host_a_idx", ctypes.POINTER(ctypes.c_int32)),
    ]


def c_array(ctype, values):
    # ctypes accepts a zero-length array, but the native ABI still receives a stable pointer.
    return (ctype * max(1, len(values)))(*(values or [0]))


class ThreadProxy:
    def __init__(
        self,
        library_path: Path,
        args: argparse.Namespace,
        inputs: PreparedInputs,
        *,
        node_rank: int,
        local_gpu_index: int,
        num_nodes: int,
        experts_per_gpu: int,
        run_id: str,
    ) -> None:
        self.handle = None
        self.ipc_exports: list[CudaIpcExport] = []
        self.library = ctypes.CDLL(str(library_path.resolve()))
        self.library.smep_proxy_create.argtypes = [
            ctypes.POINTER(CProxyConfig),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self.library.smep_proxy_create.restype = ctypes.c_void_p
        for name in ("smep_proxy_start", "smep_proxy_wait", "smep_proxy_stop"):
            function = getattr(self.library, name)
            function.restype = ctypes.c_int
        self.library.smep_proxy_start.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self.library.smep_proxy_wait.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        self.library.smep_proxy_stop.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        self.library.smep_proxy_destroy.argtypes = [ctypes.c_void_p]

        world = num_nodes * args.gpus_per_node
        flat_expert_offsets = [item for offsets in inputs.expert_offsets for item in offsets]
        try:
            for tensor in inputs.X_buffers:
                self.ipc_exports.append(export_cuda_ipc_tensor(tensor))
        except Exception:
            for export in self.ipc_exports:
                export.release()
            raise
        flat_ipc_handles = [byte for export in self.ipc_exports for byte in export.handle]
        self.arrays = {
            "rdma_recv": c_array(
                ctypes.c_uint64, [tensor.data_ptr() for tensor in inputs.rdma_recv]
            ),
            "rdma_recv_bytes": c_array(
                ctypes.c_uint64,
                [tensor.numel() * tensor.element_size() for tensor in inputs.rdma_recv],
            ),
            "nvlink_recv": c_array(
                ctypes.c_uint64, [tensor.data_ptr() for tensor in inputs.X_buffers]
            ),
            "nvlink_recv_bytes": c_array(
                ctypes.c_uint64,
                [tensor.numel() * tensor.element_size() for tensor in inputs.X_buffers],
            ),
            "nvlink_ipc_handles": c_array(ctypes.c_uint8, flat_ipc_handles),
            "nvlink_ipc_offsets": c_array(
                ctypes.c_uint64, [export.offset_bytes for export in self.ipc_exports]
            ),
            "a_idx": c_array(ctypes.c_uint64, [tensor.data_ptr() for tensor in inputs.A_idx]),
            "a_idx_counts": c_array(ctypes.c_int32, inputs.a_idx_counts),
            "out_offsets": c_array(ctypes.c_int32, inputs.outgoing_offsets),
            "out_indices": c_array(ctypes.c_int32, inputs.outgoing_indices),
            "in_offsets": c_array(ctypes.c_int32, inputs.incoming_offsets),
            "in_masks": c_array(ctypes.c_uint8, inputs.incoming_masks),
            "expert_offsets": c_array(ctypes.c_int32, flat_expert_offsets),
            "host_idx_offsets": c_array(ctypes.c_int32, inputs.host_a_idx_offsets),
            "host_idx": c_array(ctypes.c_int32, inputs.host_a_idx),
        }
        self.strings = {
            "rdma_device": args.rdma_device.encode(),
            "peer_hosts": args.peer_hosts.encode(),
            "exchange_dir": args.exchange_dir.encode(),
            "run_id": run_id.encode(),
        }
        config = CProxyConfig(
            abi_version=1,
            node_rank=node_rank,
            num_nodes=num_nodes,
            local_gpu_index=local_gpu_index,
            gpus_per_node=args.gpus_per_node,
            cuda_device=torch.cuda.current_device(),
            num_experts=args.experts,
            local_experts=experts_per_gpu,
            top_k=args.top_k,
            dtype_bytes=inputs.X.element_size(),
            num_qps_per_peer=args.num_qps_per_peer,
            tokens_per_chunk=args.tokens_per_chunk,
            forwarding_batch_tokens=args.forwarding_batch_tokens,
            completion_poll_batch=args.completion_poll_batch,
            max_in_flight_chunks=args.max_in_flight_chunks,
            send_queue_depth=args.send_queue_depth,
            recv_queue_depth=args.recv_queue_depth,
            cq_depth=args.cq_depth,
            table_rows=inputs.table_rows,
            table_width=inputs.table_width,
            gemm_tile_m=args.tile_m,
            gemm_tile_n=args.tile_n,
            gemm_cluster_m=args.cluster_m,
            gemm_output_dim=args.output_dim,
            gemm_max_swizzle=args.max_swizzle_size,
            flag_update_mode=int(args.flag_update_mode == "stream-write"),
            mock_mode=int(args.mock_mode),
            num_tokens=args.tokens,
            hidden_dim=args.hidden,
            timeout_ms=args.timeout_ms,
            control_port_base=args.control_port_base,
            rdma_port=args.rdma_port,
            gid_index=args.gid_index,
            rdma_device=self.strings["rdma_device"],
            peer_hosts_csv=self.strings["peer_hosts"],
            exchange_dir=self.strings["exchange_dir"],
            run_id=self.strings["run_id"],
            local_x=inputs.X.data_ptr(),
            local_x_bytes=inputs.X.numel() * inputs.X.element_size(),
            rdma_recv=self.arrays["rdma_recv"],
            rdma_recv_bytes=self.arrays["rdma_recv_bytes"],
            nvlink_recv=self.arrays["nvlink_recv"],
            nvlink_recv_bytes=self.arrays["nvlink_recv_bytes"],
            nvlink_ipc_handles=self.arrays["nvlink_ipc_handles"],
            nvlink_ipc_offsets=self.arrays["nvlink_ipc_offsets"],
            a_idx=self.arrays["a_idx"],
            a_idx_counts=self.arrays["a_idx_counts"],
            work_table=inputs.work_table.data_ptr(),
            ready_rows=inputs.ready_rows.data_ptr(),
            outgoing_node_offsets=self.arrays["out_offsets"],
            outgoing_token_indices=self.arrays["out_indices"],
            incoming_node_offsets=self.arrays["in_offsets"],
            incoming_token_masks=self.arrays["in_masks"],
            expert_offsets=self.arrays["expert_offsets"],
            host_a_idx_offsets=self.arrays["host_idx_offsets"],
            host_a_idx=self.arrays["host_idx"],
        )
        self.config = config
        error = ctypes.create_string_buffer(4096)
        self.handle = self.library.smep_proxy_create(ctypes.byref(config), error, len(error))
        if not self.handle:
            for export in self.ipc_exports:
                export.release()
            raise RuntimeError(error.value.decode() or "smep_proxy_create failed")

    def _call(self, name: str, *args) -> None:
        error = ctypes.create_string_buffer(4096)
        result = getattr(self.library, name)(self.handle, *args, error, len(error))
        if result:
            raise RuntimeError(error.value.decode() or f"{name} failed")

    def start(self, iteration: int) -> None:
        self._call("smep_proxy_start", iteration)

    def wait(self) -> None:
        self._call("smep_proxy_wait")

    def close(self) -> None:
        if self.handle:
            error = ctypes.create_string_buffer(4096)
            self.library.smep_proxy_stop(self.handle, error, len(error))
            self.library.smep_proxy_destroy(self.handle)
            self.handle = None
            for export in self.ipc_exports:
                export.release()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def make_launch(args: argparse.Namespace, inputs: PreparedInputs):
    from quack.gemm import gemm as quack_gemm

    B = inputs.W.transpose(1, 2)

    def launch() -> None:
        quack_gemm(
            inputs.X_buffers,
            B,
            inputs.output,
            C=None,
            tile_count_semaphore=None,
            tile_M=args.tile_m,
            tile_N=args.tile_n,
            tile_K=args.tile_k,
            cluster_M=args.cluster_m,
            cluster_N=1,
            cluster_K=1,
            pingpong=False,
            persistent=True,
            is_dynamic_persistent=False,
            max_swizzle_size=args.max_swizzle_size,
            cu_seqlens_m=None,
            A_idx=inputs.A_idx,
            gather_work_table=inputs.work_table,
            gather_work_table_ready=inputs.ready_rows,
            multi_buffer_gather=True,
            use_tma_gather=False,
        )

    return launch


@torch.inference_mode()
def check_correctness(inputs: PreparedInputs, *, atol: float, rtol: float) -> float:
    """Compare every meaningful output row with a float32 gathered reference."""
    output_offset = 0
    max_abs_error = 0.0
    for expert, source, start, end in inputs.output_segments:
        if start == end:
            continue
        reference = (
            inputs.X_buffers[source][inputs.A_idx[source][start:end].long()].float()
            @ inputs.W[expert].float()
        ).to(inputs.output.dtype)
        actual = inputs.output[output_offset : output_offset + end - start]
        if actual.numel():
            max_abs_error = max(max_abs_error, (actual - reference).abs().max().item())
        torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
        output_offset += end - start
    if output_offset != sum(inputs.a_idx_counts):
        raise AssertionError("correctness segments do not cover all meaningful routed rows")
    return max_abs_error


def run_one(
    iteration: int,
    *,
    proxy: ThreadProxy,
    launch,
    inputs: PreparedInputs,
    device: torch.device,
    timed: bool,
) -> float | None:
    dist.barrier()
    inputs.work_table.fill_(-1)
    inputs.ready_rows.zero_()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True) if timed else None
    end = torch.cuda.Event(enable_timing=True) if timed else None
    if start:
        start.record()
    launch()
    proxy.start(iteration)
    proxy.wait()
    if end:
        end.record()
        end.synchronize()
        return start.elapsed_time(end)
    torch.cuda.synchronize(device)
    return None


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device_bindings = parse_device_map(args.device_map)
    validate_device_map_topology(
        device_bindings, args.gpus_per_node, torch.cuda.device_count()
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_bindings:
        if local_rank not in device_bindings:
            raise ValueError(f"--device-map has no entry for LOCAL_RANK={local_rank}")
        local_binding = device_bindings[local_rank]
        cuda_device_id = local_binding.cuda_device_id
        args.rdma_device = local_binding.rdma_device_name
    else:
        cuda_device_id = local_rank
        local_binding = DeviceBinding(local_rank, cuda_device_id, args.rdma_device)
    if cuda_device_id >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device {cuda_device_id} selected for LOCAL_RANK={local_rank}, but only "
            f"{torch.cuda.device_count()} PyTorch-visible devices exist"
        )
    torch.cuda.set_device(cuda_device_id)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    validate_args(args, world_size)
    num_nodes = world_size // args.gpus_per_node
    node_rank, local_gpu_index = divmod(rank, args.gpus_per_node)
    if local_rank != local_gpu_index:
        raise RuntimeError(
            "LOCAL_RANK must match rank % gpus_per_node; use one torchrun worker per GPU"
        )
    hosts = args.peer_hosts.split(",")
    if len(hosts) != num_nodes or any(not host for host in hosts):
        raise ValueError("--peer-hosts must provide one non-empty host for every node")
    device = torch.device("cuda", cuda_device_id)
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 9:
        raise RuntimeError(
            "Grouped_GEMM_Stream_Gather requires SM90, "
            f"got SM{capability[0]}{capability[1]}"
        )
    run_id = distributed_run_id(args.run_id, rank)
    groups = local_node_groups(num_nodes, args.gpus_per_node)
    inputs = prepare_inputs(
        args,
        rank=rank,
        world_size=world_size,
        node_rank=node_rank,
        local_gpu_index=local_gpu_index,
        device=device,
        local_group=groups[node_rank],
    )
    experts_per_gpu = args.experts // world_size
    proxy = ThreadProxy(
        args.proxy_library,
        args,
        inputs,
        node_rank=node_rank,
        local_gpu_index=local_gpu_index,
        num_nodes=num_nodes,
        experts_per_gpu=experts_per_gpu,
        run_id=run_id,
    )
    dist.barrier()
    binding_report = {
        "global_rank": rank,
        "local_gpu_index": local_gpu_index,
        "cuda_device_id": local_binding.cuda_device_id,
        "rdma_device_name": local_binding.rdma_device_name,
    }
    all_bindings: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(all_bindings, binding_report)
    launch = make_launch(args, inputs)
    if rank == 0:
        print(
            f"Proxy_C + Grouped_GEMM_Stream_Gather: nodes={num_nodes}, "
            f"GPUs/node={args.gpus_per_node}, world={world_size}, run_id={run_id}"
        )
        print(
            f"X={tuple(inputs.X.shape)}, W/local={tuple(inputs.W.shape)}, "
            f"R={tuple(inputs.R.shape)}, table={tuple(inputs.work_table.shape)}"
        )
        print("CUDA/RDMA bindings by global rank:")
        for binding in all_bindings:
            print(
                f"  rank {binding['global_rank']}: local_gpu_index="
                f"{binding['local_gpu_index']} cuda_device_id={binding['cuda_device_id']} "
                f"rdma_device={binding['rdma_device_name']}"
            )
        print("Compiling and warming up the specialized QuACK kernel...")
    next_iteration = 1
    for _ in range(args.warmup):
        run_one(
            next_iteration,
            proxy=proxy,
            launch=launch,
            inputs=inputs,
            device=device,
            timed=False,
        )
        next_iteration += 1
    timings = []
    for _ in range(args.iterations):
        elapsed = run_one(
            next_iteration,
            proxy=proxy,
            launch=launch,
            inputs=inputs,
            device=device,
            timed=True,
        )
        timings.append(float(elapsed))
        next_iteration += 1
    local_error = None if args.skip_check else check_correctness(
        inputs, atol=args.atol, rtol=args.rtol
    )
    gathered_timings: list[list[float] | None] = [None] * world_size
    dist.all_gather_object(gathered_timings, timings)
    gathered_errors: list[float | None] = [None] * world_size
    dist.all_gather_object(gathered_errors, local_error)
    if rank == 0:
        per_rank_medians = [statistics.median(values) for values in gathered_timings if values]
        print(f"Per-rank median latency (ms): {[round(value, 4) for value in per_rank_medians]}")
        print(f"Global median latency (ms): {statistics.median(per_rank_medians):.4f}")
        print(
            "The timed interval includes Proxy_C RDMA, NVLink compaction, completion "
            "processing, gather-table publication, and GroupedGEMM."
        )
        if not args.skip_check:
            print(f"Correctness: PASS, global max abs error={max(gathered_errors):.6g}")
    proxy.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
