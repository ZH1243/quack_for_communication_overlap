"""One-time, single-node CUDA IPC token exchange for torchrun workers."""

from __future__ import annotations

import ctypes
import hashlib
import json
import mmap
import os
import time
from pathlib import Path


class SharedExchange:
    """Small /dev/shm control plane protected by a process-shared file lock."""

    def __init__(self, rank, world, timeout, namespace="tokens"):
        import fcntl

        self.fcntl = fcntl
        self.rank, self.world, self.timeout = rank, world, timeout
        identity = (os.getuid(), os.environ.get("TORCHELASTIC_RUN_ID"),
                    os.environ.get("MASTER_ADDR"), os.environ.get("MASTER_PORT"))
        digest = hashlib.sha256(repr(identity).encode()).hexdigest()[:24]
        self.path = Path("/dev/shm") / f"quack-{namespace}-{digest}"
        size = max(1048576, world * world * 1024)
        deadline = time.monotonic() + timeout
        if rank == 0:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(self.fd, size)
        else:
            while True:
                try:
                    self.fd = os.open(self.path, os.O_RDWR)
                    break
                except FileNotFoundError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Waiting for rank 0 token-exchange shared memory")
                    time.sleep(0.01)
            while os.fstat(self.fd).st_size != size:
                if time.monotonic() >= deadline:
                    os.close(self.fd)
                    raise TimeoutError("Waiting for token-exchange shared memory sizing")
                time.sleep(0.01)
        self.memory = mmap.mmap(self.fd, size)

    def state(self, key=None, value=None):
        self.fcntl.flock(self.fd, self.fcntl.LOCK_EX)
        try:
            length = int.from_bytes(self.memory[:8], "little")
            state = json.loads(self.memory[8:8 + length]) if length else {}
            if key is not None:
                state.setdefault(key, {})[str(self.rank)] = value
                data = json.dumps(state).encode()
                if len(data) + 8 > len(self.memory) - self.world * 8:
                    raise RuntimeError("Token exchange control metadata exceeds shared memory")
                self.memory[8:8 + len(data)] = data
                self.memory[:8] = len(data).to_bytes(8, "little")
            return state
        finally:
            self.fcntl.flock(self.fd, self.fcntl.LOCK_UN)

    def gather(self, stage, value=True):
        self.state(stage, value)
        deadline = time.monotonic() + self.timeout
        while True:
            state = self.state()
            if state.get("error"):
                raise RuntimeError(f"Token exchange worker failed: {state['error']}")
            values = state.get(stage, {})
            if len(values) == self.world:
                return [values[str(rank)] for rank in range(self.world)]
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Token exchange timed out at {stage}; arrived: {list(values)}")
            time.sleep(0.01)

    def close(self):
        self.memory.close()
        os.close(self.fd)
        if self.rank == 0:
            self.path.unlink(missing_ok=True)


class CudaCopies:
    """Use the CUDA runtime already loaded by PyTorch, without another CUDA context."""

    class Handle(ctypes.Structure):
        _fields_ = [("reserved", ctypes.c_char * 64)]

    def __init__(self):
        paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                 if "/libcudart.so" in line}
        if len(paths) != 1:
            raise RuntimeError(f"Expected one loaded CUDA runtime, found {sorted(paths)}")
        self.lib = ctypes.CDLL(paths.pop())
        pointer, integer, size = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
        signatures = {
            "cudaIpcOpenMemHandle": [ctypes.POINTER(pointer), self.Handle, ctypes.c_uint],
            "cudaIpcCloseMemHandle": [pointer],
            "cudaDeviceCanAccessPeer": [ctypes.POINTER(integer), integer, integer],
            "cudaMemcpyPeerAsync": [pointer, integer, pointer, integer, size, pointer],
            "cudaMemcpyAsync": [pointer, pointer, size, integer, pointer],
            "cudaStreamSynchronize": [pointer],
        }
        for name, arguments in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, integer
        self.lib.cudaGetErrorString.argtypes = [integer]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p
        self.imports = {}

    def call(self, name, *args):
        error = getattr(self.lib, name)(*args)
        if error:
            raise RuntimeError(f"{name}: {self.lib.cudaGetErrorString(error).decode()}")

    def peer_pointer(self, metadata, source_device, destination_device):
        accessible = ctypes.c_int()
        self.call("cudaDeviceCanAccessPeer", ctypes.byref(accessible),
                  source_device, destination_device)
        if not accessible.value:
            raise RuntimeError(
                f"No direct peer access: GPU {source_device} -> {destination_device}"
            )
        handle, offset = metadata
        if handle not in self.imports:
            pointer = ctypes.c_void_p()
            self.call("cudaIpcOpenMemHandle", ctypes.byref(pointer),
                      self.Handle.from_buffer_copy(bytes.fromhex(handle)), 1)
            self.imports[handle] = pointer.value
        return self.imports[handle] + offset

    def close(self):
        for pointer in self.imports.values():
            self.call("cudaIpcCloseMemHandle", pointer)
        self.imports.clear()


def copy_chunks(cuda, rank, world, source, local_destination, metadata, chunk_bytes, stream):
    """Push chunk k to destination (rank+k+1)%world, buffer indexed by source rank."""
    for chunk in range(world):
        destination = (rank + chunk + 1) % world
        source_pointer = source + chunk * chunk_bytes
        if destination == rank:
            cuda.call("cudaMemcpyAsync", local_destination, source_pointer, chunk_bytes, 3, stream)
        else:
            pointer = cuda.peer_pointer(metadata[destination][rank], rank, destination)
            cuda.call("cudaMemcpyPeerAsync", pointer, destination, source_pointer,
                      rank, chunk_bytes, stream)
    cuda.call("cudaStreamSynchronize", stream)


def exchange_tokens(args, inputs, export_allocation):
    import torch

    rank, world = args.local_rank, int(os.environ["LOCAL_WORLD_SIZE"])
    control = SharedExchange(rank, world, args.token_exchange_timeout)
    exports = []
    cuda = None
    try:
        signature = [args.source_tokens, args.hidden, args.dtype, args.seed,
                     os.environ.get("CUDA_VISIBLE_DEVICES")]
        signatures = control.gather("configuration", signature)
        if any(item != signature for item in signatures):
            raise ValueError(
                "Token exchange requires matching shapes, dtype, seed and GPU visibility"
            )
        # A private generator leaves the runner's existing routing/weight RNG unchanged.
        generator = torch.Generator(device=inputs.X[0].device).manual_seed(args.seed + rank)
        source = torch.randn((args.source_tokens, args.hidden), generator=generator,
                             device=inputs.X[0].device, dtype=inputs.X[0].dtype)
        for buffer in inputs.X:
            exports.append(export_allocation(buffer))
        torch.cuda.synchronize()
        metadata = control.gather("ready", [
            [item.handle_hex, item.allocation_offset_bytes] for item in exports
        ])
        cuda = CudaCopies()
        stream = torch.cuda.current_stream().cuda_stream
        chunk_bytes = inputs.X[0].numel() * inputs.X[0].element_size()
        start = time.perf_counter()
        copy_chunks(cuda, rank, world, source.data_ptr(), inputs.X[rank].data_ptr(),
                    metadata, chunk_bytes, stream)
        control.gather("copies_complete")
        elapsed = time.perf_counter() - start
        cuda.close()
        control.gather("imports_closed")
        for item in exports:
            item.release()
        # In-place down GEMM restores these snapshots on every subsequent invocation.
        if inputs.original_X is not None:
            for original, buffer in zip(inputs.original_X, inputs.X):
                original.copy_(buffer)
        if not args.skip_check:
            tokens = args.source_tokens // world
            for sender, buffer in enumerate(inputs.X):
                generator.manual_seed(args.seed + sender)
                expected = torch.randn(source.shape, generator=generator,
                                       device=source.device, dtype=source.dtype)
                chunk = (rank - sender - 1) % world
                torch.testing.assert_close(buffer, expected[chunk * tokens:(chunk + 1) * tokens],
                                           rtol=0, atol=0)
        torch.cuda.synchronize()
        control.gather("validated")
        print(f"One-time token exchange: {world} copies/rank, {args.source_tokens} source tokens, "
              f"{elapsed * 1000:.3f} ms including completion barrier (excluded from GEMM timing)")
    except BaseException as error:
        control.state("error", str(error))
        raise
    finally:
        # Never release exported storage before all remote imports have closed.
        # On failure, retain IPC references until process exit rather than race a peer copy.
        control.close()
