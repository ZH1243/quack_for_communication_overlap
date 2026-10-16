"""Persistent receive buffers and IPC mappings for proxy-triggered result return."""

from __future__ import annotations

import ctypes
import os

import torch

try:
    from run.token_exchange import CudaCopies, SharedExchange
except ModuleNotFoundError as error:
    if error.name != "run":
        raise
    from token_exchange import CudaCopies, SharedExchange


class ReturnExchange:
    def __init__(self, args, inputs, proxy, export_allocation):
        self.rank = args.local_rank
        self.world = int(os.environ["LOCAL_WORLD_SIZE"])
        self.control = SharedExchange(
            self.rank, self.world, args.token_exchange_timeout, namespace="return"
        )
        self.exports = []
        self.cuda = CudaCopies()
        self.closed = False
        self.outputs = inputs.down_inputs.output_buffers()
        self.received = tuple(torch.empty_like(buffer) for buffer in self.outputs)
        self.down_done = torch.cuda.Event()
        # Materialize the event before passing its handle to the native worker.
        self.down_done.record()
        torch.cuda.synchronize()
        try:
            signature = [list(self.outputs[0].shape), str(self.outputs[0].dtype),
                         args.down_n_tile_major_output, args.down_tile_n,
                         os.environ.get("CUDA_VISIBLE_DEVICES"), args.warmup,
                         args.iterations, args.timing_samples, args.skip_check]
            signatures = self.control.gather("configuration", signature)
            if any(value != signature for value in signatures):
                raise ValueError("Return exchange requires matching output layouts on all ranks")
            for buffer in (*self.received, *self.outputs):
                if not buffer.is_contiguous() or buffer.numel() != self.outputs[0].numel():
                    raise ValueError("Return exchange requires equal-sized contiguous buffers")
                self.exports.append(export_allocation(buffer))
            metadata = self.control.gather("allocations", [
                [item.handle_hex, item.allocation_offset_bytes] for item in self.exports
            ])
            self.metadata = metadata
            destinations = [
                self.received[self.rank].data_ptr() if peer == self.rank else
                self.cuda.peer_pointer(metadata[peer][self.rank], self.rank, peer)
                for peer in range(self.world)
            ]
            self.counter_view = (ctypes.c_uint64 * self.world).from_buffer(
                self.control.memory, len(self.control.memory) - self.world * 8
            )
            try:
                configure = proxy.library.stream_gather_thread_proxy_set_return_copies
            except AttributeError as error:
                raise RuntimeError("Rebuild run/cpu_proxy to enable --return-token-results") from error
            pointer_array = ctypes.c_uint64 * self.world
            configure.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_uint64,
                ctypes.c_uint64, ctypes.c_double, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t,
            ]
            configure.restype = ctypes.c_int
            error = ctypes.create_string_buffer(4096)
            result = configure(
                proxy.handle, self.world,
                pointer_array(*(buffer.data_ptr() for buffer in self.outputs)),
                pointer_array(*destinations),
                self.outputs[0].numel() * self.outputs[0].element_size(),
                self.down_done.cuda_event, ctypes.addressof(self.counter_view),
                args.token_exchange_timeout, error, len(error),
            )
            if result:
                raise RuntimeError(error.value.decode(errors="replace"))
            self.control.gather("configured")
        except BaseException as error:
            self.control.state("error", str(error))
            # The process exits on setup failure; do not release storage a peer may import.
            raise

    def record_down_done(self):
        self.down_done.record(torch.cuda.current_stream())

    @torch.inference_mode()
    def check(self):
        """Compare every returned physical buffer with its computing GPU's output."""
        self.control.gather("check_start")
        expected = torch.empty_like(self.received[0])
        stream = torch.cuda.current_stream().cuda_stream
        size = expected.numel() * expected.element_size()
        for sender, actual in enumerate(self.received):
            if sender == self.rank:
                expected.copy_(self.outputs[self.rank])
            else:
                pointer = self.cuda.peer_pointer(
                    self.metadata[sender][self.world + self.rank], self.rank, sender
                )
                self.cuda.call("cudaMemcpyPeerAsync", expected.data_ptr(), self.rank,
                               pointer, sender, size, stream)
            self.cuda.call("cudaStreamSynchronize", stream)
            # Exact bytes, including N-tile-major ordering; GEMM values are checked separately.
            torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8),
                                       rtol=0, atol=0)
        self.control.gather("check_complete")
        print("Return-copy reference check: PASSED (all received physical buffers match outputs)")

    def close(self, success=True):
        """Call only after the native proxy worker has been destroyed."""
        if self.closed:
            return
        self.closed = True
        try:
            if success:
                self.control.gather("workers_stopped")
                self.cuda.close()
                self.control.gather("imports_closed")
                for item in self.exports:
                    item.release()
            else:
                self.control.state("error", "return exchange aborted")
        finally:
            if hasattr(self, "counter_view"):
                del self.counter_view
            self.control.close()
