"""Persistent receive buffers and IPC mappings for proxy-triggered result return."""

from __future__ import annotations

import ctypes
import math
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
        plan = getattr(inputs, "routing_plan", None)
        if plan is None:
            self.received = tuple(torch.empty_like(buffer) for buffer in self.outputs)
        else:
            self.received = tuple(torch.empty(
                (args.hidden // args.down_tile_n, count * args.down_tile_n)
                if args.down_n_tile_major_output else (count, args.hidden),
                dtype=self.outputs[0].dtype, device=self.outputs[0].device,
            ) for count in plan.outgoing_counts)
        self.source_token_indices = plan.indices if plan is not None else None
        self.final_combine = None
        self.combined_output = None
        self.final_combine_tile_n = args.down_tile_n if args.down_n_tile_major_output else 0
        if args.final_combine:
            if __package__:
                from .final_combine import FinalCombine
            else:
                from final_combine import FinalCombine
            self.final_combine = FinalCombine(
                self.received, self.source_token_indices, args.source_tokens, args.hidden,
                self.final_combine_tile_n, num_ctas=args.final_combine_ctas,
                elements_per_thread=args.final_combine_elements_per_thread,
                threads_per_cta=args.final_combine_threads_per_cta,
                column_group_major=args.final_combine_column_group_major,
                pipelined=args.final_combine_pipelined,
                pipeline_stages=args.final_combine_pipeline_stages,
            )
            self.combined_output = self.final_combine.output
        self.down_done = torch.cuda.Event()
        # Materialize the event before passing its handle to the native worker.
        self.down_done.record()
        torch.cuda.synchronize()
        try:
            signature = [args.hidden if plan is not None else list(self.outputs[0].shape),
                         plan is not None, str(self.outputs[0].dtype),
                         args.down_n_tile_major_output, args.down_tile_n,
                         os.environ.get("CUDA_VISIBLE_DEVICES"), args.warmup,
                         args.iterations, args.timing_samples, args.skip_check,
                         args.return_token_results_per_group, args.down_max_swizzle_size]
            signatures = self.control.gather("configuration", signature)
            if any(value != signature for value in signatures):
                raise ValueError("Return exchange requires matching output layouts on all ranks")
            for buffer in (*self.received, *self.outputs):
                if not buffer.is_contiguous() or (
                    plan is None and buffer.numel() != self.outputs[0].numel()
                ):
                    raise ValueError("Return exchange requires equal-sized contiguous buffers")
                self.exports.append(export_allocation(buffer) if buffer.numel() else None)
            metadata = self.control.gather("allocations", [
                [item.handle_hex, item.allocation_offset_bytes] if item else None
                for item in self.exports
            ])
            self.metadata = metadata
            destinations = [
                0 if not self.outputs[peer].numel() else
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
                max(1, self.outputs[0].numel() * self.outputs[0].element_size()),
                self.down_done.cuda_event, ctypes.addressof(self.counter_view),
                args.token_exchange_timeout, error, len(error),
            )
            if result:
                raise RuntimeError(error.value.decode(errors="replace"))
            if plan is not None:
                configure_sizes = proxy.library.stream_gather_thread_proxy_set_return_sizes
                configure_sizes.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                           ctypes.POINTER(ctypes.c_size_t), ctypes.c_int,
                                           ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]
                configure_sizes.restype = ctypes.c_int
                sizes = (ctypes.c_size_t * self.world)(*[
                    buffer.numel() * buffer.element_size() for buffer in self.outputs
                ])
                if configure_sizes(proxy.handle, self.world, sizes,
                                   int(args.return_token_results_per_group), error, len(error)):
                    raise RuntimeError(error.value.decode(errors="replace"))
            elif args.return_token_results_per_group:
                # Physical shape is [N tiles, tokens * tile_n], with no padding for
                # in-place combine. A column group consists of adjacent N tiles.
                down = inputs.down_args
                group_tiles = min(down.max_swizzle_size, math.ceil(down.output_dim / down.tile_n))
                group_bytes = (group_tiles * self.outputs[0].shape[1]
                               * self.outputs[0].element_size())
                try:
                    configure_groups = proxy.library.stream_gather_thread_proxy_set_return_groups
                except AttributeError as error:
                    raise RuntimeError(
                        "Rebuild run/cpu_proxy to enable --return-token-results-per-group"
                    ) from error
                configure_groups.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                             ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]
                configure_groups.restype = ctypes.c_int
                if configure_groups(proxy.handle, group_bytes, error, len(error)):
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
        stream = torch.cuda.current_stream().cuda_stream
        for sender, actual in enumerate(self.received):
            if not actual.numel():
                continue
            expected = torch.empty_like(actual)
            size = expected.numel() * expected.element_size()
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
        if self.final_combine is not None:
            reference = torch.zeros_like(self.combined_output, dtype=torch.float32)
            hidden = reference.shape[1]
            tile_n = self.final_combine_tile_n
            for buffer, index in zip(self.received, self.source_token_indices):
                count = index.numel()
                logical = (buffer.view(hidden // tile_n, count, tile_n)
                           .permute(1, 0, 2).reshape(count, hidden)
                           if tile_n else buffer.view(count, hidden))
                reference.index_add_(0, index.to(reference.device), logical.float())
            torch.testing.assert_close(self.combined_output, reference.to(self.combined_output.dtype),
                                       rtol=0, atol=0)
            print("Final-combine reference check: PASSED (source-token sums, including zeros)")

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
                    if item:
                        item.release()
            else:
                self.control.state("error", "return exchange aborted")
        finally:
            if hasattr(self, "counter_view"):
                del self.counter_view
            self.control.close()
