"""One-time variable-sized CUDA IPC dispatch driven by destination bitmasks."""

from __future__ import annotations

import ctypes
import os
import time

import torch

try:
    from run.token_exchange import CudaCopies, SharedExchange
except ModuleNotFoundError as error:
    if error.name != "run":
        raise
    from token_exchange import CudaCopies, SharedExchange


def destination_indices(metadata, rank, world):
    """Return ascending source-token indices, indexed by absolute destination rank."""
    if not 1 <= world <= 8 or not 0 <= rank < world:
        raise ValueError("bitmask routing requires 1..8 ranks and a valid source rank")
    return tuple(
        torch.nonzero(metadata & (1 << ((destination - rank - 1) % world)),
                      as_tuple=True)[0]
        for destination in range(world)
    )


def make_router_metadata(tokens, world, probability, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    bits = torch.rand((tokens, world), generator=generator, device=device) < probability
    weights = 1 << torch.arange(world, device=device)
    return (bits * weights).sum(dim=1).to(torch.uint8)


class BitmaskRoutingPlan:
    def __init__(self, args, device):
        self.rank = args.local_rank
        self.world = int(os.environ["LOCAL_WORLD_SIZE"])
        # A separate generator leaves weights and local-expert routing RNG untouched.
        self.metadata = make_router_metadata(
            args.source_tokens, self.world, args.token_route_probability,
            args.seed + 104729 + self.rank, device,
        )
        self.indices = tuple(index.cpu() for index in
                             destination_indices(self.metadata, self.rank, self.world))
        self.outgoing_counts = [index.numel() for index in self.indices]
        control = SharedExchange(self.rank, self.world, args.token_exchange_timeout,
                                 namespace="bitmask-plan")
        try:
            signature = [args.source_tokens, args.hidden, args.dtype, args.seed,
                         args.token_route_probability, args.local_routes_per_token,
                         os.environ.get("CUDA_VISIBLE_DEVICES")]
            if any(value != signature for value in control.gather("configuration", signature)):
                raise ValueError("bitmask exchange requires matching routing configuration")
            self.counts = control.gather("counts", self.outgoing_counts)
            self.incoming_counts = [row[self.rank] for row in self.counts]
            control.gather("complete")
        except BaseException as error:
            control.state("error", str(error))
            raise
        finally:
            control.close()

    def exchange(self, args, inputs, export_allocation):
        control = SharedExchange(self.rank, self.world, args.token_exchange_timeout,
                                 namespace="bitmask-data")
        exports = []
        cuda = CudaCopies()
        try:
            generator = torch.Generator(device=inputs.W.device).manual_seed(args.seed + self.rank)
            self.source = torch.randn((args.source_tokens, args.hidden), generator=generator,
                                      device=inputs.W.device, dtype=inputs.W.dtype)
            for buffer in inputs.X:
                exports.append(export_allocation(buffer) if buffer.numel() else None)
            torch.cuda.synchronize()
            allocations = control.gather("allocations", [
                [item.handle_hex, item.allocation_offset_bytes] if item else None
                for item in exports
            ])
            library = ctypes.CDLL(str(args.thread_proxy_library.resolve()))
            try:
                batch = library.stream_gather_copy_token_batch
            except AttributeError as error:
                raise RuntimeError("Rebuild run/cpu_proxy with CUDA >=12.9 for bitmask routing") from error
            batch.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_int64),
                              ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint64,
                              ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]
            batch.restype = ctypes.c_int
            # A non-default stream is required by the batch API. Source initialization
            # and allocation exports above have already been synchronized.
            stream = torch.cuda.Stream()
            row_bytes = args.hidden * self.source.element_size()
            start = time.perf_counter()
            for destination, indices in enumerate(self.indices):
                if not indices.numel():
                    continue
                pointer = (inputs.X[self.rank].data_ptr() if destination == self.rank else
                           cuda.peer_pointer(allocations[destination][self.rank],
                                             self.rank, destination))
                error = ctypes.create_string_buffer(4096)
                result = batch(pointer, self.source.data_ptr(),
                               ctypes.cast(indices.data_ptr(), ctypes.POINTER(ctypes.c_int64)),
                               indices.numel(), row_bytes, stream.cuda_stream, error, len(error))
                if result:
                    raise RuntimeError(error.value.decode(errors="replace"))
            stream.synchronize()
            control.gather("copied")
            elapsed = time.perf_counter() - start
            cuda.close()
            control.gather("imports_closed")
            for item in exports:
                if item:
                    item.release()
            if inputs.original_X is not None:
                for saved, buffer in zip(inputs.original_X, inputs.X):
                    saved.copy_(buffer)
            if not args.skip_check:
                for sender, actual in enumerate(inputs.X):
                    generator.manual_seed(args.seed + sender)
                    source = torch.randn(self.source.shape, generator=generator,
                                         device=self.source.device, dtype=self.source.dtype)
                    metadata = make_router_metadata(
                        args.source_tokens, self.world, args.token_route_probability,
                        args.seed + 104729 + sender, self.source.device,
                    )
                    selected = destination_indices(metadata, sender, self.world)[self.rank]
                    torch.testing.assert_close(actual, source[selected], rtol=0, atol=0)
            torch.cuda.synchronize()
            control.gather("validated")
            print(f"Bitmask exchange: outgoing={self.outgoing_counts}, "
                  f"incoming={self.incoming_counts}, p={args.token_route_probability}; "
                  f"{elapsed * 1000:.3f} ms including barrier (excluded from GEMM timing)")
        except BaseException as error:
            control.state("error", str(error))
            raise
        finally:
            control.close()
