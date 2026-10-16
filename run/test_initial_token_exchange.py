#!/usr/bin/env python3
"""Numerical CUDA IPC exchange and restore regression; launch with single-node torchrun."""

import os
from types import SimpleNamespace

import torch

from hopper_stream_gather_table_gemm import export_cuda_ipc_allocation
from token_exchange import exchange_tokens


def main():
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    for dtype, seed in ((torch.float32, 123), (torch.bfloat16, 456)):
        tokens, hidden = 19, 32
        args = SimpleNamespace(
            local_rank=rank, token_exchange_timeout=120.0,
            source_tokens=world * tokens, hidden=hidden, dtype=str(dtype),
            seed=seed, skip_check=False,
        )
        with torch.inference_mode():
            buffers = tuple(torch.full((tokens, hidden), float("nan"),
                                       device=device, dtype=dtype) for _ in range(world))
            inputs = SimpleNamespace(X=buffers, original_X=tuple(t.clone() for t in buffers))
            exchange_tokens(args, inputs, export_cuda_ipc_allocation)
            for sender, (buffer, saved) in enumerate(zip(inputs.X, inputs.original_X)):
                generator = torch.Generator(device=device).manual_seed(seed + sender)
                source = torch.randn((world * tokens, hidden), generator=generator,
                                     device=device, dtype=dtype)
                # Explicitly enumerate the sender's destination sequence as the reference.
                destinations = [(sender + offset) % world for offset in range(1, world + 1)]
                reference = source.split(tokens)[destinations.index(rank)]
                torch.testing.assert_close(buffer.float(), reference.float(), rtol=0, atol=0)
                torch.testing.assert_close(saved.float(), reference.float(), rtol=0, atol=0)
                buffer.fill_(-999)
                buffer.copy_(saved)
                torch.testing.assert_close(buffer.float(), reference.float(), rtol=0, atol=0)
        # exchange_tokens removes its shared-memory name on rank 0. Rendezvous before
        # reusing the same name for the next case (CPU Gloo is only used by this test).
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("gloo")
        torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    print(f"Rank {rank}: token mapping and in-place restoration numerical checks passed")


if __name__ == "__main__":
    main()
