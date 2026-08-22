"""SM90 gather-table GEMM tests."""

import math
import time

import pytest
import torch

from quack.cute_dsl_utils import get_device_capacity
from quack.gemm import gemm as quack_gemm


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_device_capacity(torch.device("cuda"))[0] != 9,
    reason="gather_work_table is SM90 only",
)


@torch.inference_mode()
@pytest.mark.parametrize(("tile_m", "num_buffers"), [(64, 5), (128, 32), (192, 16)])
def test_multi_buffer_gather_selects_one_source_row(tile_m, num_buffers):
    """Loader threads cooperatively prepare one address for each packed row."""
    torch.manual_seed(0)
    num_experts = 2
    tokens, routes, k, n = 32, 19, 64, 128
    counts = (10, 9)

    x_buffers = tuple(
        torch.randn(tokens, k, dtype=torch.bfloat16, device="cuda")
        for _ in range(num_buffers)
    )
    a_idx_buffers = tuple(
        torch.randperm(tokens, dtype=torch.int32, device="cuda")[:routes]
        for _ in range(num_buffers)
    )
    weights = torch.randn(
        num_experts, n, k, dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(k)

    # Build one work-table row per M cluster. This covers tile_M below, equal
    # to, and above the 128 address-preparation threads; CTA boundaries also
    # cross input-buffer boundaries and the final cluster is ragged.
    rows = []
    expert_offsets = (0, counts[0], routes)
    for expert in range(num_experts):
        start, end = expert_offsets[expert : expert + 2]
        count = end - start
        packed_count = num_buffers * count
        for packed_start in range(0, packed_count, 2 * tile_m):
            packed_end = min(packed_start + 2 * tile_m, packed_count)
            ranges = []
            for buffer_idx in range(num_buffers):
                buffer_start = buffer_idx * count
                local_start = min(max(packed_start - buffer_start, 0), count)
                local_end = min(max(packed_end - buffer_start, 0), count)
                ranges.extend((start + local_start, start + local_end))
            rows.append((expert, 0, *ranges))
    work_table = torch.tensor(rows, dtype=torch.int32, device="cuda")

    output = torch.empty(num_buffers * routes, n, dtype=torch.bfloat16, device="cuda")
    quack_gemm(
        x_buffers,
        weights,
        output,
        C=None,
        tile_count_semaphore=None,
        tile_M=tile_m,
        tile_N=128,
        cluster_M=2,
        cluster_N=1,
        cluster_K=1,
        pingpong=False,
        persistent=True,
        is_dynamic_persistent=False,
        cu_seqlens_m=None,
        A_idx=a_idx_buffers,
        gather_work_table=work_table,
        multi_buffer_gather=True,
        use_tma_gather=False,
    )

    reference = torch.cat(
        [
            x_buffers[buffer_idx][
                a_idx_buffers[buffer_idx][start:end].long()
            ].float()
            @ weights[expert].float().mT
            for expert, (start, end) in enumerate(zip(expert_offsets[:-1], expert_offsets[1:]))
            for buffer_idx in range(num_buffers)
        ]
    ).to(output.dtype)
    torch.testing.assert_close(output, reference, atol=3e-2, rtol=1e-3)


@torch.inference_mode()
def test_multi_buffer_gather_waits_for_table_ready_prefix():
    """A copy-engine flag publication releases a scheduler initially polling HBM."""
    torch.manual_seed(0)
    tokens, routes, k, n = 32, 17, 64, 128
    x_buffers = tuple(
        torch.randn(tokens, k, dtype=torch.bfloat16, device="cuda") for _ in range(2)
    )
    a_idx_buffers = tuple(
        torch.randperm(tokens, dtype=torch.int32, device="cuda")[:routes] for _ in range(2)
    )
    weights = torch.randn(1, n, k, dtype=torch.bfloat16, device="cuda") / math.sqrt(k)
    work_table = torch.tensor(
        [[0, 0, 0, routes, 0, routes]], dtype=torch.int32, device="cuda"
    )
    ready_rows = torch.zeros(1, dtype=torch.int32, device="cuda")
    host_ready = torch.tensor([1], dtype=torch.int32, pin_memory=True)
    output = torch.empty(2 * routes, n, dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()

    quack_gemm(
        x_buffers,
        weights,
        output,
        C=None,
        tile_count_semaphore=None,
        tile_M=64,
        tile_N=128,
        cluster_M=2,
        cluster_N=1,
        cluster_K=1,
        pingpong=False,
        persistent=True,
        is_dynamic_persistent=False,
        cu_seqlens_m=None,
        A_idx=a_idx_buffers,
        gather_work_table=work_table,
        gather_work_table_ready=ready_rows,
        multi_buffer_gather=True,
        use_tma_gather=False,
    )
    time.sleep(0.01)
    copy_stream = torch.cuda.Stream()
    with torch.cuda.stream(copy_stream):
        ready_rows.copy_(host_ready, non_blocking=True)
    torch.cuda.synchronize()

    reference = torch.cat(
        [
            x[a_idx.long()].float() @ weights[0].float().mT
            for x, a_idx in zip(x_buffers, a_idx_buffers)
        ]
    ).to(output.dtype)
    torch.testing.assert_close(output, reference, atol=3e-2, rtol=1e-3)
