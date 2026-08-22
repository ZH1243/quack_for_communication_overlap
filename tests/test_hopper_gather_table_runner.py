"""CPU tests for the Hopper gather-table runner's table construction."""

import torch

from run.hopper_gather_table_gemm import (
    balanced_buffer_allocations,
    build_multi_buffer_work_table,
    multi_buffer_route_counts,
)
from run.hopper_stream_gather_table_gemm import (
    build_stream_metadata,
    raw_cuda_ipc_handle,
)


def test_balanced_buffer_allocations_redistributes_exhausted_buffer():
    assert balanced_buffer_allocations([10, 300, 400], 512) == [10, 251, 251]


def test_balanced_multi_buffer_table_uses_every_buffer_per_cluster():
    table, offsets, output_segments, group_size = build_multi_buffer_work_table(
        [[300], [400], [200]],
        output_dim=256,
        tile_m=256,
        tile_n=256,
        cluster_m=2,
        max_swizzle_size=8,
        device=torch.device("cpu"),
        balance_buffers=True,
    )

    assert table.tolist() == [
        [0, 0, 0, 172, 0, 170, 0, 170],
        [0, 0, 172, 300, 170, 400, 170, 200],
    ]
    assert offsets == ((0, 300), (0, 400), (0, 200))
    assert output_segments == (
        (0, 0, 0, 172),
        (0, 1, 0, 170),
        (0, 2, 0, 170),
        (0, 0, 172, 300),
        (0, 1, 170, 400),
        (0, 2, 170, 200),
    )
    assert group_size == 1


def test_default_multi_buffer_table_remains_buffer_major():
    table, _, _, _ = build_multi_buffer_work_table(
        [[300], [400], [200]],
        output_dim=256,
        tile_m=256,
        tile_n=256,
        cluster_m=2,
        max_swizzle_size=8,
        device=torch.device("cpu"),
    )

    assert table.tolist() == [
        [0, 0, 0, 300, 0, 212, 0, 0],
        [0, 0, 300, 300, 212, 400, 0, 200],
    ]


def test_balanced_segments_are_not_duplicated_across_n_groups():
    table, _, output_segments, group_size = build_multi_buffer_work_table(
        [[300], [400], [200]],
        output_dim=4096,
        tile_m=256,
        tile_n=256,
        cluster_m=2,
        max_swizzle_size=8,
        device=torch.device("cpu"),
        balance_buffers=True,
    )

    assert group_size == 8
    assert table[:, 1].tolist() == [0, 0, 8, 8]
    assert table[2, 2:].tolist() == table[1, 2:].tolist()
    assert table[3, 2:].tolist() == table[0, 2:].tolist()
    assert sum(end - start for _, _, start, end in output_segments) == 900


def test_round_robin_table_interleaves_complete_m_cluster_bundles():
    table, offsets, output_segments, group_size = build_multi_buffer_work_table(
        [[5, 2, 9], [1, 5, 0]],
        output_dim=1024,
        tile_m=2,
        tile_n=256,
        cluster_m=2,
        max_swizzle_size=2,
        device=torch.device("cpu"),
        round_robin_m_clusters=True,
    )

    assert table.tolist() == [
        [0, 0, 0, 4, 0, 0],
        [0, 2, 0, 4, 0, 0],
        [1, 0, 5, 7, 1, 3],
        [1, 2, 5, 7, 1, 3],
        [2, 0, 7, 11, 6, 6],
        [2, 2, 7, 11, 6, 6],
        [0, 0, 4, 5, 0, 1],
        [0, 2, 4, 5, 0, 1],
        [1, 0, 7, 7, 3, 6],
        [1, 2, 7, 7, 3, 6],
        [2, 0, 11, 15, 6, 6],
        [2, 2, 11, 15, 6, 6],
        [2, 0, 15, 16, 6, 6],
        [2, 2, 15, 16, 6, 6],
    ]
    assert offsets == ((0, 5, 7, 16), (0, 1, 6, 6))
    assert output_segments == (
        (0, 0, 0, 4),
        (0, 0, 4, 5),
        (0, 1, 0, 1),
        (1, 0, 5, 7),
        (1, 1, 1, 3),
        (1, 1, 3, 6),
        (2, 0, 7, 11),
        (2, 0, 11, 15),
        (2, 0, 15, 16),
    )
    assert group_size == 2


def test_round_robin_order_is_independent_of_balanced_buffer_allocation():
    kwargs = dict(
        output_dim=1024,
        tile_m=2,
        tile_n=256,
        cluster_m=2,
        max_swizzle_size=2,
        device=torch.device("cpu"),
        balance_buffers=True,
    )
    default_table, offsets, output_segments, _ = build_multi_buffer_work_table(
        [[5, 2], [1, 5]], **kwargs
    )
    round_robin_table, rr_offsets, rr_output_segments, _ = build_multi_buffer_work_table(
        [[5, 2], [1, 5]], **kwargs, round_robin_m_clusters=True
    )

    assert rr_offsets == offsets
    assert rr_output_segments == output_segments
    assert sorted(map(tuple, round_robin_table.tolist())) == sorted(
        map(tuple, default_table.tolist())
    )
    assert round_robin_table[:, :2].tolist() == [
        [0, 0],
        [0, 2],
        [1, 0],
        [1, 2],
        [0, 0],
        [0, 2],
        [1, 0],
        [1, 2],
    ]


def test_stream_metadata_matches_full_table_without_constructing_rows():
    counts = multi_buffer_route_counts(routes_per_buffer=19, experts=5, num_buffers=8)
    kwargs = dict(
        output_dim=2048,
        tile_m=3,
        tile_n=128,
        cluster_m=2,
        max_swizzle_size=4,
        balance_buffers=True,
    )
    offsets, output_segments, group_size, table_rows = build_stream_metadata(
        counts, **kwargs
    )
    table, expected_offsets, expected_segments, expected_group_size = (
        build_multi_buffer_work_table(
            counts,
            device=torch.device("cpu"),
            **kwargs,
        )
    )

    assert offsets == expected_offsets
    assert output_segments == expected_segments
    assert group_size == expected_group_size
    assert table_rows == table.shape[0]


def test_cuda_ipc_handle_decode_accepts_legacy_and_versioned_cuda_malloc():
    raw = bytes(range(64))
    assert raw_cuda_ipc_handle(raw) == raw
    assert raw_cuda_ipc_handle(bytes([3]) + b"c" + raw) == raw


def test_cuda_ipc_handle_validation_rejects_expandable_segments():
    try:
        raw_cuda_ipc_handle(bytes([3]) + b"e" + bytes(64))
    except RuntimeError as error:
        assert "expandable segments" in str(error)
    else:
        raise AssertionError("expandable-segment IPC handle was accepted")
