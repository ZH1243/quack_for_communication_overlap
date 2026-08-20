"""CPU tests for the Hopper gather-table runner's table construction."""

import torch

from run.hopper_gather_table_gemm import (
    balanced_buffer_allocations,
    build_multi_buffer_work_table,
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
