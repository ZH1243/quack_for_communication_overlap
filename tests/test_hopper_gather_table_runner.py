"""CPU tests for the Hopper gather-table runner's table construction."""

from pathlib import Path
from types import SimpleNamespace

import torch

from run.hopper_gather_table_gemm import (
    balanced_buffer_allocations,
    build_multi_buffer_work_table,
    build_work_table,
    multi_buffer_route_counts,
)
from run.hopper_stream_gather_table_gemm import (
    build_stream_metadata,
    proxy_command,
    raw_cuda_ipc_handle,
)


def test_n_group_major_table_preserves_grouped_gemm_values():
    """Global serpentine ordering preserves all output tiles, including tails."""
    counts = [5, 0, 3]
    kwargs = dict(
        output_dim=15,
        tile_m=2,
        tile_n=2,
        cluster_m=2,
        max_swizzle_size=2,
        device=torch.device("cpu"),
    )
    default, _, _ = build_work_table(counts, **kwargs)
    table, _, group_size = build_work_table(counts, n_group_major=True, **kwargs)
    expected_rows = []
    forward = [(0, 0, 4), (0, 4, 5), (2, 5, 8)]
    for group, n_base in enumerate((0, 2, 4, 6)):
        traversal = forward if group % 2 == 0 else forward[::-1]
        expected_rows.extend((*row, n_base) for row in traversal)
    assert table.tolist() == [list(row) for row in expected_rows]
    assert sorted(table.tolist()) == sorted(default.tolist())
    assert default[:, 0].tolist() == [0] * 8 + [2] * 4

    torch.manual_seed(0)
    source = torch.randn(6, 4)
    routes = torch.tensor([4, 0, 2, 4, 1, 5, 3, 0])
    a = source[routes]
    identity = torch.arange(8)
    weights = torch.randn(3, 4, 15)
    output = torch.full((8, 15), float("nan"))
    writes = torch.zeros((8, 15), dtype=torch.int32)
    # Model the cluster work-ID expansion and each CTA's clipped output slice.
    for expert, start, end, n_base in table.tolist():
        for n_in_group in range(group_size):
            col = (n_base + n_in_group) * 2
            col_end = min(col + 2, 15)
            for cta in range(2):
                lo, hi = min(start + cta * 2, end), min(start + (cta + 1) * 2, end)
                output[lo:hi, col:col_end] = (
                    a[identity[lo:hi]] @ weights[expert, :, col:col_end]
                )
                writes[lo:hi, col:col_end] += 1
    reference = torch.cat((source[routes[:5]] @ weights[0], source[routes[5:]] @ weights[2]))
    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(writes, torch.ones_like(writes))


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


def test_stream_proxy_uses_gated_gemm_width_instead_of_postactivation_width():
    args = SimpleNamespace(
        proxy_binary=Path("stream_gather_proxy"),
        device=0,
        routes_per_buffer=None,
        routes=19,
        experts=5,
        num_input_buffers=3,
        output_dim=128,
        tile_m=64,
        tile_n=128,
        cluster_m=2,
        max_swizzle_size=4,
        flush_entries=1,
        flush_interval_us=10,
        dma_kick_bytes=0,
        balanced_multi_buffer_gather=True,
        round_robin_m_clusters=False,
        flag_update_mode="memcpy",
        indexed_gather=False,
    )
    inputs = SimpleNamespace(
        work_table=torch.empty((7, 8), dtype=torch.int32),
        W=torch.empty((args.experts, 64, 2 * args.output_dim)),
        route_offsets=((0, 4, 8, 12, 16, 19),) * args.num_input_buffers,
    )

    command = proxy_command(args, inputs, "00" * 64, 0)

    output_dim_arg = command.index("--output-dim")
    assert command[output_dim_arg + 1] == "256"


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


def test_indexed_table_numerical_mapping():
    """Decode the table independently and compare GEMM values with route order."""
    from run.hopper_gather_table_gemm import build_indexed_work_table

    torch.manual_seed(42)
    counts = [5, 0, 9]
    indices = torch.tensor([7, 1, 7, 0, 9, 3, 8, 2, 6, 1, 0, 4, 9, 2], dtype=torch.int32)
    X, W = torch.randn(10, 3), torch.randn(3, 3, 11)
    table, packed, offsets, x = build_indexed_work_table(
        counts, indices, output_dim=11, tile_m=2, tile_n=2, cluster_m=2,
        max_swizzle_size=2, device=torch.device("cpu"),
    )
    torch.testing.assert_close(packed, indices)
    assert offsets == ((0, 5, 5, 14),)
    actual = torch.full((sum(counts), 11), float("nan"))
    for row in table.tolist():
        expert, cid_n, output_start, output_end, *tokens = row
        assert output_end - output_start == sum(token >= 0 for token in tokens)
        n_start, n_end = cid_n * 2, min((cid_n + x) * 2, 11)
        for m, token in enumerate(tokens):
            if token >= 0:
                actual[output_start + m, n_start:n_end] = X[token] @ W[expert, :, n_start:n_end]
    source_start = 0
    expected = torch.zeros_like(actual)
    for expert, count in enumerate(counts):
        expected[offsets[0][expert] : offsets[0][expert] + count] = (
            X[indices[source_start : source_start + count].long()] @ W[expert]
        )
        source_start += count
    torch.testing.assert_close(actual, expected)


def test_indexed_packed_output_across_partial_ctas():
    """Numerically check 300-row experts, empty experts, and per-CTA token offsets."""
    from run.hopper_gather_table_gemm import build_indexed_work_table

    torch.manual_seed(43)
    counts = [300, 0, 129, 1]
    indices = torch.randint(0, 37, (sum(counts),), dtype=torch.int32)
    X, W = torch.randn(37, 13), torch.randn(4, 13, 259)
    for cluster_m in (1, 2):
        table, _, offsets, _ = build_indexed_work_table(
            counts, indices, output_dim=259, tile_m=128, tile_n=128,
            cluster_m=cluster_m, max_swizzle_size=1, device=torch.device("cpu"),
        )
        actual = torch.full((sum(counts), 259), float("nan"))
        for expert, cid_n, start, end, *tokens in table.tolist():
            n_start, n_end = cid_n * 128, min((cid_n + 1) * 128, 259)
            for cta in range(cluster_m):
                begin = min(start + cta * 128, end)
                stop = min(begin + 128, end)
                if begin == stop:
                    continue
                local_start = begin - start
                token_ids = torch.tensor(
                    tokens[local_start : local_start + stop - begin], dtype=torch.long
                )
                actual[begin:stop, n_start:n_end] = X[token_ids] @ W[expert, :, n_start:n_end]
        expected = torch.cat([
            X[indices[offsets[0][expert] : offsets[0][expert + 1]].long()] @ W[expert]
            for expert in range(len(counts))
        ])
        torch.testing.assert_close(actual, expected)


def test_indexed_and_legacy_modes_use_identical_inputs():
    """The flag changes route encoding, preserving each tile's tokens and GEMM values."""
    from run.hopper_gather_table_gemm import make_arg_parser, prepare_inputs
    from run.hopper_stream_gather_table_gemm import prepare_inputs as prepare_stream_inputs

    args = make_arg_parser(include_down_projection=True).parse_args([])
    args.tokens, args.routes, args.experts = 17, 29, 3
    args.hidden, args.output_dim = 8, 11
    args.tile_m, args.cluster_m, args.tile_n = 2, 2, 2
    args.max_swizzle_size = 2
    args.activation, args.down_projection = "relu", True
    device = torch.device("cpu")
    for replacement in (False, True):
        args.routing_with_replacement = replacement
        args.indexed_gather = False
        torch.manual_seed(42)
        legacy = prepare_inputs(args, device)
        args.indexed_gather = True
        torch.manual_seed(42)
        indexed = prepare_inputs(args, device)
        torch.manual_seed(42)
        streamed, ready_rows, backing = prepare_stream_inputs(args, device)
        for name in ("X", "W", "W_down", "A_idx", "cu_seqlens_m", "up_output"):
            torch.testing.assert_close(
                getattr(streamed, name), getattr(indexed, name), atol=0, rtol=0
            )
        assert streamed.route_offsets == indexed.route_offsets
        assert streamed.output_segments == indexed.output_segments
        assert streamed.work_table.shape == indexed.work_table.shape
        torch.testing.assert_close(ready_rows, torch.zeros_like(ready_rows), atol=0, rtol=0)
        torch.testing.assert_close(
            backing[streamed.work_table.numel() + 1 :], legacy.A_idx, atol=0, rtol=0
        )
        torch.testing.assert_close(indexed.X, legacy.X, atol=0, rtol=0)
        torch.testing.assert_close(indexed.W, legacy.W, atol=0, rtol=0)
        torch.testing.assert_close(indexed.W_down, legacy.W_down, atol=0, rtol=0)

        for expert, route_start, route_end, cid_n in legacy.work_table.tolist():
            local_start = route_start - legacy.route_offsets[0][expert]
            output_start = indexed.route_offsets[0][expert] + local_start
            matches = (indexed.work_table[:, 0] == expert) & (
                indexed.work_table[:, 1] == cid_n
            ) & (indexed.work_table[:, 2] == route_start)
            row = indexed.work_table[matches].squeeze(0)
            count = route_end - route_start
            expected_indices = legacy.A_idx[route_start:route_end]
            torch.testing.assert_close(row[4 : 4 + count], expected_indices, atol=0, rtol=0)
            torch.testing.assert_close(
                row[4 + count:], torch.full_like(row[4 + count:], -1), atol=0, rtol=0
            )
            n_start = cid_n * args.tile_n
            n_end = min(n_start + legacy.work_group_size * args.tile_n, args.output_dim)
            actual = indexed.X[row[4 : 4 + count].long()].float() @ indexed.W[
                expert, :, n_start:n_end
            ].float()
            reference = legacy.X[expected_indices.long()].float() @ legacy.W[
                expert, :, n_start:n_end
            ].float()
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            streamed_result = streamed.X[
                streamed.A_idx[output_start : output_start + count].long()
            ].float() @ streamed.W[expert, :, n_start:n_end].float()
            torch.testing.assert_close(streamed_result, reference, atol=0, rtol=0)
