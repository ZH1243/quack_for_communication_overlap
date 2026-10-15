"""CPU numerical tests; runnable without CUDA/CuTe with pytest --noconftest."""

import pytest
import torch

from run.wave_ready import build_wave_ready_prefix


@pytest.mark.parametrize("x", [1, 2])
@pytest.mark.parametrize(
    "order,rows_per_wave,expected",
    [
        # Expert borrowing: group 1 is visited before expert 1 finishes group 0.
        ([(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (0, 2)], 2, [0, 2, 3]),
        # An early complete higher group cannot jump over an incomplete group 0.
        ([(0, 1), (1, 1), (0, 0), (1, 0), (0, 2), (1, 2)], 2, [0, 2, 3]),
        # Partial last wave and a single wave both require explicit final publication.
        ([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)], 4, [1, 3]),
        ([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)], 6, [3]),
    ],
)
def test_ready_prefix_matches_accumulated_scatter_values(x, order, rows_per_wave, expected):
    # Two experts, three routes each. cluster_M=2, tile_M=2: a predicated row
    # exists in every cluster. Duplicate destinations span experts and buffers.
    rows = [[e, e * 3, (e + 1) * 3, g * x] for e, g in order]
    prefixes = build_wave_ready_prefix(
        rows, group_size=x, num_groups=3, wave_work_ids=rows_per_wave * x
    )
    assert prefixes == expected
    a = torch.arange(1, 19, dtype=torch.float32).reshape(6, 3)
    weights = torch.arange(1, 2 * 3 * 6 * x + 1, dtype=torch.float32).reshape(2, 3, 6 * x)
    destinations = torch.tensor([0, 2, 0, 2, 1, 0])
    reference = torch.zeros(4, 6 * x)
    for e in range(2):
        reference.index_add_(
            0, destinations[e * 3 : (e + 1) * 3], a[e * 3 : (e + 1) * 3] @ weights[e]
        )

    buffers = [torch.zeros(2, 6 * x), torch.zeros(2, 6 * x)]
    snapshots = []
    for r, (e, start, end, n_base) in enumerate(rows):
        for n in range(n_base, n_base + x):
            lo, hi = n * 2, (n + 1) * 2
            for cta in range(2):
                for route in range(start + cta * 2, min(start + (cta + 1) * 2, end)):
                    dst = destinations[route].item()
                    buffers[dst // 2][dst % 2, lo:hi] += a[route] @ weights[e, :, lo:hi]
        if (r + 1) % rows_per_wave == 0 or r == len(rows) - 1:
            ready = prefixes[r // rows_per_wave]
            actual = torch.cat(buffers)
            torch.testing.assert_close(actual[:, : ready * x * 2], reference[:, : ready * x * 2])
            # Positive contributions ensure an unfinished next group differs.
            if ready < 3:
                assert not torch.equal(
                    actual[:, ready * x * 2 : (ready + 1) * x * 2],
                    reference[:, ready * x * 2 : (ready + 1) * x * 2],
                )
            snapshots.append((ready, actual[:, : ready * x * 2].clone()))
    torch.testing.assert_close(torch.cat(buffers), reference)
    for ready, snapshot in snapshots:
        torch.testing.assert_close(torch.cat(buffers)[:, : ready * x * 2], snapshot)
