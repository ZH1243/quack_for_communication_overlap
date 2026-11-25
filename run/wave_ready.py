"""Host metadata for publishing completed column prefixes of a static work table."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def build_expert_wave_schedule(
    counts: list[int], *, cluster_rows: int, num_groups: int, group_size: int,
    wave_work_ids: int, with_ready_prefix: bool = True,
) -> tuple[torch.Tensor, list[int] | None]:
    """Pack expert-major row ranges into a CPU int32 table in wave execution order."""
    import torch

    if min(cluster_rows, num_groups, group_size, wave_work_ids) <= 0 or wave_work_ids % group_size:
        raise ValueError("expert waves require positive geometry and whole column groups per wave")
    if any(count < 0 for count in counts):
        raise ValueError("expert row counts must be nonnegative")
    if not any(counts):
        if with_ready_prefix:
            raise ValueError("ready publication requires nonempty work")
        return torch.empty(0, dtype=torch.int32, device="cpu"), None
    if max(sum(counts), len(counts) - 1, (num_groups - 1) * group_size) > 2**31 - 1:
        raise RuntimeError("expert wave descriptors must fit in int32")

    # Each expert has a contiguous stream of descriptors. Odd column groups
    # visit the same row blocks in reverse order, including the clipped tail.
    column_bases = torch.arange(num_groups, dtype=torch.int64, device="cpu") * group_size
    blocks = [(count + cluster_rows - 1) // cluster_rows for count in counts]
    streams = []
    offset = 0
    for expert, (count, num_blocks) in enumerate(zip(counts, blocks)):
        rows = torch.empty((num_groups, num_blocks, 4), dtype=torch.int32, device="cpu")
        starts = offset + torch.arange(num_blocks, dtype=torch.int64, device="cpu") * cluster_rows
        ends = (starts + cluster_rows).clamp_max(offset + count)
        ranges = torch.stack((starts, ends), dim=1)
        rows[:, :, 0] = expert
        rows[::2, :, 1:3] = ranges
        rows[1::2, :, 1:3] = ranges.flip((0,))
        rows[:, :, 3] = column_bases[:, None]
        streams.append(rows.view(-1, 4))
        offset += count

    rows_per_wave = wave_work_ids // group_size
    spans = []
    cursors = [0] * len(counts)
    last_wave = [-1] * num_groups
    written = 0
    for group in range(num_groups):
        experts = range(len(counts))
        for expert in (reversed(experts) if group % 2 else experts):
            begin = cursors[expert]
            num_blocks = blocks[expert]
            needed = (group + 1) * num_blocks - begin
            if needed <= 0:
                continue
            # Borrow from this expert's following groups to reach the next
            # wave boundary. Another expert may fill the wave upon exhaustion.
            borrow_rows = -(written + needed) % rows_per_wave
            end = min(begin + needed + borrow_rows, len(streams[expert]))
            spans.append(streams[expert][begin:end])
            if with_ready_prefix:
                # A borrowed span may cross several column groups. Its final
                # row in each group determines the latest wave seen so far.
                for column_group in range(begin // num_blocks, (end - 1) // num_blocks + 1):
                    stop = min(end, (column_group + 1) * num_blocks)
                    last_wave[column_group] = (written + stop - begin - 1) // rows_per_wave
            written += end - begin
            cursors[expert] = end

    num_waves = (written + rows_per_wave - 1) // rows_per_wave
    prefix = _ready_prefix(last_wave, num_waves) if with_ready_prefix else None
    return torch.cat(spans), prefix


def build_wave_ready_prefix(
    rows: list[list[int]], *, group_size: int, num_groups: int, wave_work_ids: int
) -> list[int]:
    """Return ready group counts after each wave of the FINAL four-column table.

    Each row expands to ``group_size`` cluster work IDs. Column groups have
    ``group_size * tile_N`` columns. Include all experts and all scheduled rows,
    even those whose final CTA is fully predicated. The launch must use the same
    persistent cluster count as ``wave_work_ids``.
    """
    if not rows or min(group_size, num_groups, wave_work_ids) <= 0:
        raise ValueError("ready publication requires nonempty work and positive geometry")
    if wave_work_ids % group_size:
        raise ValueError("expert wave size must be divisible by group_size")
    last_wave = [-1] * num_groups
    for r, row in enumerate(rows):
        if len(row) != 4 or row[3] % group_size or not 0 <= row[3] // group_size < num_groups:
            raise ValueError("expected a four-column work table with aligned N-group bases")
        group = row[3] // group_size
        last_wave[group] = ((r + 1) * group_size - 1) // wave_work_ids
    num_waves = (len(rows) * group_size + wave_work_ids - 1) // wave_work_ids
    return _ready_prefix(last_wave, num_waves)


def _ready_prefix(last_wave: list[int], num_waves: int) -> list[int]:
    if min(last_wave) < 0:
        raise ValueError("work table must cover every column group")
    prefix = []
    ready = 0
    for wave in range(num_waves):
        while ready < len(last_wave) and last_wave[ready] <= wave:
            ready += 1
        prefix.append(ready)
    return prefix
