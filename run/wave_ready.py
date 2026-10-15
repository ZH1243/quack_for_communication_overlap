"""Host metadata for publishing completed column prefixes of a static work table."""


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
    if min(last_wave) < 0:
        raise ValueError("work table must cover every column group")
    num_waves = (len(rows) * group_size + wave_work_ids - 1) // wave_work_ids
    prefix = []
    ready = 0
    for wave in range(num_waves):
        while ready < num_groups and last_wave[ready] <= wave:
            ready += 1
        prefix.append(ready)
    return prefix
