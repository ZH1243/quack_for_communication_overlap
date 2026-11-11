"""CPU metadata builders for the Blackwell communication GEMM tests/benchmarks.

``tile_m`` is the native MMA configuration, not the per-CTA row count. Keep
this distinction when translating the C++ proxy's table geometry to SM100.
"""

from quack.gemm_config import cta_tile_shape_m


def make_table(counts, *, tile_m=128, cluster_m=2, tile_n=128, n=384, group_size=1):
    """Return (Python descriptor rows, expert offsets) for one or many buffers.

    counts[buffer][expert] describes the valid prefix of each index buffer.
    Capacity padding is deliberately excluded. Groups are ordered by N, so a
    completed group can be consumed before later groups finish.
    """
    if not counts or not counts[0] or any(len(c) != len(counts[0]) for c in counts):
        raise ValueError("counts must be a nonempty rectangular buffer/expert matrix")
    if any(c < 0 for cs in counts for c in cs):
        raise ValueError("route counts must be nonnegative")
    ntiles = (n + tile_n - 1) // tile_n
    if group_size <= 0 or ntiles % group_size:
        raise ValueError("N tile count must be divisible by group_size")
    cluster_rows = cta_tile_shape_m(tile_m, cluster_m, 10) * cluster_m
    offsets = []
    for cs in counts:
        off = [0]
        for c in cs:
            off.append(off[-1] + c)
        offsets.append(off)
    descriptors = []
    for expert in range(len(counts[0])):
        consumed = [0] * len(counts)
        while any(consumed[b] < counts[b][expert] for b in range(len(counts))):
            room = cluster_rows
            ranges = []
            for b in range(len(counts)):
                start = offsets[b][expert] + consumed[b]
                take = min(room, counts[b][expert] - consumed[b])
                ranges.extend((start, start + take))
                consumed[b] += take
                room -= take
            descriptors.append((expert, ranges))
    rows = []
    for ng in range(0, ntiles, group_size):
        for expert, ranges in descriptors:
            rows.append(
                [expert, ranges[0], ranges[1], ng] if len(counts) == 1
                else [expert, ng, *ranges]
            )
    return rows, offsets
