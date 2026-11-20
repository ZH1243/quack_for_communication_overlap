"""Blackwell work-table builder, retaining the existing tests/benchmark API."""

if __package__:
    from .gemm_tables import make_table as _make_table
else:
    from gemm_tables import make_table as _make_table


def make_table(counts, *, tile_m=128, cluster_m=2, tile_n=128, n=384, group_size=1):
    return _make_table(
        counts, device_capacity=10, tile_m=tile_m, cluster_m=cluster_m,
        tile_n=tile_n, n=n, group_size=group_size,
    )
