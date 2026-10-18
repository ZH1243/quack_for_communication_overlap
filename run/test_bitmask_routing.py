"""CPU routing/reference tests: python -m unittest run.test_bitmask_routing."""

import unittest

import torch

from run.bitmask_exchange import destination_indices, make_router_metadata
from run.token_routing import covered_token_routes


class BitmaskRoutingTest(unittest.TestCase):
    def test_bit_mapping_and_ignored_bits(self):
        masks = torch.tensor([0, 1, 2, 4, 8, 15, 240, 255], dtype=torch.uint8)
        for world in (2, 3, 4, 8):
            for source in range(world):
                selected = destination_indices(masks, source, world)
                for destination, actual in enumerate(selected):
                    expected = [i for i, mask in enumerate(masks.tolist())
                                if any((mask & (1 << bit)) and
                                       (source + bit + 1) % world == destination
                                       for bit in range(world))]
                    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.int64))

    def test_probability_endpoints_and_seed(self):
        for world in (2, 4, 8):
            for probability, value in ((0.0, 0), (1.0, (1 << world) - 1)):
                actual = make_router_metadata(19, world, probability, 42, "cpu")
                torch.testing.assert_close(actual, torch.full((19,), value, dtype=torch.uint8))
        a = make_router_metadata(257, 4, 0.25, 123, "cpu")
        b = make_router_metadata(257, 4, 0.25, 123, "cpu")
        torch.testing.assert_close(a, b)
        self.assertTrue(bool((a < 16).all()))

    def test_variable_dispatch_and_group_return_values(self):
        world, tokens, hidden, tile = 4, 19, 8, 2
        masks = [make_router_metadata(tokens, world, .25, s, "cpu") for s in range(world)]
        selected = [destination_indices(mask, s, world) for s, mask in enumerate(masks)]
        sources = [torch.arange(tokens * hidden, dtype=torch.float32).reshape(tokens, hidden)
                   + s * 1000 for s in range(world)]
        counts = [[index.numel() for index in row] for row in selected]
        # The receive sizes use outgoing counts, which need not match incoming counts.
        self.assertTrue(any(counts[s][d] != counts[d][s]
                            for s in range(world) for d in range(world)))
        for source in range(world):
            for compute in range(world):
                indices = selected[source][compute]
                x = sources[source][indices]
                count = len(indices)
                if count:
                    quotas = [count, count]
                    routes = covered_token_routes(count, quotas, replacement=False, device="cpu")
                    visits = torch.bincount(routes.long(), minlength=count)
                    torch.testing.assert_close(visits, torch.full((count,), 2))
                # Distinct contributions make wrong destination/source mappings visible.
                output = x * (compute + 1)
                physical = output.reshape(count, hidden // tile, tile).permute(1, 0, 2).contiguous()
                for group_tiles in (1, 2):
                    returned = torch.empty_like(physical).view(-1)
                    flat = physical.view(-1)
                    group_elements = count * tile * group_tiles
                    for group in range(hidden // tile // group_tiles):
                        begin = group * group_elements
                        returned[begin:begin + group_elements].copy_(flat[begin:begin + group_elements])
                    logical = returned.view(hidden // tile, count, tile).permute(1, 0, 2)
                    reference = torch.stack([sources[source][i] * (compute + 1)
                                             for i in indices.tolist()]) if count else x
                    torch.testing.assert_close(logical.reshape(count, hidden), reference)


if __name__ == "__main__":
    unittest.main()
