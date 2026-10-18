"""Synthetic expert routing with guaranteed coverage of each input token."""

import torch


def covered_token_routes(tokens, counts, *, replacement, device):
    """Return expert-contiguous int32 indices, preserving each expert's quota.

    Partition a random permutation of all tokens among experts first, then fill
    their remaining slots. Without replacement, extras exclude that expert's
    guaranteed tokens (but may visit tokens assigned to other experts).
    """
    routes = sum(counts)
    if tokens <= 0 or not counts or any(count < 0 for count in counts):
        raise ValueError("routing requires positive tokens and nonnegative expert quotas")
    if routes < tokens:
        raise ValueError("routes per buffer must be at least tokens per buffer for full coverage")
    if not replacement and max(counts) > tokens:
        raise ValueError("an expert quota exceeds tokens without replacement")

    # Proportional quotas also handle uneven counts and zero-work experts.
    guaranteed = [tokens * count // routes for count in counts]
    remaining = tokens - sum(guaranteed)
    for expert, count in enumerate(counts):
        if remaining and guaranteed[expert] < count:
            guaranteed[expert] += 1
            remaining -= 1

    permutation = torch.randperm(tokens, dtype=torch.int32, device=device)
    indices = []
    offset = 0
    for count, covered in zip(counts, guaranteed):
        mandatory = permutation[offset:offset + covered]
        offset += covered
        extra_count = count - covered
        if replacement:
            extras = torch.randint(tokens, (extra_count,), dtype=torch.int32, device=device)
        else:
            candidates = torch.randperm(tokens, dtype=torch.int32, device=device)
            available = torch.ones(tokens, dtype=torch.bool, device=device)
            available[mandatory.long()] = False
            extras = candidates[available[candidates.long()]][:extra_count]
        selected = torch.cat((mandatory, extras))
        indices.append(selected[torch.randperm(count, device=device)])
    return torch.cat(indices)
