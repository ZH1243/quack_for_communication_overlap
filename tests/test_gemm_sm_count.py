"""Persistent CTA budgets for the proxy up path on SM90 and SM100/SM103.

Remote: CUDA_VISIBLE_DEVICES=0 pytest -q tests/test_gemm_sm_count.py -x
Every case checks numerical values and the scheduler argument actually launched.
"""

import importlib
import math

import pytest
import torch
import torch.nn.functional as F

from quack.cute_dsl_utils import get_max_active_clusters
from quack.gemm_config import GemmConfig, cta_tile_shape_m
from quack.gemm_interface import gemm_act


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10),
    reason="requires Hopper or Blackwell datacenter CUDA",
)


@pytest.fixture
def launched_schedulers(monkeypatch):
    """Observe real host launch arguments without replacing any kernel work."""
    plain = importlib.import_module("quack.gemm")
    host = importlib.import_module("quack.gemm_runtime.host")
    utils = importlib.import_module("quack.gemm_tvm_ffi_utils")
    original_make = utils.make_scheduler_args
    original_launch = utils.launch_gemm
    budgets, launches = {}, []

    def make_scheduler(capacity, *args, **kwargs):
        scheduler = original_make(capacity, *args, **kwargs)
        # Retain the object too: Python may otherwise recycle id() across calls.
        budgets[id(scheduler)] = (scheduler, capacity)
        return scheduler

    def launch(plan, a, b, d, c, epi_args, scheduler, *args, **kwargs):
        assert id(scheduler) in budgets, "launch bypassed the configured scheduler budget"
        assert budgets[id(scheduler)][0] is scheduler
        launches.append((plan, budgets[id(scheduler)][1]))
        return original_launch(plan, a, b, d, c, epi_args, scheduler, *args, **kwargs)

    for module in (plain, host, utils):
        monkeypatch.setattr(module, "make_scheduler_args", make_scheduler)
    for module in (plain, host):
        monkeypatch.setattr(module, "launch_gemm", launch)
    return launches


def inputs(cluster_m, architecture, projection_n=512):
    """Three sources, one empty source/expert, repeated indices and ragged tails."""
    tile_m, tile_n, k = 128, 128, 64
    cluster_rows = cta_tile_shape_m(tile_m, cluster_m, architecture) * cluster_m
    counts = [[cluster_rows + 3, 0, 9], [0, 0, 0], [11, 0, cluster_rows // 2 + 1]]
    offsets = [[0] for _ in counts]
    for source, values in enumerate(counts):
        for count in values:
            offsets[source].append(offsets[source][-1] + count)
    capacity = max(map(sum, counts)) + 7
    xs = tuple(torch.randn(281, k, device="cuda", dtype=torch.bfloat16) for _ in counts)
    indices, host_indices = [], []
    for source, values in enumerate(counts):
        index = torch.full((capacity,), -987654, dtype=torch.int32)
        index[:sum(values)] = (torch.arange(sum(values)) * 7 + source * 13) % xs[source].shape[0]
        host_indices.append(index)
        indices.append(index.cuda())
    descriptors = []
    for expert in range(len(counts[0])):
        consumed = [0] * len(counts)
        while any(consumed[source] < values[expert] for source, values in enumerate(counts)):
            room, ranges = cluster_rows, []
            for source, values in enumerate(counts):
                start = offsets[source][expert] + consumed[source]
                take = min(room, values[expert] - consumed[source])
                ranges.extend((start, start + take))
                consumed[source] += take
                room -= take
            for group in range(projection_n // tile_n):
                descriptors.append([expert, group, *ranges])
    table = torch.tensor(descriptors, device="cuda", dtype=torch.int32)
    # concat-B keeps gate/up matrices contiguous in physical storage.
    weights = torch.randn(3, projection_n, k, device="cuda", dtype=xs[0].dtype) / math.sqrt(k)
    return xs, tuple(indices), host_indices, offsets, weights, table


def reference(host_inputs, host_indices, offsets, weights, activation):
    matrices = weights.cpu().float()
    parts = []
    for expert in range(weights.shape[0]):
        for source, x in enumerate(host_inputs):
            begin, end = offsets[source][expert:expert + 2]
            value = x[host_indices[source][begin:end].long()].float() @ matrices[expert].T
            if activation == "relu":
                value = F.relu(value)
            elif activation == "swiglu":
                gate, up = value.chunk(2, dim=-1)
                value = F.silu(gate) * up
            parts.append(value)
    return torch.cat(parts)


def check_output(output, expected):
    actual = output[:len(expected)].float().cpu()
    rounding_error = (expected.to(output.dtype).float() - expected).abs().max().item()
    torch.testing.assert_close(actual, expected, atol=max(0.05, 2 * rounding_error), rtol=0.03)
    assert torch.isnan(output[len(expected):]).all().item(), "wrote into padded route capacity"


@torch.inference_mode()
@pytest.mark.parametrize("cluster_m", [1, 2], ids=["cluster1", "cluster2"])
@pytest.mark.parametrize("activation", [None, "relu", "swiglu"], ids=["plain", "relu", "swiglu"])
def test_streamed_up_sm_count_numerical_cache_and_budget(
    activation, cluster_m, launched_schedulers,
):
    torch.manual_seed(7153)
    architecture = torch.cuda.get_device_capability()[0]
    device = torch.cuda.current_device()
    xs, indices, host_indices, offsets, weights, table = inputs(cluster_m, architecture)
    projection_n = weights.shape[1]
    output_n = projection_n // 2 if activation == "swiglu" else projection_n
    output = torch.empty(sum(index.numel() for index in indices), output_n,
                         dtype=xs[0].dtype, device="cuda")
    ready = torch.full((1,), len(table), dtype=torch.int32, device="cuda")
    config = GemmConfig(tile_m=128, tile_n=128, cluster_m=cluster_m, cluster_n=1,
                        max_swizzle_size=1, device_capacity=architecture, pingpong=False,
                        is_dynamic_persistent=False, use_tma_gather=False)
    hardware_clusters = get_max_active_clusters(cluster_m, device_id=device)
    hardware_sms = torch.cuda.get_device_properties(device).multi_processor_count
    oversized = (hardware_sms // cluster_m + 32) * cluster_m
    caps = [None, cluster_m, 2 * cluster_m, cluster_m, oversized, None]
    host_inputs = tuple(x.cpu().pin_memory() for x in xs)
    original_inputs = tuple(x.clone() for x in host_inputs)
    host_table = table.cpu().pin_memory()
    parts = [(start, min(start + 3, len(table))) for start in range(0, len(table), 3)]
    host_ready = [torch.tensor([end], dtype=torch.int32, pin_memory=True) for _, end in parts]

    def launch(cap):
        previous = len(launched_schedulers)
        if activation is None:
            importlib.import_module("quack.gemm").gemm(
                xs, weights, output, C=None, tile_count_semaphore=None,
                tile_M=128, tile_N=128, cluster_M=cluster_m, cluster_N=1,
                persistent=True, pingpong=False, is_dynamic_persistent=False,
                max_swizzle_size=1, A_idx=indices, gather_work_table=table,
                gather_work_table_ready=ready, multi_buffer_gather=True, sm_count=cap,
            )
        else:
            preact, result = gemm_act(
                xs, weights.transpose(1, 2), activation=activation, A_idx=indices,
                gather_work_table=table, gather_work_table_ready=ready,
                multi_buffer_gather=True, postact_out=output, store_preact=False,
                concat_layout=("B",) if activation == "swiglu" else None,
                tuned=False, config=config, sm_count=cap,
            )
            assert preact is None and result is output
        assert len(launched_schedulers) == previous + 1
        plan, actual_budget = launched_schedulers[-1]
        expected_budget = (
            hardware_clusters if cap is None else min(hardware_clusters, cap // cluster_m)
        )
        assert actual_budget == expected_budget
        assert plan.max_active_clusters == expected_budget
        return plan

    plans = {}
    expected = reference(host_inputs, host_indices, offsets, weights, activation)
    baseline = None
    # Compile every requested budget with fully ready inputs before attempting
    # streamed publication. This also exercises uncapped -> capped -> uncapped.
    for cap in caps:
        output.fill_(float("nan"))
        plan = launch(cap)
        torch.cuda.synchronize()
        check_output(output, expected)
        if baseline is None:
            baseline = output[:len(expected)].clone()
        else:
            torch.testing.assert_close(output[:len(expected)], baseline, atol=0, rtol=0)
        if cap in plans:
            assert plan is plans[cap], "repeated cap missed the launch-plan cache"
        plans[cap] = plan
    assert len({id(plan.compiled_fn) for plan in plans.values()}) == 1

    producer, compute = torch.cuda.Stream(), torch.cuda.Stream()
    for iteration, cap in enumerate(caps):
        for host, original in zip(host_inputs, original_inputs):
            host.copy_(original * (1 + iteration / 8))
        expected = reference(host_inputs, host_indices, offsets, weights, activation)
        for x in xs:
            x.zero_()
        table.fill_(-1)
        ready.zero_()
        output.fill_(float("nan"))
        producer.wait_stream(torch.cuda.current_stream())
        compute.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(compute):
            assert launch(cap) is plans[cap]
        # H2D copy-engine publication needs no spare SM, including uncapped runs.
        with torch.cuda.stream(producer):
            for x, host in zip(xs, host_inputs):
                x.copy_(host, non_blocking=True)
            for (start, end), count in zip(parts, host_ready):
                table[start:end].copy_(host_table[start:end], non_blocking=True)
                ready.copy_(count, non_blocking=True)
        torch.cuda.current_stream().wait_stream(producer)
        torch.cuda.current_stream().wait_stream(compute)
        torch.cuda.synchronize()
        check_output(output, expected)

    # Validation must remain active on the now-warm host plan/cache paths.
    for invalid in (0, -cluster_m, 1 if cluster_m == 2 else -1):
        with pytest.raises(ValueError, match="positive.*divisible"):
            launch(invalid)
    # The failing requests did not corrupt previously valid launch plans/results.
    output.fill_(float("nan"))
    assert launch(cluster_m) is plans[cluster_m]
    torch.cuda.synchronize()
    check_output(output, expected)
