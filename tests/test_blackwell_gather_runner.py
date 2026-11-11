"""Numerical coverage of Blackwell indexed-gather runner launches and graph replay."""

import pytest
import torch

from run import blackwell_gather_gemm as runner


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires Blackwell SM10x",
)


@pytest.mark.parametrize("tma_gather", [False, True], ids=["cpasync", "tma"])
@pytest.mark.parametrize(
    "activation,down,dynamic",
    [(None, False, False), ("gelu_tanh_approx", False, True), ("swiglu", True, True)],
    ids=["plain-static", "gelu-dynamic", "swiglu-down-dynamic"],
)
@torch.inference_mode()
def test_blackwell_gather_runner(tma_gather, activation, down, dynamic):
    argv = [
        "--tokens", "257", "--hidden", "128", "--output-dim", "128",
        "--experts", "2", "--routes", "258",
        "--tile-m", "256", "--tile-n", "128", "--cluster-m", "2",
        "--warmup", "1", "--iterations", "2", "--timing-samples", "2",
    ]
    if tma_gather:
        argv.append("--use-tma-gather")
    if not dynamic:
        argv.append("--no-dynamic-persistent")
    if activation is not None:
        argv.extend(("--activation", activation))
    if down:
        argv.append("--down-projection")
    args = runner.parse_args(argv)
    runner.shared.validate_args(args)
    torch.manual_seed(args.seed)
    inputs = runner.shared.prepare_inputs(args, torch.device("cuda"))
    # 129 routes/expert exercises partial M tiles with 2-CTA MMA. Include
    # repeated and boundary token IDs, so the check cannot pass as dense GEMM.
    inputs.A_idx[0] = 256
    inputs.A_idx[1] = 256
    inputs.A_idx[129] = 0
    launch = runner.shared.make_launch(args, inputs)
    launch()
    runner.shared.check_correctness(
        inputs, activation=activation, atol=args.atol, rtol=args.rtol
    )

    # Graph replay must overwrite poisoned outputs and preserve the two-GEMM
    # dependency when a down projection consumes the gathered activation.
    inputs.up_output.fill_(float("nan"))
    inputs.output.fill_(float("nan"))
    runner.shared.benchmark(launch, iterations=2, samples=2, use_cuda_graph=True)
    runner.shared.check_correctness(
        inputs, activation=activation, atol=args.atol, rtol=args.rtol
    )
