"""Prepared SM90 SwiGLU gather from fixed source buffers with changing live prefixes."""

from functools import partial
from typing import Callable, NamedTuple

import torch

from quack.cute_dsl_utils import get_device_capacity
from quack.gemm_interface import _gemm_act_call
from quack.gemm_runtime.host import run_gemm_epi_plan
from quack.tile_scheduler import MultiBufferGatherArguments


class BoundSwiGLUGather(NamedTuple):
    indices: tuple
    table: torch.Tensor
    output: torch.Tensor
    launch: Callable


class HopperSwiGLUGatherPlan:
    """Bind uniform per-source index prefixes and a live table to a concat-B SwiGLU.

    The handle owns its fixed tensors. Recreate it when replacing or resizing a backing
    tensor. Each binding retains its own views. The caller orders table publication and
    retires GPU readers before reusing the shared storage. Construction and binding
    inspect only metadata. The first nonempty launch resolves the ordinary QuACK plan.
    """

    def __init__(self, A, B, *, A_idx, gather_work_table, gather_work_table_ready,
                 postact_out, config, sm_count=None):
        self._xs, self._indices = tuple(A), tuple(A_idx)
        if len(self._xs) < 2 or len(self._xs) != len(self._indices):
            raise ValueError("prepared SwiGLU gather requires matching multi-buffer sources")
        first = self._xs[0]
        device = first.device
        if device.type != "cuda" or get_device_capacity(device)[0] != 9:
            raise ValueError("prepared SwiGLU gather requires Hopper SM90")
        if (config.device_capacity != 9 or config.is_dynamic_persistent or config.use_tma_gather
                or config.cluster_n != 1 or config.cluster_k != 1 or config.swap_ab
                or config.split_k != 1):
            raise ValueError("prepared SwiGLU gather requires static SM90 multi-buffer scheduling")
        if gather_work_table_ready is None or any(
            t.device != device for t in (*self._xs, *self._indices, B, postact_out,
                                         gather_work_table, gather_work_table_ready)
        ):
            raise ValueError("all buffers and the ready prefix must share the source device")
        if (first.dtype not in (torch.float16, torch.bfloat16) or first.ndim != 2
                or first.shape[1] == 0 or B.ndim != 3 or B.shape[1] != first.shape[1]
                or B.dtype != first.dtype or B.stride(1) != 1
                or postact_out.ndim != 2 or postact_out.dtype != first.dtype
                or postact_out.shape[1] == 0 or postact_out.shape[1] * 2 != B.shape[2]
                or postact_out.stride(1) != 1):
            raise ValueError("expected half-precision X[T,K], concat B[E,K,2H] and output[M,H]")
        if (any(x.ndim != 2 or x.shape[1] != first.shape[1] or x.dtype != first.dtype
                or x.stride() != first.stride() for x in self._xs)
                or not any(x.shape[0] for x in self._xs)):
            raise ValueError("X buffers must share column geometry and contain source rows")
        if any(i.dtype != torch.int32 or i.ndim != 1 or i.stride(0) != 1 for i in self._indices):
            raise ValueError("expected contiguous int32 source index vectors")
        if (gather_work_table.dtype != torch.int32 or gather_work_table.ndim != 2
                or gather_work_table.shape[1] != 2 + 2 * len(self._xs)
                or not gather_work_table.is_contiguous()
                or gather_work_table_ready.dtype != torch.int32
                or gather_work_table_ready.shape != (1,)):
            raise ValueError("invalid gather descriptor table or ready prefix")
        self._weights = B
        self._kernel_weights = B.mT
        self._table = gather_work_table
        self._ready = gather_work_table_ready
        self._output = postact_out
        self._max_source_rows = min(postact_out.shape[0] // len(self._xs),
                                    *(i.shape[0] for i in self._indices))
        self._config, self._sm_count = config, sm_count
        self._plan = None

    def bind(self, source_index_rows: int, table_rows: int) -> BoundSwiGLUGather:
        """Make the current views once. Padding stays outside every kernel operand."""
        if (not 0 <= source_index_rows <= self._max_source_rows
                or not 0 <= table_rows <= self._table.shape[0]
                or (table_rows and not source_index_rows)):
            raise ValueError("invalid live extents for prepared SwiGLU gather storage")
        indices = tuple(i[:source_index_rows] for i in self._indices)
        table = self._table[:table_rows]
        output = self._output[:source_index_rows * len(indices)]
        extra = MultiBufferGatherArguments(self._xs[1:], indices[1:])
        launch = partial(self._run, indices[0], extra, table, output)
        return BoundSwiGLUGather(indices, table, output, launch)

    def _run(self, first_index, extra, table, output, *, plan_observer=None):
        if table.shape[0] == 0:
            return
        if self._plan is None:
            self._plan = _gemm_act_call(
                self._xs[0], self._weights, None, output, None, None, activation="swiglu",
                A_idx=first_index, gather_work_table=table, gather_work_table_ready=self._ready,
                multi_buffer_gather_args=extra, concat_layout=("B",), dynamic_scheduler=False,
                tuned=False, config=self._config, sm_count=self._sm_count,
                plan_observer=plan_observer,
            )
            return
        if plan_observer is not None:
            plan_observer("plan_reused")
        # The SM90 table height, all index lengths and the TileStore output M are symbolic.
        # bind() varies only these extents and supplies views of the validated operands.
        run_gemm_epi_plan(
            self._plan, self._xs[0], self._kernel_weights, None, None, {"mAuxOut": output},
            A_idx=first_index, gather_table=table, gather_table_ready=self._ready,
            multi_buffer_gather=extra,
        )
