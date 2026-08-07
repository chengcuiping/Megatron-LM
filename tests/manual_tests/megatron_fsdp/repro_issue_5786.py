#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Minimal reproduction for issue #5786: Megatron-FSDP deadlocks during initialization
when tensor parallelism is expressed with a torch-native rowwise-sharded DTensor.

Requires exactly 4 GPUs. Run from the repository root:

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 60s \
        torchrun --standalone --nproc_per_node=4 \
        tests/manual_tests/megatron_fsdp/repro_issue_5786.py

Expected: 4x INIT_OK, exit 0.
Observed on 9fa816bfd9dd994dc5acf6012fb4e841a56cfa4a: no rank prints INIT_OK.
Ranks 1 and 3 (tp_rank == 1) raise

    AssertionError: [Megatron-FSDP] DTensor chunk metadata is invalid.
    Offsets: (0, 8), Sizes: (2, 8), Global shape: torch.Size([8, 8]),
    Local shape: torch.Size([2, 8])

while ranks 0 and 2 block forever inside the all_reduce in
validate_uneven_dtensor(). torchrun notices the two dead ranks and tears the job
down, so the visible result is exit 1 after ~14s; launched without a supervisor
the two surviving ranks were still wedged after 90s.

Why. The weight is a DTensor of global shape [8, 8] sharded rowwise (`Shard(1)`)
over TP=2, so each rank stores a TP-local [8, 4]. FSDP then splits those 32
elements over DP=2, giving a rank-local buffer of 16 elements. But
make_fsdp_dtensor() views that buffer with the *global* trailing shape:

    param_and_grad_buffer.py:5143
        if len(orig_param.shape) > 1:
            local_shape = (-1, *orig_param.shape[1:])   # 8, not 4

16 elements viewed as (-1, 8) gives [2, 8] where [4, 4] is required.
validate_uneven_dtensor() does catch the bad layout -- but only on the ranks
whose chunk offset actually lands out of bounds, and only *after* entering a
collective. Half the ranks raise, the other half wait for them: a shape bug
becomes a deadlock.

Taking the trailing dims from the TP-local tensor instead, e.g.

    local_param_shape = to_local_if_dtensor(orig_param).shape

makes this script print 4x INIT_OK and exit 0.
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from megatron.core.distributed.fsdp.src.megatron_fsdp import fully_shard_model


def main():
    # Required: without it every rank would open NCCL on device 0.
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    # 4 ranks -> DP=2 x TP=2.
    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))

    # One nn.Linear, bias=False, so the model holds exactly one parameter: [8, 8].
    # (With a bias the two parameters share a bucket and the failure changes into a
    # plain "shape '[-1, 8]' is invalid for input of size 12" RuntimeError instead.)
    model = nn.Linear(8, 8, bias=False, device="cuda")

    # Torch-native rowwise tensor parallelism. This is the whole trigger: it makes
    # the parameter a DTensor whose `.shape` is global while its storage is TP-local.
    # With Shard(0), or with a plain non-DTensor weight, this script exits 0.
    model.weight = nn.Parameter(distribute_tensor(model.weight.detach(), mesh["tp"], [Shard(1)]))
    print(
        f"[rank {rank}] weight global={tuple(model.weight.shape)} "
        f"tp_local={tuple(model.weight.to_local().shape)}",
        flush=True,
    )

    # Megatron-FSDP V1 public API. Initialization alone reaches make_fsdp_dtensor();
    # no forward, backward, optimizer or training step is needed.
    # `fsdp_unit_modules` is not needed to trigger the bug, but the default
    # zero_dp_strategy=3 ("optim_grads_params") divides by the number of FSDP units,
    # so without it even a correct build dies with ZeroDivisionError.
    fully_shard_model(
        model, device_mesh=mesh, dp_shard_dim="dp", tp_dim="tp", fsdp_unit_modules=[nn.Linear]
    )

    print(f"[rank {rank}] INIT_OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
