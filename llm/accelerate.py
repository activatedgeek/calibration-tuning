import torch.distributed as torchdist
from accelerate import (
    Accelerator as __HFAccelerator,
    PartialState as __HFAcceleratorState,
)


__all__ = [
    "Accelerator",
    "AcceleratorState",
]


class Accelerator(__HFAccelerator):
    """Use with torchrun.

    Usage:
        torchrun --nnodes=${__NNODES} \
                    --nproc_per_node=${__NPROC_PER_NODE} \
                    --rdzv_endpoint=${__IP}:${__PORT} \
                    "${@}"
    """

    def sync_object(self, obj):
        __sync_obj = [None for _ in range(self.num_processes)]
        torchdist.all_gather_object(__sync_obj, obj)
        obj = __sync_obj[0]
        return obj


class AcceleratorState(__HFAcceleratorState):
    def sync_object(self, obj):
        __sync_obj = [None for _ in range(self.num_processes)]
        torchdist.all_gather_object(__sync_obj, obj)
        obj = __sync_obj[0]
        return obj
