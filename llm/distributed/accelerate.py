import torch.distributed as torchdist
from accelerate import (
    Accelerator as __HFAccelerator,
    PartialState as __HFAcceleratorState,
    DeepSpeedPlugin,
)


class Accelerator(__HFAccelerator):
    """Use with torchrun.

    Usage:
        torchrun --nnodes=${__NNODES} \
                 --nproc_per_node=${__NPROC_PER_NODE} \
                 --rdzv_endpoint=${__IP}:${__PORT} \
                 "${@}"
    """

    def __init__(self, *args, deepspeed_config=None, **kwargs):
        deepspeed_plugin = (
            None
            if deepspeed_config is None
            else DeepSpeedPlugin(
                hf_ds_config=deepspeed_config,
                zero3_init_flag=True,
            )
        )
        super().__init__(*args, **kwargs, deepspeed_plugin=deepspeed_plugin)

    def sync_object(self, obj):
        if self.num_processes == 1:
            return obj

        __sync_obj = [None for _ in range(self.num_processes)]
        torchdist.all_gather_object(__sync_obj, obj)
        obj = __sync_obj[0]
        return obj


class AcceleratorState(__HFAcceleratorState):
    def sync_object(self, obj):
        if self.num_processes == 1:
            return obj

        __sync_obj = [None for _ in range(self.num_processes)]
        torchdist.all_gather_object(__sync_obj, obj)
        obj = __sync_obj[0]
        return obj
