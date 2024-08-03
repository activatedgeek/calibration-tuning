import os
from datetime import timedelta
import torch
import torch.distributed as torchdist
from torch.distributed.fsdp import (
    ShardingStrategy,
    BackwardPrefetch,
    StateDictType,
    CPUOffload,
)
from accelerate import (
    Accelerator as __HFAccelerator,
    PartialState as __HFAcceleratorState,
    DeepSpeedPlugin,
    FullyShardedDataParallelPlugin,
    InitProcessGroupKwargs,
)
from accelerate.utils import PrecisionType


class Accelerator(__HFAccelerator):
    def __init__(self, *args, **kwargs):
        deepspeed_plugin = None
        if os.getenv("ACCELERATE_USE_DEEPSPEED", "false") == "true":
            deepspeed_plugin = DeepSpeedPlugin(zero3_init_flag=True)

        fsdp_plugin = None
        if os.getenv("ACCELERATE_USE_FSDP", "false") == "true":
            os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"

            fsdp_plugin = FullyShardedDataParallelPlugin(
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                sync_module_states=True,
                use_orig_params=True,
                forward_prefetch=False,
                auto_wrap_policy="TRANSFORMER_BASED_WRAP",
                cpu_offload=CPUOffload(offload_params=False),
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
            )

        super().__init__(
            *args,
            **kwargs,
            mixed_precision=(
                PrecisionType.BF16
                if torch.cuda.is_bf16_supported()
                else PrecisionType.FP16
            ),
            fsdp_plugin=fsdp_plugin,
            deepspeed_plugin=deepspeed_plugin,
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=600))]
        )

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
