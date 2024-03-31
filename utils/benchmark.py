from local_configs.NYUDepthv2.DFormer_Tiny import config as cfg
from models.builder import EncoderDecoder as segmodel
import torch.nn as nn
import torch.distributed as dist
dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=cfg.background)
BatchNorm2d = nn.SyncBatchNorm
model = segmodel(
    cfg=cfg,
    criterion=criterion,
    norm_layer=BatchNorm2d,
    single_GPU=False,
).cuda()

from fvcore.nn import FlopCountAnalysis
import torch

x = (torch.rand(1, 3, 530, 730).cuda(), torch.rand(1, 3, 530, 730).cuda())
flops = FlopCountAnalysis(model, x)

from typing import List, Any
from numbers import Number
import numpy as np
from fvcore.nn.jit_handles import get_shape


def add_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    return np.prod(get_shape(outputs[0]))


flops.set_op_handle(
    "aten::add",
    add_flop_jit,
    "aten::add_",
    add_flop_jit,
    "aten::mul",
    add_flop_jit,
    "aten::mul_",
    add_flop_jit,
)

print(flops.total() / 1e9, "GFLOPs")

from mmengine.analysis import get_model_complexity_info

analysis_results = get_model_complexity_info(model, inputs=x)

print(analysis_results["params_str"])

print(analysis_results['out_table'])