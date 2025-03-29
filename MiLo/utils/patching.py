# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#####################################################
import torch
from torch import Tensor
from ..core.quantize import Quantizer, MiLoLinear
from ..core.utils import cleanup

from ..models.hf.base import AutoMiLoHFModel

from termcolor import colored
try:
    from ..backends.marlin import patch_hqq_to_marlin
except Exception:
        patch_hqq_to_marlin = None
        print(colored('Warning: failed to import the Marlin backend. Check if marlin is correctly installed if you want to use the Marlin backend (https://github.com/IST-DASLab/marlin).', 'yellow'))
try:
   from ..backends.milo import patch_hqq_to_miloWithZeros
except Exception:
       patch_hqq_to_miloWithZeros = None
       print(colored('Warning: failed to import the MiLoWithZero backend. Check if milo is correctly installed if you want to use the MiLo backend (in MiLo/kernels), please run setup.py install first.', 'yellow'))


def patch_linearlayers(model, fct, patch_param=None, verbose=False):
    base_class = model.base_class if (hasattr(model, "base_class")) else AutoMiLoHFModel
    base_class.setup_model(model)
    model.base_class.patch_linearlayers(
        model, fct, dict([(k, patch_param) for k in model.linear_tags]), verbose=verbose
    )


def patch_add_quant_config(layer, patch_param):
    if type(layer) is MiLoLinear:
        layer.compress_config = patch_param
    return layer


# add dummy weights to a layer
def patch_add_weight_param(layer, patch_param):
    if hasattr(layer, "weight") is False:
        if hasattr(layer, "device"):
            device_ = layer.device
        else:
            param = [p for p in layer.parameters()]
            device_ = param[0].device if (len(param) > 0) else patch_param["device"]

        fp_param = [p for p in layer.parameters() if p.is_floating_point()]
        dtype_ = fp_param[0].dtype if (len(fp_param) > 0) else patch_param["dtype"]

        layer.weight = torch.nn.Parameter(
            torch.zeros((1,), device=device_, dtype=dtype_), requires_grad=False
        )
    return layer


# Optimize HQQLinear.forward for inference
def patch_hqq_inference(layer, patch_param):
    def forward_hqq_inferece(self, x):
        out = torch.matmul(x.to(self.device), self.dequantize().T)  # TODO GEMV use-case
        if self.bias is not None:
            out += self.bias
        return out

    if type(layer) is MiLoLinear:
        layer.forward = lambda x: forward_hqq_inferece(layer, x)


    return layer


# Optimize HQQLinearLoRA.forward for inference
def patch_lora_inference(layer, patch_param):
    def forward_lora_inference(self, x):
        out = torch.matmul(torch.matmul(x, self.lora_A), self.lora_B) * self.scaling
        return out

    return layer

# Copied from https://github.com/pytorch/ao/blob/b523f9f9e15b6fb80d10f585d9cf45e0c5e4d10e/torchao/quantization/utils.py#L486-L501
def recommended_inductor_config_setter():
    """
    Set inductor config to use the following optimizations which have been showed to improve performance for quantized models:
        coordinate_descent_tuning = True
        coordinate_descent_check_all_directions = True
        force_fuse_int_mm_with_mul = True
        fx_graph_cache = True
        triton.unique_kernel_names = True
        torch.set_float32_matmul_precision("high")
    """
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.coordinate_descent_check_all_directions = True
    torch._inductor.config.force_fuse_int_mm_with_mul = True
    torch._inductor.config.fx_graph_cache = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch.set_float32_matmul_precision("high")

def prepare_for_inference(model, allow_merge=False, backend="default", verbose=False):

    patch_linearlayers(model, patch_hqq_inference)
    patch_linearlayers(model, patch_lora_inference)
    cleanup()

    if backend == "marlin" and (patch_hqq_to_marlin is not None):
        patch_linearlayers(model, patch_hqq_to_marlin, verbose=verbose)
        cleanup()
    if backend == "milo" and (patch_hqq_to_miloWithZeros is not None):
        patch_linearlayers(model, patch_hqq_to_miloWithZeros, verbose=verbose)
        cleanup()

    patch_linearlayers(
        model, patch_add_weight_param, {"device": model.device, "dtype": model.dtype}
    )
    cleanup()

    return backend


def get_lowrank_tuple_torch_gpu(tensor, max_rank, eps=None):
    t = tensor.t().float()
    u, s, v = torch.linalg.svd(t)
    u, s, v = u[:, :max_rank], s[:max_rank], v[:max_rank, :]
    us = torch.matmul(u, torch.diag(s))
    A, B = (v.t(), us.t())  # t ~ AB
    if eps is not None:
        A = A.clamp(min=-eps, max=eps)
        B = B.clamp(min=-eps, max=eps)
    return A.to(tensor.dtype), B.to(tensor.dtype)


