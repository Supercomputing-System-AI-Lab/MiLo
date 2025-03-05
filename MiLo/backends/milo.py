# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#####################################################
# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#####################################################
import torch
import sys
import milo
from ..core.quantize import HQQLinear, Quantizer
from ..core.peft import HQQLinearLoRA


class MiLoWithZeros(torch.nn.Module):
    def __init__(
        self, W: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,bias= None, groupsize=64):
        super().__init__()

        m, n = W.shape
        device = W.device
        _linear = torch.nn.Linear(m, n)
        _linear.weight.data = W.half().t()
        _layer = marlin.Layer3bit_64_256_WithZero(m, n, groupsize)
        _layer.k = m
        _layer.n = n
        _layer.groupsize = groupsize
        _layer.B1 = torch.empty((m // 16, n * 16 // 16), dtype=torch.int, device=device)
        _layer.B2 = torch.empty((m // 16, n * 16 // 32), dtype=torch.int, device=device)
        _layer.s = torch.empty(
            (m // groupsize, n), dtype=torch.half, device=device
        )
        _layer.z = torch.empty(
            (m // groupsize, n), dtype=torch.half, device=device
        )
        _layer.pack(_linear, scales.t(),zeros.t())
        self.bias = bias.half() if (bias is not None) else None
        self.Wq_packed1 = _layer.B1.clone()
        self.Wq_packed2 = _layer.B2.clone()
        self.scales = _layer.s.clone()
        self.zeros = _layer.z.clone()
        self.workspace_fp = torch.zeros(n // 128 * 16, device=device)
        self.in_features = m
        self.out_features = n
        self.group_size = groupsize
        self.axis = 1
        self.device = device
        self.compute_dtype = torch.float16
        del _linear, _layer
        torch.cuda.empty_cache()

    @torch.no_grad()
    def matmul(self, x):
        out = torch.empty(
            x.shape[:-1] + (self.scales.shape[1],), dtype=x.dtype, device=x.device
        )
        #print("marlin.py,171,x,B1,C,s",x.shape,self.Wq_packed1.shape,out.shape,self.scales.shape)
        milo.mul_3bit_with_zero(
            x.to(self.device).view((-1, x.shape[-1])),
            self.Wq_packed1,
            self.Wq_packed2,
            out.view((-1, out.shape[-1])),
            self.scales,
            self.zeros,
            self.workspace_fp,
        )
        return out

    @torch.jit.ignore
    def forward(self, x):
        #print("here in 3bit forward! \n")
        out = self.matmul(x)
        #out = out + self.u[:x.shape[0]] * self.v 
        if self.bias is not None:
            out += self.bias
        return out

# ONLY WORKS WITH AXIS=1, group_size= 64
def patch_hqq_to_miloWithZeros(layer, patch_params):
    hqq_layer = None
    if type(layer) is HQQLinear:
        hqq_layer = layer
    if type(layer) is HQQLinearLoRA:
        hqq_layer = layer.linear_layer
    if hqq_layer is None:
        return layer

    hqq_layer = layer.linear_layer if hasattr(layer, "linear_layer") else layer
    # Check config suppport
    if (
        (hqq_layer.meta["axis"] == 0)
        or (hqq_layer.meta["group_size"] != 64)
        or (hqq_layer.meta["nbits"] != 3)
    ):
        print("Skipping marlin conversion for", hqq_layer.name)
        return layer
    
    z = hqq_layer.meta["zero"]
    s = hqq_layer.meta["scale"]
    z = - z * s
    W_r = hqq_layer.unpack(dtype=hqq_layer.compute_dtype)
    W_r = W_r[:s.shape[0]]
    #W_r = W_r.t()
    
    #print(W_r.shape)  # Shape of the first tensor
    #print(s.shape)    # Shape of the second tensor
    #print(z.shape)    # Shape of the third tensor

    W_r = W_r * s + z
    W_r = W_r.reshape(hqq_layer.meta["shape"])
    #print("in marlin.py",W_r.shape,s.shape)
    marlin_3bit_withzero_layer =    MiLoWithZeros(W_r.t(), s.t(), z.t(),bias=hqq_layer.bias)

    del hqq_layer.W_q
    del hqq_layer.meta
    del hqq_layer.bias
    del hqq_layer
    torch.cuda.empty_cache()

    if isinstance(layer, HQQLinear):
        return marlin_3bit_withzero_layer

    if isinstance(layer, HQQLinearLoRA):
        layer.linear_layer = marlin_3bit_withzero_layer

    torch.cuda.empty_cache()

    return layer