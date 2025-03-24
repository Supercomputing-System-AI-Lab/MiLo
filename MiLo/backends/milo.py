import torch
import sys
import milo
from ..core.quantize import MiLoLinear, Quantizer


class MiLoWithZeros(torch.nn.Module):
    def __init__(
        self, W: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,u, v, bias= None, groupsize=64):
        super().__init__()

        m, n = W.shape
        device = W.device
        _linear = torch.nn.Linear(m, n)
        _linear.weight.data = W.half().t()
        _layer = milo.Layer3bitWithZeros(m, n, groupsize)
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
        self.U = u
        self.V = v
        del _linear, _layer
        torch.cuda.empty_cache()

    @torch.no_grad()
    def matmul(self, x):
        out = torch.empty(
            x.shape[:-1] + (self.scales.shape[1],), dtype=x.dtype, device=x.device
        )
        milo.mul_3bit_with_zeros(
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
        if self.U != None and self.V != None:
            out = out + (x @ self.V) @ self.U
        if self.bias is not None:
            out += self.bias
        return out

# ONLY WORKS WITH AXIS=1, group_size= 64
def patch_hqq_to_miloWithZeros(layer, patch_params):
    milo_layer = None
    if type(layer) is MiLoLinear:
        milo_layer = layer
    if milo_layer is None:
        return layer

    milo_layer = layer.linear_layer if hasattr(layer, "linear_layer") else layer
    # Check config suppport
    if (
        (milo_layer.meta["axis"] == 0)
        or (milo_layer.meta["group_size"] != 64)
        or (milo_layer.meta["nbits"] != 3)
    ):
        print("Skipping milo conversion for", milo_layer.name)
        return layer
    
    z = milo_layer.meta["zero"]
    s = milo_layer.meta["scale"]
    z = - z * s
    W_r = milo_layer.unpack(dtype=milo_layer.compute_dtype)
    W_r = W_r[:s.shape[0]]
    if milo_layer.U != None and milo_layer.V != None:
        u = milo_layer.U.t() 
        v = milo_layer.V.t()
    else:
        u = None
        v = None
    #W_r = W_r.t()
    #print(W_r.shape)  # Shape of the first tensor
    #print(s.shape)    # Shape of the second tensor
    #print(z.shape)    # Shape of the third tensor

    W_r = W_r * s + z
    n = milo_layer.meta["shape"][0]
    W_r = W_r.reshape((n,-1))
    s = s.reshape((n,-1))
    z = z.reshape((n,-1))
    milo_withzero_layer =  MiLoWithZeros(W_r.t(), s.t(), z.t(),u,v,bias=milo_layer.bias)

    del milo_layer.W_q
    del milo_layer.meta
    del milo_layer.bias
    del milo_layer
    torch.cuda.empty_cache()

    if isinstance(layer, MiLoLinear):
        return milo_withzero_layer

    if isinstance(layer, HQQLinearLoRA):
        layer.linear_layer = milo_withzero_layer

    torch.cuda.empty_cache()

    return layer