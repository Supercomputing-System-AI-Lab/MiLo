import torch
import sys
import milo 
from ..core.quantize import MiLoLinear, Quantizer

def compensator_dequantize(UV_quantized, orig_shape, rank, compensator_quantize_gs, compensator_dtype):
    if compensator_dtype == 'int3':
        zero = 4
    else:
        raise NotImplementedError
    (U_scale,U_packed),(V_scale,V_packed) = UV_quantized
    U_q = unpack_3bit_32_sign(U_packed)
    V_q = unpack_3bit_32_sign(V_packed)
    U_q = U_q[:int(orig_shape[0] * (rank / compensator_quantize_gs)),:]

    V_q = V_q[:int(orig_shape[1] * rank / compensator_quantize_gs), :]

    U_dq = ((U_q - zero) * 2 * U_scale / 7).reshape(orig_shape[0], -1)
    V_dq = ((V_q - zero) * 2 * V_scale / 7).reshape(-1, orig_shape[1])
    return U_dq.half(),V_dq.half()
    
class MiLo_Asymmetric_Linear(torch.nn.Module):
    def __init__(self, W: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, u: None, v: None,
                 bias=None, groupsize=64):
        super().__init__()
        m, n = W.shape
        device = W.device

        _linear = torch.nn.Linear(m, n)
        _linear.weight.data = W.half().t()

        _layer = milo.Layer3bitWithZeros(m, n, groupsize)
        _layer.k = m
        _layer.n = n
        _layer.groupsize = 64

        _layer.B1 = torch.empty(
            (m // 16, n * 16 // 16), dtype=torch.int, device=device
        )
        _layer.B2 = torch.empty(
            (m // 16, n * 16 // 32), dtype=torch.int, device=device
        )
        _layer.s = torch.empty((m // groupsize, n), dtype=torch.half, device=device)
        _layer.z = torch.empty((m // groupsize, n), dtype=torch.half, device=device)
        _layer.pack(_linear, scales.t(), zeros.t())
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
        self.U = torch.nn.Parameter(u, requires_grad=False) if (u is not None) else None
        self.V = torch.nn.Parameter(v, requires_grad=False) if (v is not None) else None


        del _linear, _layer
        torch.cuda.empty_cache()

    @torch.no_grad()
    def matmul(self, x):
        out = torch.empty(
            x.shape[:-1] + (self.scales.shape[1],), 
            dtype=x.dtype, 
            device=x.device
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
        out = self.matmul(x)
        if self.bias is not None:
            out += self.bias
        if self.U != None and self.V != None:
            out = out + (x @ self.V) @ self.U
        return out


def patch_hqq_to_milo_asymmetric(layer, patch_params):
    # import sys
    # print("Starting patch_hqq_to_milo_asymmetric")
    # sys.stdout.flush()
    milo_layer = None
    if isinstance(layer, MiLoLinear):
        #print("Layer is MiLoLinear")
        milo_layer = layer

    if milo_layer is None:
        print("milo_layer is None, returning original layer")
        return layer

    milo_layer = layer.linear_layer if hasattr(layer, "linear_layer") else layer
    # print(f"Layer meta: {milo_layer.meta}")
    # Check config
    if (
        (milo_layer.meta["axis"] == 0)
        or (milo_layer.meta["group_size"] != 64)
        or (milo_layer.meta["nbits"] != 3)
    ):
        print("Skipping milo conversion for", milo_layer)
        return layer

    z = milo_layer.meta["zero"]
    s = milo_layer.meta["scale"]

    W_r = milo_layer.unpack(dtype=milo_layer.compute_dtype)
    W_r = W_r[: s.shape[0]]

    # Combine them
    W_r = (W_r - z) * s
    z = -z * s

    n = milo_layer.meta["shape"][0]
    W_r = W_r.reshape((n, -1))
    s = s.reshape((n, -1))
    z = z.reshape((n, -1))
    
    # print(f"milo layer contains: {dir(milo_layer)}")
    # print(f"milo layer U: {layer.U}")
    # print(f"milo layer V: {layer.V}")
    # print(f"milo layer meta: {milo_layer.meta}")
    # if milo_layer.UV_quantized is not None:
    #     print("Found UV_quantized, dequantizing...")
    #     u, v = compensator_dequantize(milo_layer.UV_quantized, milo_layer.meta["shape"], milo_layer.meta["rank"], milo_layer.meta["compensator_quant_gs"], milo_layer.meta["compensator_dtype"])
    #     print(f"Dequantized U shape: {u.shape}, V shape: {v.shape}")
    # else:
    #     print("UV_quantized is None")
    if milo_layer.U is not None and milo_layer.V is not None:
        u = milo_layer.U.t()
        v = milo_layer.V.t()
    else:
        u = None
        v = None

    MiLo_Asymmetric_Linear_layer = MiLo_Asymmetric_Linear(
        W_r.t(), s.t(), z.t(),u,v, bias=milo_layer.bias
    )

    del milo_layer.W_q
    del milo_layer.meta
    del milo_layer.bias
    del milo_layer
    torch.cuda.empty_cache()

    if isinstance(layer, MiLoLinear):
        return  MiLo_Asymmetric_Linear_layer
    if isinstance(layer, MiLoLinearLoRA):
        layer.linear_layer = MiLo_Asymmetric_Linear_layer
    return layer


class MiLo_Symmetric_Layer(torch.nn.Module):
    def __init__(self, W: torch.Tensor, scales: torch.Tensor, qz=None,u=None, v=None,
                 bias=None, groupsize=64):
        super().__init__()
        m, n = W.shape
        device = W.device

        _linear = torch.nn.Linear(m, n)
        _linear.weight.data = W.half().t()

        _layer = milo.Layer3bit(m, n, groupsize)
        _layer.k = m
        _layer.n = n
        _layer.groupsize = groupsize

        _layer.B1 = torch.empty(
            (m // 16, n * 16 // 16), dtype=torch.int, device=device
        )
        _layer.B2 = torch.empty(
            (m // 16, n * 16 // 32), dtype=torch.int, device=device
        )
        _layer.s = torch.empty(
            (m // groupsize, n), dtype=torch.half, device=device
        )

        _layer.pack(_linear, scales.t())
        self.bias = bias.half() if (bias is not None) else None
        self.Wq_packed1 = _layer.B1.clone()
        self.Wq_packed2 = _layer.B2.clone()
        self.scales = _layer.s.clone()

        self.workspace_fp = torch.zeros(n // 128 * 16, device=device)
        self.in_features = m
        self.out_features = n
        self.group_size = groupsize
        self.axis = 1
        self.device = device
        self.compute_dtype = torch.float16
        self.U = torch.nn.Parameter(u, requires_grad=False) if (u is not None) else None
        self.V = torch.nn.Parameter(v, requires_grad=False) if (v is not None) else None
        self.qz = torch.nn.Parameter(qz, requires_grad=False) if (qz is not None) else None

        del _linear, _layer
        torch.cuda.empty_cache()

    @torch.no_grad()
    def matmul(self, x):
        out = torch.empty(
            x.shape[:-1] + (self.scales.shape[1],), 
            dtype=x.dtype, 
            device=x.device
        )
        milo.mul_3bit(
            x.to(self.device).view((-1, x.shape[-1])),
            self.Wq_packed1,
            self.Wq_packed2,
            out.view((-1, out.shape[-1])),
            self.scales,
            self.workspace_fp,
            thread_k=64,
            thread_n=256,
        )
        return out

    @torch.jit.ignore
    def forward(self, x):
        out = self.matmul(x)
        if self.qz is not None:
            # Extra shift or correction if desired
            y = x.reshape(*x.shape[:-1], -1, 64).sum(axis=-1)
            out += torch.matmul(y, self.qz)

        if self.bias is not None:
            out += self.bias
        if self.U != None and self.V != None:
            out = out + (x @ self.V) @ self.U
        return out


# ONLY WORKS WITH AXIS=1, group_size=64
def patch_hqq_to_milo_symmetric(layer, patch_params):
    hqq_layer = None
    if isinstance(layer, MiLoLinear):
        hqq_layer = layer
    # elif isinstance(layer, MiLoLinearLoRA):
    #     hqq_layer = layer.linear_layer

    if hqq_layer is None:
        return layer

    # Check config suppport
    if (
        (hqq_layer.meta["axis"] == 0)
        or (hqq_layer.meta["group_size"] != 64)
        or (hqq_layer.meta["nbits"] != 3)
    ):
        print("Skipping milo conversion for", hqq_layer)
        return layer

    z = hqq_layer.meta["zero"]
    s = hqq_layer.meta["scale"]

    W_r = hqq_layer.unpack(dtype=hqq_layer.compute_dtype)
    # Make sure shapes match
    W_r = W_r[: s.shape[0]]
    # print("W_r.shape: ", W_r.shape)
    # print("hqq_layer.meta['shape']: ",hqq_layer.meta["shape"])

    # Possibly you want a shift; for now we skip it or define it:
    z_shift = 4.0  # If you truly want the same offset as 4-bit, define it

    # This logic is somewhat different from the 4-bit approach.
    # Adjust to your real desired formula:
    W_r = (W_r - z_shift) * s
    n = hqq_layer.meta["shape"][0]
    W_r = W_r.reshape((n, -1))
    s = s.reshape((n, -1))
    z = z.reshape((n, -1))

    # If you truly need an extra 'u' term, define it similarly:
    if isinstance(z, (torch.Tensor, torch.nn.Parameter)):
        # Example usage mimicking the 3-bit style:
        qz = s * (-z + z_shift)
    
    else:
        qz = None

    if milo_layer.U != None and milo_layer.V != None:
        u = milo_layer.U.t() 
        v = milo_layer.V.t()
    else:
        u = None
        v = None
    milo_layer = MiLo_Symmetric_Layer(
        W_r.t(),
        s.t(),
        qz.t() if (qz is not None) else None,
        u,
        v,
        bias=hqq_layer.bias,
        groupsize=hqq_layer.meta["group_size"],
    )

    del hqq_layer.W_q
    del hqq_layer.meta
    del hqq_layer.bias
    del hqq_layer
    torch.cuda.empty_cache()

    if isinstance(layer, MiLoLinear):
        return milo_layer
    # if isinstance(layer, MiLoLinearLoRA):
    #     layer.linear_layer = milo_layer

    return layer
