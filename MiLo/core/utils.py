import torch
import gc
import math
from typing import Union
from .bitpack import BitPack

def cleanup() -> None:
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def is_divisible(val1: int, val2: int) -> bool:
    return int(val2 * math.ceil(val1 / val2)) == val1


def zero_pad_row(
    tensor: torch.Tensor, num_rows: int, dtype: Union[torch.dtype, None] = None
) -> torch.Tensor:
    out = torch.zeros(
        [num_rows, tensor.shape[1]],
        device=tensor.device,
        dtype=tensor.dtype if (dtype is None) else dtype,
    )
    out[: len(tensor)] = tensor

    return out

def quantize_full_to_int3(tensor_in, group_size):
    tensor_in = tensor_in.reshape(-1,group_size)
    scale, _ = torch.max(tensor_in, dim=1, keepdim=True)
    tensor_int8 = torch.round(tensor_in * 7/(2*scale) ) + 4
    tensor_int8 = torch.clamp(tensor_int8,0,7).to(torch.int32)
    tensor_packed = BitPack.pack_3bit_32(tensor_int8)
    return  (scale, tensor_packed)

# Map a Pytorch dtype into a safetensor dtype
def encode_safetensor_type(data):
    if isinstance(data, (torch.Tensor, torch.nn.Parameter)):
        return data
    if isinstance(data, torch.Size):
        return torch.tensor(data)
    if isinstance(data, torch.dtype):
        data = str(data)
    if isinstance(data, bool):
        return torch.tensor(int(data), dtype=torch.uint8)
    if isinstance(data, int):
        return torch.tensor(data, dtype=torch.int32)
    if isinstance(data, float):
        return torch.tensor(data, dtype=torch.float32)
    if isinstance(data, str):
        return torch.tensor([ord(i) for i in data], dtype=torch.uint8)


# Decode a safetensor dtype into a Pytorch dtype
def decode_safetensor_type(data, data_type):
    if data_type in [torch.Tensor, torch.nn.Parameter]:
        return data
    if data_type is torch.Size:
        return torch.Size(data)
    if data_type is bool:
        return bool(data.item())
    if data_type is int:
        return int(data.item())
    if data_type is float:
        return float(data.item())
    if data_type is str:
        return "".join([chr(i) for i in data])
    if data_type is torch.dtype:
        return eval("".join([chr(i) for i in data]))

def compensator_dequantize(UV_quantized, orig_shape, rank, compensator_quantize_gs, compensator_dtype):
    if compensator_dtype == 'int3':
        zero = 4
    else:
        raise NotImplementedError
    print(orig_shape)
    (U_scale,U_packed),(V_scale,V_packed) = UV_quantized
    U_q = BitPack.unpack_3bit_32(U_packed)
    V_q = BitPack.unpack_3bit_32(V_packed)

    U_q = U_q[:int(orig_shape[0] * (rank / compensator_quantize_gs)),:]

    V_q = V_q[:int(orig_shape[1] * rank / compensator_quantize_gs), :]

    U_dq = ((U_q - zero) * 2 * U_scale / 7).reshape(orig_shape[0], -1)
    V_dq = ((V_q - zero) * 2 * V_scale / 7).reshape(-1, orig_shape[1])
    return U_dq.half(),V_dq.half()
