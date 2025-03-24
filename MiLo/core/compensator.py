import torch
from .bitpack import BitPack
import json
from ..core.quantize import MiLoLinear
from .utils import compensator_dequantize
MIXTRAL_LAYERS = {
    "dense": ["self_attn"],
    "sparse": ["experts"],
    "layer_count": 32
}
DEEPSEEK_LAYERS = {
    "dense": ['self_attn', 'shared', 'layers.0.mlp'],
    "sparse": ['mlp.experts'],
    "layer_count": 27
}

def rank_generate(model_id, sparse_rank,dense_rank,strategy):
    if model_id == "mistralai/Mixtral-8x7B-v0.1":
        model_layer_info = MIXTRAL_LAYERS
    elif model_id == "deepseek-ai/deepseek-moe-16b-base":
        model_layer_info = DEEPSEEK_LAYERS
    else:
        raise NotImplementedError
    
    if strategy == None:
        ranks = {
            **{name: dense_rank for name in model_layer_info["dense"]},
            **{name: sparse_rank for name in model_layer_info["sparse"]},
        }
    elif strategy == "frequency" and model_id == "deepseek-ai/deepseek-moe-16b-base":
        ranks = {
            **{name: dense_rank for name in model_layer_info["dense"]},
        }
        with open("DeepSeek_expt_freq.json", "r") as f:
            data = json.load(f)
        for layer_index in range(27):
            freq = data[layer_index]
            freq_sum = sum(freq)
            for expert_index in range(len(freq)):
                rank = int(round(freq[expert_index] / freq_sum * (sparse_rank*len(freq))))   # Assign rank based on the weight
                ranks[f'layers.{layer_index + 1}.mlp.experts.{expert_index}.'] = rank

    elif strategy == "Kurtosis" and model_id == "mistralai/Mixtral-8x7B-v0.1":
        if sparse_rank == 16:
            k = 2
        elif sparse_rank == 32:
            k = 3
        else:
            raise NotImplementedError("Currently Mixtral Kurtosis strategy only support the avg rank of 16 and 32")
        with open("Mixtral_kurtosis_values.json", "r") as f:
            data = json.load(f)
        for name, kurtosis in data.items():
            if "self_attn" in name:
                continue
            else:
                if kurtosis < 1:
                    rank = 0
                elif kurtosis > 9:
                    rank = 1024
                else:
                    rank = 2 ** (kurtosis+k)
            text = name.replace('.weight', '').strip()
            ranks[text] = rank
    else:
        raise NotImplementedError
    print(ranks)
    return ranks


# def quantize_full_to_int8(tensor_in):
#     max_val, _ = torch.max(tensor_in, dim=1, keepdim=True)
#     min_val, _ = torch.min(tensor_in, dim=1, keepdim=True)
#     max_min = max_val - min_val
#     max_min[max_min==0] = 255  #deal with the case max = min
#     scale = 255 / max_min
#     zero = - torch.round(scale * min_val) - 128  
#     tensor_int8 = torch.round(tensor_in * scale + zero).to(torch.int8)
#     return  scale,zero,tensor_int8



def load_compensators(model,compensators,ranks):
    for name, module in model.named_modules():
        if type(module) is MiLoLinear:
            UV_quantized = compensators.pop(name, None)
            orig_shape=module.meta['shape']
            # module.compress_config["compensator_params"]["ranks"] = ranks
            rank = next((value for key, value in ranks.items() if key in name), None)
            compensator_dtype = module.compress_config["compensator_params"]["compensator_dtype"]
            compensator_quantize_gs = module.compress_config["compensator_params"]["compensator_quant_gs"]
            if rank is not None and rank > 0:
                module.U, module.V = compensator_dequantize(UV_quantized, orig_shape, rank, compensator_quantize_gs, compensator_dtype)




    


