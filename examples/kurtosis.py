import torch
from safetensors.torch import save_file
from safetensors import safe_open
from huggingface_hub import snapshot_download

def kurtosis_value(tensor):
    n = tensor.numel()
    mean = torch.mean(tensor)
    std = torch.std(tensor)  
    deviations = tensor - mean
    fourth_moment = torch.mean(deviations ** 4)
    kurt = fourth_moment / (std ** 4)   
    excess_kurtosis = kurt - 3   
    return excess_kurtosis

def model_kurtosis(model_name):
    model_file_path = snapshot_download(model_name)
    if "Mixtral-8x7B" in model_name:
        file_count = 19
    elif "DeepSeek" in model_name:
        file_count = 7
    quant_list = ["expert","self_attn"]
    for f_idx in range(1,file_count+1):
        fname = f"{model_file_path}/model-{f_idx:05}-of-{file_count:05}.safetensors"
        with safe_open(fname, framework="pt", device="cuda") as f:
            for key in f.keys():
                if not any(q in key for q in quant_list): continue
                W_orig = f.get_tensor(key)
                with open(f'kurtosis_{model_name}.txt','a') as ftxt:
                    ftxt.write(f"{key}: {kurtosis_value(W_orig)}\n")

