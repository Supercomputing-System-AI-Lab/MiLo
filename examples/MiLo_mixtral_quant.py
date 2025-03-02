import torch
from transformers import AutoModelForCausalLM
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MiLo.models.hf.mixtral import MixtralMiLo
from MiLo.models.hf.deepseek import DeepSeekMoEMiLo
from MiLo.core.quantize import *
import argparse
import json
import time

def kurtosis_flag(value):
    if value is None:
        return "s1"  #default is s1
    elif value in ['s1', 's2']:
        return value
    else:
        raise argparse.ArgumentTypeError("the options are s1 or s2")
    
def main():
    parser = argparse.ArgumentParser(description="Use MiLo to quantize Mixtral-8x7B to reproduce the results from paper")
    parser.add_argument('--base_dir', type=str, required=True, help="base directory to save the quantized model")
    parser.add_argument('--model_id', type=str, required=True, help="model id to quantize")
    parser.add_argument('--dense_rank', type=int, nargs='?', default = 8, help="set rank for dense layers")
    parser.add_argument('--sparse_rank', type=int, nargs='?', default = 8, help="set rank for sparse layers")
    parser.add_argument('--kurtosis_flag', type =kurtosis_flag, nargs='?', default=False, help="Use kurtosis strategy for experts, only valid for Mixtral")
    parser.add_argument('--frequency_flag', type = int, nargs='?', default=False, help="Use experts frequency strategy for experts, only valid for DeepSeekMoE")
    args = parser.parse_args()

    if "Mixtral" in args.model_id:
        model_id = "mistralai/Mixtral-8x7B-v0.1" 
        AutoMiLoHFModel = MixtralMiLo
    elif "DeepSeek" in args.model_id:
        model_id = "deepseek-ai/deepseek-moe-16b-base"
        AutoMiLoHFModel = DeepSeekMoEMiLo
    else:
        NotImplementedError("This model is not implemented yet")

    compute_dtype = torch.float16
    device        = "cuda"

    quant_model_dir = f"{args.base_dir}/model"
    lorc_dir = f"{args.base_dir}/lorc"
    os.makedirs(quant_model_dir,exist_ok=True)
    os.makedirs(lorc_dir,exist_ok=True)

    ### Quantization Config
    quant_config = BaseQuantizeConfig(nbits=3, group_size=64, quant_scale=False, quant_zero=False, axis=1) 

    ### LoRC Config
    iteration  = 20
    lorc_dtype = "int3_symm"
    if model_id == "mistralai/Mixtral-8x7B-v0.1":
        if args.kurtosis_flag == False:
            ranks={'self_attn': args.dense_rank, 'experts':args.sparse_rank}
        else:
            ranks = {}
            if args.kurtosis_flag == "s1":
                k = 2
            elif args.kurtosis_flag == "s2":
                k = 3
            else:
                NotImplementedError("This strategy is not implemented yet")
            with open('examples/kurtosis_mixtral.txt', 'r') as file:
                lines = file.readlines()
            for line in lines:
                parts = line.split(':')
                text = parts[0].replace('.weight', '').strip()
                if "self_attn" in text: 
                    rank = args.dense_rank
                else:
                    number = parts[1].strip()
                    kurtosis = round(float(number))
                    if kurtosis < 1:
                        rank = 0
                    elif kurtosis > 9:
                        rank = 1024
                    else:
                        rank = 2 ** (kurtosis+k)
                ranks[text] = rank
    elif model_id == "deepseek-ai/deepseek-moe-16b-base":
        if args.frequency_flag == False:
            ranks={'self_attn': args.dense_rank, 'shared':args.dense_rank, 'layers.0.mlp':args.dense_rank,'mlp.experts':args.sparse_rank}
        else:
            NotImplementedError("DeepSeek expert freq TODO")

    with open(f"{args.base_dir}/ranks.json", "w", encoding="utf-8") as f:
        json.dump(ranks, f, ensure_ascii=False, indent=4)

    print(f"Doing MiLo to: {model_id}")
    print(f"Dense Rank: {args.dense_rank}")
    print(f"Sparse Rank: {args.sparse_rank}")
    print(f"Using Kurtosis strategy: {args.kurtosis_flag}")
    print("Notice: if using Kurtosis strategy for Mixtral, the input sparse rank is invalid")
    print(f"Using expert frequency for DeepSeek: {args.kurtosis_flag}")
    print(f"Save quantized model to: {args.base_dir}")

    model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=compute_dtype, trust_remote_code=True)
    begin_time = time.time()
    AutoMiLoHFModel.quantize_model(model, 
                                quant_config=quant_config, 
                                compute_dtype=compute_dtype, 
                                device=device, 
                                lorc_path=lorc_dir, 
                                ranks=ranks,  #choose from rank strategy defined above
                                iters=iteration,
                                lorc_dtype=lorc_dtype)
    AutoMiLoHFModel.save_quantized(model, quant_model_dir)
    end_time = time.time()
    print(f"Quantization time: {end_time - begin_time:.2f}s")


if __name__ == "__main__":
    main()

