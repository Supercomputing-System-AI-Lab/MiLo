import torch
from transformers import AutoModelForCausalLM
from MiLo.models.hf.mixtral import MixtralMiLo as AutoMiLoHFModel
from MiLo.core.quantize import *


def main():

    device = "cuda"
    quant_model_dir = "/scratch/bcjw/zshao3/huggingface/mixtral3bit"
    compress_config = BaseCompressConfig(
                                        # quantization config
                                         nbits = 3, 
                                         group_size = 64, 
                                         quant_scale = False, 
                                         quant_zero = False, 
                                         axis = 1,
                                        # compensator config
                                         iter = 0,
                                         sparse_rank = 16,
                                         dense_rank = 512,
                                         rank_strategy = "Kurtosis",
                                         compensator_dtype  = "int3"
                                         ) 
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1",
                                                 torch_dtype=torch.float16,
                                                 trust_remote_code=True)
    AutoMiLoHFModel.compress_model(model, 
                                   compress_config=compress_config, 
                                   device=device)    
    AutoMiLoHFModel.save_compressed(model, quant_model_dir)



if __name__ == "__main__":
    main()

