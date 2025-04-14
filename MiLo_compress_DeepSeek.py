import torch
from transformers import AutoModelForCausalLM
from MiLo.models.hf.deepseek import DeepSeekMoEMiLo as AutoMiLoHFModel
from MiLo.core.quantize import *


def main():
    device = "cuda"
    save_dir = "/DeepSeek_MiLo_s1"
    compress_config = BaseCompressConfig(
                                        # quantization config
                                         nbits = 3, 
                                         group_size = 64, 
                                         quant_scale = False, 
                                         quant_zero = False, 
                                         axis = 1,
                                        # compensator config
                                         iter = 10,
                                         sparse_rank = 16,
                                         dense_rank = 512,
                                         rank_strategy = None,
                                         compensator_dtype  = "int3"
                                         ) 
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-moe-16b-base", 
                                                 torch_dtype=torch.float16,
                                                 trust_remote_code=True)
    AutoMiLoHFModel.compress_model(model, 
                                   compress_config=compress_config, 
                                   device=device)   
    AutoMiLoHFModel.save_compressed(model, save_dir)




if __name__ == "__main__":
    main()

