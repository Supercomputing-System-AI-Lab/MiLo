import torch
from transformers import AutoModelForCausalLM
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MiLo.models.hf.mixtral import MixtralMiLo
from MiLo.models.hf.deepseek import DeepSeekMoEMiLo
from MiLo.core.quantize import *
import time
from MiLo.engine.hf import AutoTokenizer
from evaluation.eval_wikitext2_ppl import eval_perplexity




def main():
    # model_id = "Mixtral"
    model_id = "DeepSeek"
    if "Mixtral" in model_id:
        model_id = "mistralai/Mixtral-8x7B-v0.1" 
        AutoMiLoHFModel = MixtralMiLo
    elif "DeepSeek" in model_id:
        model_id = "deepseek-ai/deepseek-moe-16b-base"
        AutoMiLoHFModel = DeepSeekMoEMiLo
    else:
        NotImplementedError("This model is not implemented yet")

    device = "cuda"
   
    quant_model_dir = "/media/volume/MiLo_v3/MiLo_api_d800"
    compress_config = BaseCompressConfig(
                                        # quantization config
                                         nbits = 3, 
                                         group_size = 64, 
                                         quant_scale = False, 
                                         quant_zero = False, 
                                         axis = 1,
                                        # compensator config
                                         iter = 10,
                                         sparse_rank = 0,
                                         dense_rank = 800,
                                         rank_strategy = None,
                                         compensator_dtype  = "int3"
                                         ) 


    model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float16,trust_remote_code=True)

    AutoMiLoHFModel.compress_model(model, 
                                   compress_config=compress_config, 
                                   device=device)
    AutoMiLoHFModel.save_compressed(model, quant_model_dir)


    del model
    model = AutoMiLoHFModel.from_quantized(quant_model_dir)
    tokenizer  = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token =tokenizer.eos_token
    begin = time.time()
    ppl = eval_perplexity(model,tokenizer)
    end = time.time()
    print(f"taking {end - begin}")
if __name__ == "__main__":
    main()

