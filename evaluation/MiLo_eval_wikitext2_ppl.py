import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MiLo.core.quantize import *
from MiLo.models.hf.mixtral import MixtralMiLo
from MiLo.models.hf.deepseek import DeepSeekMoEMiLo
from MiLo.engine.hf import AutoTokenizer
from evaluation.eval_wikitext2_ppl import eval_perplexity
import time
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--base_dir', type=str, required=True, help="base directory to save the quantized model")
    parser.add_argument('--model_id', type=str, required=True, help="base model type")
    args = parser.parse_args()

    print(f"Start few-shot evaluation on {args.base_dir}")
    
    if "Mixtral" in args.model_id:
        model_id = "mistralai/Mixtral-8x7B-v0.1" 
        AutoMiLoHFModel = MixtralMiLo
    elif "DeepSeek" in args.model_id:
        model_id = "deepseek-ai/deepseek-moe-16b-base"
        AutoMiLoHFModel = DeepSeekMoEMiLo
    else:
        NotImplementedError("This model is not implemented yet")

    quant_model_dir = f"{args.base_dir}/model"
    lorc_dir = f"{args.base_dir}/lorc"
    lorc_dtype = "int3"
    with open(f"{args.base_dir}/ranks.json", "r", encoding="utf-8") as f:
        ranks  = json.load(f)
   
    model = AutoMiLoHFModel.from_compressed(quant_model_dir,LoRC_weight_path=lorc_dir,
                                            LoRC_dtype = lorc_dtype,
                                            ranks=ranks)
    tokenizer  = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token =tokenizer.eos_token

    save_file_path = os.path.join(args.base_dir, "eval_result.json")
    begin = time.time()
    ppl = eval_perplexity(model,tokenizer)

    if os.path.exists(save_file_path):
        with open(save_file_path, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {} 
    else:
        data = {}

    data['wikitext2_ppl'] = ppl

    with open(save_file_path, 'w') as file:
        json.dump(data, file, indent=4)

    end = time.time()
    print(f"taking {end - begin}")
    return

if __name__ == "__main__":
    main()




