# import argparse
# import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation/lm_eval')))
# from transformers import AutoTokenizer
# from MiLo.core.quantize import *
from MiLo.models.hf.mixtral import MixtralMiLo
# from MiLo.models.hf.deepseek import DeepSeekMoEMiLo
# from evaluation.eval_wikitext2_ppl import eval_perplexity

# from datasets import load_dataset
import torch, time
import numpy as np
from tqdm import tqdm

import gc
def cleanup():
	torch.cuda.empty_cache()
	gc.collect()


model_path = "/scratch/bcjw/bhuang4/mixtral/noIns_myQuant_HQQ_3bit_gs64-int3_symm-iter10-u32"
lorc_dir = "/scratch/bcjw/bhuang4/HQQ_LoRC/u32-int3-symm-iter10-iter10"


#quant_config = HqqConfig(nbits=3, group_size=64, axis=1)
ranks = {'self_attn': 0, 'experts':0}

model = MixtralMiLo.from_quantized(model_path,LoRC_weight_path=lorc_dir,
                                            LoRC_dtype = "int3_symm",
                                            ranks=ranks)

#tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1", trust_remote_code=True)


# model.save_pretrained(quant_model)
# tokenizer.save_pretrained(quant_model)
from MiLo.utils.patching import prepare_for_inference
prepare_for_inference(model, backend="milo3bitwithzero")


#ppl = eval_perplexity(model,tokenizer)

#Adapted from https://huggingface.co/transformers/v4.2.2/perplexity.html


# def eval_wikitext2(model, tokenizer, max_length=128, stride=16, verbose=True):
# 	model.eval()
# 	tokenizer.pad_token     = tokenizer.eos_token 
# 	tokenizer.padding_side  = "right" 
# 	tokenizer.add_eos_token = False

# 	dataset   = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
# 	encodings = tokenizer('\n\n'.join(dataset['text']), return_tensors='pt')
	
# 	encodings['input_ids'] = encodings['input_ids'].to('cuda')

# 	lls, t = [], []
# 	for i in tqdm(range(0, encodings['input_ids'].size(1), stride), disable=not verbose):
# 		begin_loc  = max(i + stride - max_length, 0)
# 		end_loc    = min(i + stride, encodings['input_ids'].size(1))
# 		trg_len    = end_loc - i  
# 		input_ids  = encodings['input_ids'][:,begin_loc:end_loc]
# 		target_ids = input_ids.clone()
# 		target_ids[:,:-trg_len] = -100 #ignore context 
# 		if i == 0:
# 			for _ in range(20):
# 				loss = model(input_ids, labels=target_ids).loss
# 		t1 = time.time()
# 		with torch.no_grad():
# 			loss = model(input_ids, labels=target_ids).loss 
# 		torch.cuda.synchronize()
# 		t2 = time.time()
# 		t.append((t2-t1))
# 		log_likelihood = loss * trg_len
# 		lls.append(log_likelihood)

# 		del input_ids, target_ids

# 	ppl       = np.round(float(torch.exp(torch.stack(lls).sum() / end_loc)), 4)
# 	pred_time = np.round(np.mean(t), 3)
# 	if(verbose):
# 		print('perplexity', ppl)
# 		print('pertoken time', str(pred_time) + '  sec')
# 		print('first token time',str(t[0])+' sec ')

# 	del encodings
# 	cleanup()

def test_first_token_latency(model):
	model.eval()
	batchsizes = [1, 16, 32]
	seq_len = 1024
	for batch_size in batchsizes:
		input_ids = torch.randint(10000, (batch_size, seq_len)).to("cuda:0")
		for _ in range(20): #warmup
			model(input_ids = input_ids)
		start = time.time()
		for i in range(300):
			model(input_ids = input_ids)
		torch.cuda.synchronize()
		end = time.time()
		TTFT = np.round((end - start)/300, 3)
		print("batchsize:", batch_size)
		print('first token time',str(TTFT)+' sec ')


# def main():
#     parser = argparse.ArgumentParser(description="")
#     parser.add_argument('--base_dir', type=str, required=True, help="base directory to save the quantized model")
#     parser.add_argument('--model_id', type=str, required=True, help="base model type")
#     args = parser.parse_args()

#     print(f"Start MiLo end-to-end latency evaluation on {args.base_dir}")

# 	if "Mixtral" in args.model_id:
#         model_id = "mistralai/Mixtral-8x7B-v0.1" 
#         AutoMiLoHFModel = MixtralMiLo
#     else:
#         NotImplementedError("This model is not implemented yet")
	
# 	quant_model_dir = f"{args.base_dir}/model"
#     lorc_dir = f"{args.base_dir}/lorc"
#     lorc_dtype = "int3_symm"
#     ranks = {'self_attn': 0, 'experts':0}
   
#     model = AutoMiLoHFModel.from_quantized(quant_model_dir,LoRC_weight_path=lorc_dir,
#                                             LoRC_dtype = lorc_dtype,
#                                             ranks=ranks)
#     tokenizer  = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)

#     if tokenizer.pad_token is None:
#         tokenizer.pad_token =tokenizer.eos_token

# 	save_file_path = os.path.join(args.base_dir, "eval_result.json")
# 	eval_wikitext2(model, tokenizer, max_length=1024, stride=512, verbose=True)
# 	return

# if __name__ == "__main__":
#     #main()
# 	eval_wikitext2(model, tokenizer, max_length=1024, stride=512, verbose=True)

	
#eval_wikitext2(model, tokenizer, max_length=1024, stride=512, verbose=True)
test_first_token_latency(model)