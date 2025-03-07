# import argparse
# import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation/lm_eval')))
from transformers import AutoTokenizer
from MiLo.core.quantize import *
from MiLo.models.hf.mixtral import MixtralMiLo
# from evaluation.eval_wikitext2_ppl import eval_perplexity
from datasets import load_dataset
import torch, time
import numpy as np
from tqdm import tqdm

import gc
def cleanup():
	torch.cuda.empty_cache()
	gc.collect()


model_path = "/scratch/bcjw/zshao3/huggingface/mixtral_milo/model"
lorc_dir = "/scratch/bcjw/zshao3/huggingface/mixtral_milo/lorc"


#quant_config = HqqConfig(nbits=3, group_size=64, axis=1)
ranks = {'self_attn': 0, 'experts':0}

model = MixtralMiLo.from_quantized(model_path,LoRC_weight_path=lorc_dir,
                                            LoRC_dtype = "int3_symm",
                                            ranks=ranks)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1", trust_remote_code=True)


# model.save_pretrained(quant_model)
# tokenizer.save_pretrained(quant_model)
from MiLo.utils.patching import prepare_for_inference
prepare_for_inference(model, backend="milo")


#ppl = eval_perplexity(model,tokenizer)

#Adapted from https://huggingface.co/transformers/v4.2.2/perplexity.html


def eval_wikitext2(model, tokenizer, max_length=128, stride=16, verbose=True):
	model.eval()
	tokenizer.pad_token     = tokenizer.eos_token 
	tokenizer.padding_side  = "right" 
	tokenizer.add_eos_token = False

	dataset   = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
	encodings = tokenizer('\n\n'.join(dataset['text']), return_tensors='pt')
	
	encodings['input_ids'] = encodings['input_ids'].to('cuda')

	lls, t = [], []
	for i in tqdm(range(0, encodings['input_ids'].size(1), stride), disable=not verbose):
		begin_loc  = max(i + stride - max_length, 0)
		end_loc    = min(i + stride, encodings['input_ids'].size(1))
		trg_len    = end_loc - i  
		input_ids  = encodings['input_ids'][:,begin_loc:end_loc]
		target_ids = input_ids.clone()
		target_ids[:,:-trg_len] = -100 #ignore context 
		with torch.no_grad():
			loss = model(input_ids, labels=target_ids).loss 
		log_likelihood = loss * trg_len
		lls.append(log_likelihood)

		del input_ids, target_ids

	ppl = np.round(float(torch.exp(torch.stack(lls).sum() / end_loc)), 4)
	if(verbose):
		print('perplexity', ppl)

	del encodings
	cleanup()
	
eval_wikitext2(model, tokenizer, max_length=32, stride=8, verbose=True)
