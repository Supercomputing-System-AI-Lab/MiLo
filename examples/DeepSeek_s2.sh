#!/bin/bash

#This is a script to use MiLo to quantize DeepSeek-moe to reproduce the results from paper Table 3 MiLo-s2
#MiLo-s2: dense layer rank 1024, avg sparse layer rank 32 setted according to expert frequency

save_dir="$1"

echo "===== MiLo Quantization ====="
python utils/MiLo_quant_main.py --base_dir ${save_dir} --model_id DeepSeek --dense_rank 1024 --sparse_rank 32 --frequency_flag 
echo "===== Wikietxt2 PPL eval ====="
python utils/MiLo_eval_wikitext2_ppl.py --base_dir ${save_dir} --model_id DeepSeek
echo "===== Zero-Shot eval ====="
python utils/MiLo_eval_zeroshot.py --base_dir ${save_dir} --model_id DeepSeek

#optional running fewshots, since MMLU and TriQA take long time
# echo "===== Zero-Shot eval ====="
# python MiLo_eval_fewshot.py --base_dr ${save_dir} --model DeepSeek
