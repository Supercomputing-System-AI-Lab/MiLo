#!/bin/bash

#This is a script to use MiLo to quantize Mixtral-8x7B to reproduce the results from paper Table 3 MiLo-s1
#MiLo-s1: dense layer rank 512, avg sparse layer rank 32 setted according to Kurtosis value


save_dir="$1"
echo "This script contain the quantization and basic experiments for Mixtral-s1 as in Table 3"
echo "Evaluation includes Wikitext2 perplexity, zero-shot evaluation with HellaSwag, Lambada and PIQA "
echo "Estimated running time: 1 hr"
echo "Estimated disk space: 22 GB"
echo "===== MiLo Quantization ====="
python utils/MiLo_quant_main.py --base_dir ${save_dir} --model_id Mixtral --dense_rank 512 --kurtosis_flag s1
echo "===== Wikietxt2 PPL eval ====="
python utils/MiLo_eval_wikitext2_ppl.py --base_dir ${save_dir} --model_id Mixtral
echo "===== Zero-Shot eval ====="
python utils/MiLo_eval_zeroshot.py --base_dir ${save_dir} --model_id Mixtral



