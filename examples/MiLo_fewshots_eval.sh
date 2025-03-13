#!/bin/bash

#This is a script to use MiLo to quantize Mixtral-8x7B to reproduce the results from paper Table 3 MiLo-s1
#MiLo-s1: dense layer rank 512, avg sparse layer rank 32 setted according to Kurtosis value


save_dir="$1"  #Directory of your quantized model
model="$2"     #DeepSeek or Mixtral

echo "This script contain the few-shots evaluation with MMLU and TriQA"
echo "Estimated running time: 10 hr"
python utils/MiLo_eval_fewshots.py --base_dr ${save_dir} --model ${model}


