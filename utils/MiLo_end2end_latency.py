import argparse
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation/lm_eval')))
from transformers import AutoTokenizer
from evaluation.lm_eval import evaluator
from evaluation.lm_eval.models.huggingface import HFLM
from evaluation.lm_eval.tasks import initialize_tasks
from MiLo.core.quantize import *
from transformers import AutoTokenizer
from MiLo.core.quantize import *
from MiLo.models.hf.mixtral import MixtralMiLo
from MiLo.models.hf.deepseek import DeepSeekMoEMiLo

model_path = "/scratch/bcjw/bhuang4/mixtral/noIns_myQuant_HQQ_3bit_gs64-int3_symm-iter10-u32"
lorc_dir = "/scratch/bcjw/bhuang4/HQQ_LoRC/u32-int3-symm-iter10-iter10"


#quant_config = HqqConfig(nbits=3, group_size=64, axis=1)
ranks = {'self_attn': 32, 'experts':32}

model = MixtralMiLo.from_quantized(model_path,LoRC_weight_path=lorc_dir,
                                            LoRC_dtype = "int3_symm",
                                            ranks=ranks)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1", trust_remote_code=True)


# model.save_pretrained(quant_model)
# tokenizer.save_pretrained(quant_model)
from MiLo.utils.patching import prepare_for_inference
prepare_for_inference(model, backend="milo3bitwithzero")


from evaluation.lm_eval import eval_wikitext2
eval_wikitext2(model, tokenizer, verbose=True)
