import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig


model_path = "/scratch/bcjw/bhuang4/zelei_test/model"
lorc_dir = "/scratch/bcjw/bhuang4/zelei_test/lorc"


quant_config = HqqConfig(nbits=3, group_size=64, axis=1)

model = AutoMiLoHFModel.from_quantized(model_path,LoRC_weight_path=lorc_dir,
                                            LoRC_dtype = lorc_dtype,
                                            ranks=ranks)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


# model.save_pretrained(quant_model)
# tokenizer.save_pretrained(quant_model)
from hqq.utils.patching import prepare_for_inference
prepare_for_inference(model, backend="milo3bitwithzero")


from eval_model import eval_wikitext2
eval_wikitext2(model, tokenizer, verbose=True)
