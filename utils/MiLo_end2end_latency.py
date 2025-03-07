import argparse
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MiLo.core.quantize import *
from MiLo.models.hf.mixtral import MixtralMiLo
from MiLo.utils.patching import prepare_for_inference
import torch, time
import numpy as np
from tqdm import tqdm

import gc
def cleanup():
    torch.cuda.empty_cache()
    gc.collect()

ranks = {'self_attn': 0, 'experts':0}



def test_first_token_latency(model):
    model.eval()
    batchsizes = [1, 16, 32]
    seq_len = 1
    for batch_size in batchsizes:
        input_ids = torch.randint(10000, (batch_size, seq_len)).to("cuda:0")
        for _ in range(20): #warmup
            model(input_ids = input_ids)
        start = time.time()
        for i in range(400):
            model(input_ids = input_ids)
        torch.cuda.synchronize()
        end = time.time()
        latency = np.round((end - start)/400, 3)
        print("batchsize:", batch_size)
        print('latency',str(latency)+' sec ')


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--base_dir', type=str, required=True, help="base directory to save the quantized Mixtral model")
    args = parser.parse_args()

    print(f"Start MiLo end-to-end latency evaluation on {args.base_dir}")
    quant_model_dir = f"{args.base_dir}/model"
    lorc_dir = f"{args.base_dir}/lorc"
    lorc_dtype = "int3_symm"
    ranks = {'self_attn': 0, 'experts':0}
   
    model = MixtralMiLo.from_quantized(quant_model_dir,LoRC_weight_path=lorc_dir,
                                            LoRC_dtype = lorc_dtype,
                                            ranks=ranks)

    from MiLo.utils.patching import prepare_for_inference
    backend = prepare_for_inference(model, backend="milo")
    print(backend)
    test_first_token_latency(model)
    cleanup()

if __name__ == "__main__":
    main()