# MiLo

MiLo is a MoE compression algorithm, focusing on ultra-low-bit quantization, e.g. 3-bit, with compensators aiming at a fast execution speed and high quantization quality. 

MiLo introduces the adaptive Low-rank compensators, which compensate for the error of the ultra-low-bit quantization with a minimal additional memory overhead. The adaptive low-rank compensator determines the rank according to the property of the weight, leading to a significant performance improvement.

MiLo also brings INT3 CUDA kernel, which optimizes the dequantization, GeMM, and memory pipeline to facilitate the inference.

## Features

- Optimization based quantization and compensation algorithm with fast compression speed
- Quantized low rank compensator
- Highly efficient INT3 Kernel to accelerate quantized model inference
- Adaptive rank selection strategy to suit the tradeoff between performance and memory

## Installation of Python Packages

Create a new conda environment:

```bash
conda create -n milo python=3.10
conda activate milo
```

Install dependent packages and CUDA 12.4.0 using bash scripts:

```bash
bash conda_env_setup.sh
```

Note: Make sure you have CUDA 12.4 compatible GPUs and drivers installed on your system before installation.

## Installation of kernel

```bash
bash kernel_setup.sh
```

## Quick Start
Here is an example using MiLo to quantize Mixtral-8x7B to INT3, with a dense layer (i.e. self attention) rank of 1024 and sparse layer (i.e. experts) rank of 32.

```python
from MiLo.models.hf.mixtral import MixtralMiLo as AutoMiLoHFModel
from MiLo.core.quantize import *
from transformers import AutoModelForCausalLM
import torch

device = "cuda"
quant_model_dir = "YOUR_DIR"
compress_config = BaseCompressConfig(
				# quantization config
				 nbits = 3, 
				 group_size = 64, 
				 quant_scale = False, 
				 quant_zero = False, 
				 axis = 1,
				# compensator config
				 iter = 20,
				 sparse_rank = 32,
				 dense_rank = 1024,
				 rank_strategy = None,
				 compensator_dtype  = "int3"
				 ) 
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1",
					 torch_dtype=torch.float16,
					 trust_remote_code=True)
AutoMiLoHFModel.compress_model(model, 
			   compress_config=compress_config, 
			   device=device)    
AutoMiLoHFModel.save_compressed(model, quant_model_dir)

```


## Examples

We also provide MiLo compression example to compress DeepSeek-MoE [MiLo_compress_DeepSeek.py](MiLo_compress_DeepSeek.py) at and Mixtral-8x7B at [MiLo_compress_Mixtral.py](MiLo_compress_Mixtral.py), also an evaluation script at [MiLo_eval.py](MiLo_eval.py), including Wikitext2 perplexity, zero-shot evaluation, and few-shots evaluation.

## License

MIT license

## Citation
```
@article{beichen-milo,
  title={MiLo: Efficient Quantized MoE Inference with Mixture of Low-Rank Compensators}, 
  author={Beichen Huang, Yueming Yuan, Zelei Shao, and Minjia Zhang},
  year={2025},
}
```

## Acknowledgments

This project is built on top of [HQQ](https://github.com/mobiusml/hqq), an optimization based quantiation algorithm library, and [Marlin](https://github.com/IST-DASLab/marlin), an efficient fp16xINT4 GeMM CUDA kernel. We thank the HQQ and Marlin team for providing this foundation.
