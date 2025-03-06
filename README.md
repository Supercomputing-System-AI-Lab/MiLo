# MiLo

MiLo is a MoE quantization algorithm, focusing on ultra-low-bit quantization, e.g. 3-bit, with fast execution speed and high quantization quality. 

MiLo introduces the adaptive Low-rank compensators, which compensate the error of the ultra-low-bit quantization with a minimal additional memory overhead. The adaptive low-rank compensator determins the rank according to the property of the weight, leading to a significant performance improvement.

MiLo also brings INT3 CUDA kernel, which optimize the dequantization, GeMM, and memory pipeline to facilitate the inference.

## Features

- Optimization based quantization algorithm with fast execution speed
- Quantized low rank compensator
- INT3 Kernel to accelerate quantized model inference
- Adaptive rank selection strategy to suit the tradeoff between performance and memory
- Easy to use Python APIs

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

Here is an example using MiLo to quante Mixtral-8x7B to INT3, with a dense layer (i.e. self attention) rank of 1024 and sparse layer (i.e. experts) rank of 32.

```python
from MiLo.models.hf.mixtral import MixtralMiLo
from MiLo.core.quantize import *
from transformers import AutoModelForCausalLM
import torch

# Directory to save the quantized model
base_dir = YOUR_DIR
quant_model_dir = f"{base_dir}/model"
lorc_dir = f"{base_dir}/lorc"

# ==Quantization config==
quant_config = BaseQuantizeConfig(nbits=3, 
	group_size=64, 
	quant_scale=False, 
	quant_zero=False, 
	axis=1)

# ==LoRC config==
iteration = 20 
# Rank settings using a string matching mechanism. 
# The weight name containing the key of the directory is given the corresponding value from the directory, as the rank of this weight.
ranks = {'self_attn': 1024, 
	'experts':32} 
lorc_dtype = ‘int3_symm’
#load model and do MiLo
model_id = "mistralai/Mixtral-8x7B-v0.1" 
model = AutoModelForCausalLM.from_pretrained(model_id,
	torch_dtype=torch.float16)
AutoMiLoHFModel.quantize_model(model, 
	quant_config=quant_config, 
	compute_dtype=torch.float16,
	device=‘cuda:0’,
	lorc_path=lorc_dir ,
	ranks=ranks,  
	iters=iteration,
	lorc_dtype=lorc_dtype)
AutoMiLoHFModel.save_quantized(model, 
	quant_model_dir)
```







## Examples

For detailed usage examples and tutorials, please refer to the [examples](examples/) in the repository.

## License

MIT license

## Acknowledgments

This project is built on top of [HQQ]([https://github.com/rapidsai/cuvs](https://github.com/mobiusml/hqq)), an optimization based quantiation algorithm library, and [Marlin](https://github.com/IST-DASLab/marlin), a efficient fp16xINT4 GeMM CUDA kernel. We thank the HQQ and Marlin team for providing this foundation.
