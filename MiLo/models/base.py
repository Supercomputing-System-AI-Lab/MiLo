# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################
import os
import torch
from torch import nn
from torch import float16
from os.path import join as pjoin
from typing import Callable
from tqdm import tqdm
from abc import abstractmethod
from functools import partial
from typing import Union

from huggingface_hub import snapshot_download
from ..core.utils import cleanup
from ..core.quantize import HQQLinear
from ..core.peft import PeftUtils, _HQQ_LORA_CLASSES
from ..backends.torchao import HQQLinearTorchWeightOnlynt4

from safetensors import safe_open
from safetensors.torch import save_file
import re
from torch import uint8, int32, Tensor
import pickle

_HQQ_BACKEND_CLASSES = [HQQLinearTorchWeightOnlynt4]

try:
    from ..backends.bitblas import HQQLinearBitBlas

    _HQQ_BACKEND_CLASSES.append(HQQLinearBitBlas)
except Exception:
    pass

try:
    from ..backends.marlin import MarlinLinear

    _HQQ_BACKEND_CLASSES.append(MarlinLinear)
except Exception:
    pass


# Defined what is qualified as "linear layer"
_QUANT_LAYERS = [nn.Linear, HQQLinear] + _HQQ_LORA_CLASSES + _HQQ_BACKEND_CLASSES
_IGNORE_LINEAR = ["lm_head"]


# Finds the parent of a node module named "name"
def find_parent(model, name: str) -> nn.Module:
    module_tree = name.split(".")[:-1]
    parent = model
    for m in module_tree:
        parent = parent._modules[m]
    return parent


# checks if a module is a leaf: doesn't have another module inside
def is_leaf_module(module) -> bool:
    return len(module._modules) == 0


# Get the linear_tag from a modul name. For example: model.layers.31.self_attn.k_proj -> self_attn.k_proj
def name_to_linear_tag(name: str) -> str:
    return ".".join(
        [
            n
            for n in name.split(".")
            if ((n not in ["model", "layers"]) and (not n.isnumeric()))
        ]
    )


# returns all children nodes from model
def get_all_children_from_model(model, ignore: list = []) -> list:
    tags = []
    for name, module in model.named_modules():
        if is_leaf_module(module) and (name.split(".")[-1] not in ignore):
            tags.append(name)
    return tags


# Get all linear tags available
def get_linear_tags_from_model(model, ignore: list) -> list:
    linear_tags = set()
    for name, module in model.named_modules():
        if (type(module) in _QUANT_LAYERS) and (name.split(".")[-1] not in ignore):
            linear_tags.add(name_to_linear_tag(name))
    return list(linear_tags)


def forward_device_hooked(self, *args, **kwargs):
    args = list(args)

    # eddit this to make torch.compile compatible
    for i in range(len(args)):
        if isinstance(
            args[i], (torch.Tensor, torch.nn.Parameter)
        ):  # if hasattr(args[i], "to"):
            args[i] = args[i].to(self.device)

    for i in kwargs:
        if isinstance(
            kwargs[i], (torch.Tensor, torch.nn.Parameter)
        ):  # if hasattr(kwargs[i], "to"):
            kwargs[i] = kwargs[i].to(self.device)

    # return self.__class__.forward(self, *args, **kwargs)
    return self.forward_orig(*args, **kwargs)


# Base patching class. Patching defines how nn.Linear and other layers are replaced via a patching function.
class BasePatch:
    # Override these OR override the main patch_model() function
    ############################################
    # This method iterates through layers of the model that are NOT nn.Linear and processes them via new_nodule = patch_fct(module, params)
    @classmethod
    def patch_nonlinearlayers(
        cls, model, patch_fct: Callable, verbose: bool = True
    ) -> None:
        ignore_tags = cls.get_ignore_layers(model)

        tmp_mapping = {}
        for name, module in model.named_modules():
            if (type(module) not in _QUANT_LAYERS) and (name not in ignore_tags):
                tmp_mapping[name] = module

        for name in tqdm(tmp_mapping, disable=not verbose):
            setattr(
                find_parent(model, name),
                name.split(".")[-1],
                patch_fct(tmp_mapping[name]),
            )

        cleanup()

    # This method iterates through layers of the model that are nn.Linear and processes them via new_nodule = patch_fct(module, params)
    @classmethod
    def patch_linearlayers(
        cls,
        model,
        patch_fct: Callable,
        patch_params: Union[dict, None],
        verbose: bool = True,
    ) -> None:
        ignore_tags = cls.get_ignore_layers(model)

        tmp_mapping = {}
        for name, module in model.named_modules():
            if (type(module) in _QUANT_LAYERS) and (name not in ignore_tags):
                tmp_mapping[name] = module

        for name in tqdm(tmp_mapping, disable=not verbose):
            linear_tag = name_to_linear_tag(name)
            patch_param = (
                patch_params[linear_tag] if (linear_tag in patch_params) else None
            )
            setattr(
                find_parent(model, name),
                name.split(".")[-1],
                patch_fct(tmp_mapping[name], patch_param),
            )

        cleanup()

    ############################################
    # These tags are used to specfiy parameters of the patching in patch_linearlayers()
    @classmethod
    def set_auto_linear_tags(cls, model, ignore: list = _IGNORE_LINEAR) -> None:
        if hasattr(model, "linear_tags") is False:
            linear_tags = cls.get_linear_tags()
            model.linear_tags = (
                linear_tags
                if len(linear_tags) > 0
                else get_linear_tags_from_model(model, ignore=ignore)
            )
            model.base_class = cls

    # Returns the current linear tags
    @classmethod
    def get_linear_tags(cls) -> list:
        return []

    @classmethod
    def get_ignore_layers(cls, model) -> list:
        layers = {""}
        for name, module in model.named_modules():
            if not is_leaf_module(module):
                layers.add(name)
        return list(layers)

    # Autmatically name modules. This is very important to save/load the weights
    @classmethod
    def autoname_modules(cls, model) -> None:
        for name, module in model.named_modules():
            module.name = name

    # Freeze all layers
    @classmethod
    def freeze_model(cls, model) -> None:
        for param in model.parameters():
            param.requires_grad = False
        try:
            for param in model.model.parameters():
                param.requires_grad = False
        except Exception:
            pass

    # Main patching function
    @classmethod
    def patch_model(
        cls,
        model,
        patch_nonlinear_fct: Callable,
        patch_linear_fct: Callable,
        patch_params: dict,
        verbose: bool = True,
    ) -> None:
        model.eval()
        cls.freeze_model(model)
        cls.autoname_modules(model)
        cls.patch_nonlinearlayers(model, patch_nonlinear_fct, verbose=verbose)
        cls.patch_linearlayers(model, patch_linear_fct, patch_params, verbose=verbose)
        cleanup()


class BaseHQQModel:
    # Override these
    ############################################
    # This method creates and empty model based on the specfied architecture
    @abstractmethod
    def create_model(cls, save_dir, kwargs):
        pass

    # This method saves the model architecture only without inculding the weights (for example to a config.json)
    @abstractmethod
    def cache_model(cls, model, save_dir: str):
        pass

    ############################################

    @classmethod
    def get_config_file(cls, save_dir: str) -> str:
        return pjoin(save_dir, "config.json")

    @classmethod
    def get_weight_file(cls, save_dir: str) -> str:
        return pjoin(save_dir, "qmodel.pt")

    # Save weights to disk
    @classmethod
    def save_weights(cls, weights: dict, save_dir: str) -> None:
        torch.save(weights, cls.get_weight_file(save_dir))

    # Load weights from disk
    @classmethod
    def load_weights(cls, save_dir: str, map_location=None):
        return torch.load(cls.get_weight_file(save_dir), map_location=map_location)

    # Set-up model with the necessary data
    @classmethod
    def setup_model(cls, model):
        cls.autoname_modules(model)
        cls.set_auto_linear_tags(model)

    # Main function to quantize a model. Basically goes through the linear layers specfied in the patching function and replaces them with HQQLinear
    @classmethod
    def quantize_model(
        cls,
        model,
        quant_config: dict,
        mixed_precision: bool = False,
        expert_tags = None,
        experts_quant_config = None,
        compute_dtype: torch.dtype = float16,
        device: Union[str, list, dict] = "cuda",
        lorc_path = None,
        iters: int = 0,
        iter_expert_only: bool = False,
        ranks: dict = {},
        lorc_dtype = 'int3_symm'
    ):
        # # Check if the model was already quantized
        # if getattr(model, "hqq_quantized", False):
        #     print("Model was already quantized")
        #     return

        # Set linear tags automatically
        cls.setup_model(model)

        assert (not mixed_precision) or (mixed_precision and (experts_quant_config is not None) and (expert_tags is not None))
        assert lorc_dtype in ['int8', 'int3_symm'], f"LoRC quantization [{lorc_dtype}] is not supported"

        if mixed_precision:
            print("using mixed precision quantization.")
            patch_params = {}
            for k in model.linear_tags:
                if k in expert_tags:
                    patch_params[k] = experts_quant_config
                else:
                    patch_params[k] = quant_config
            print(patch_params)

        # Use the same quantization config for all linear layers. Use None to skip quantizing a specfic layer.
        if True in [(key in model.linear_tags) for key in quant_config.keys()]:
            # If the user doesn't specify a key from get_linear_tags, the layer is not quantized via (key, None)
            patch_params = {key: None for key in model.linear_tags}
            patch_params.update(quant_config)
        else:
            # Same quant_config for all layers
            patch_params = {k: quant_config for k in model.linear_tags}

        # Get list of all nodes in order
        all_nodes = get_all_children_from_model(model, [])  # ordered nodes
        try:
            # Extract block names: This is following Hugging Face models.
            num_blocks = (
                len(model.model.layers)
                if hasattr(model, "model")
                else len(model.layers)
            )
            all_blocks = ["model.layers." + str(i) for i in range(num_blocks)]
        except Exception:
            all_blocks = None
            print(
                "Default model structure not supported. Make sure you feed device as dictionary as {name_block: device}"
            )

        if isinstance(
            device, dict
        ):  # input as {module block name (str): device (str or torch.device)}
            device_map = device
            num_devices = len(set([device_map[k] for k in device_map]))
            all_blocks = list(device_map.keys())

        node_to_block = {}
        for node in all_nodes:
            res = [block for block in all_blocks if (block in node)]
            node_to_block[node] = res[-1] if (len(res) > 0) else node

        # Set device-map
        if isinstance(device, str):  # single device as str
            device_map = {k: device for k in all_blocks + all_nodes}
            num_devices = 1

        if isinstance(device, list):  # list of devices
            num_devices = len(device)
            device_map = {}
            for node in all_nodes:
                if ".layers" in node:
                    break
                device_map[node] = device[0]

            for node in all_nodes[::-1]:
                if ".layers" in node:
                    break
                device_map[node] = device[-1]

            step, k = len(all_blocks) // num_devices, 0
            for i in range(0, len(all_blocks), step):
                for j in range(i, i + step):
                    device_map[all_blocks[min(j, len(all_blocks) - 1)]] = device[
                        min(k, num_devices - 1)
                    ]
                k += 1

        # Map nodes to block devices
        for node in all_nodes:
            device_map[node] = device_map[node_to_block[node]]

        # We replace the nn.Linear layers with HQQLinear
        def _patch_linear(linear_layer, quant_config):
            if type(linear_layer) is HQQLinear:
                return linear_layer

            current_device = device_map[linear_layer.name]

            if quant_config is not None:
                out_module = HQQLinear(
                    linear_layer,
                    quant_config,
                    compute_dtype=compute_dtype,
                    device=current_device,
                    lorc_path = lorc_path,
                    iters = (iters if ((not iter_expert_only) or ('mlp.experts' in linear_layer.name)) else 0),
                    rank = next((value for key, value in ranks.items() if key in linear_layer.name), None),
                    lorc_dtype=lorc_dtype
                )
            else:
                out_module = linear_layer.to(device=current_device, dtype=compute_dtype)

            out_module.device = current_device
            return out_module

        def _patch_other(layer):
            current_device = device_map[layer.name]
            layer.device = current_device
            return layer.to(device=current_device, dtype=compute_dtype)

        cls.patch_model(model, _patch_other, _patch_linear, patch_params)

        all_U_h_q_weight = {}
        all_U_h_q_scale = {}
        all_U_h_q_zero = {}
        all_V_h_q_weight = {}
        all_V_h_q_scale = {}
        all_V_h_q_zero = {}

        for name, module in model.named_modules():
            if isinstance(module, HQQLinear):
                UV_quantized = module.pop_UV_quantized()
                if UV_quantized is not None:
                    print(f"{name}'s UV saved, using type {lorc_dtype}")
                    if lorc_dtype == 'int3_symm':
                        (U_h_scale, U_h_q), (V_h_scale, V_h_q) = UV_quantized
                    else:
                        (U_h_scale, U_h_zero, U_h_q), (V_h_scale, V_h_zero, V_h_q) = UV_quantized
                    all_U_h_q_weight[name] = U_h_q.to('cpu')
                    all_U_h_q_scale[name] = U_h_scale.to('cpu')
                
                    all_V_h_q_weight[name] = V_h_q.to('cpu')
                    all_V_h_q_scale[name] = V_h_scale.to('cpu')
                
                    if lorc_dtype != 'int3_symm':
                        all_U_h_q_zero[name] = U_h_zero.to('cpu')
                        all_V_h_q_zero[name] = V_h_zero.to('cpu')

        save_file(all_U_h_q_weight, f"{lorc_path}/U_{lorc_dtype}_weight.safetensors")
        save_file(all_U_h_q_scale, f"{lorc_path}/U_{lorc_dtype}_scale.safetensors")
        save_file(all_V_h_q_weight, f"{lorc_path}/V_{lorc_dtype}_weight.safetensors")
        save_file(all_V_h_q_scale, f"{lorc_path}/V_{lorc_dtype}_scale.safetensors")
        if lorc_dtype != 'int3_symm':
            save_file(all_U_h_q_zero, f"{lorc_path}/U_{lorc_dtype}_zero.safetensors")
            save_file(all_V_h_q_zero, f"{lorc_path}/V_{lorc_dtype}_zero.safetensors")
        print(f">>{lorc_dtype} saved to {lorc_path}<<")

        # Insert device switcher
        if num_devices > 1:
            core_model = model if hasattr(model, "layers") else model.model

            # Make sure the input (first node) has the input in the right device during generation
            input_node_child_name = all_nodes[0].split(".")[-1]
            input_node = getattr(core_model, input_node_child_name)
            input_node.device = device_map[all_nodes[0]]
            input_node.forward_orig = input_node.forward
            input_node.forward = partial(forward_device_hooked, input_node)
            setattr(core_model, input_node_child_name, input_node)

            # Make sure all inputs to the blocks are in the right device
            for i in range(len(core_model.layers)):
                core_model.layers[i].device = device_map[core_model.layers[i].name]
                core_model.layers[i].forward_orig = core_model.layers[i].forward
                core_model.layers[i].forward = partial(
                    forward_device_hooked, core_model.layers[i]
                )

        # Set base class
        model.base_class = cls

        model.hqq_quantized = True

        return model

    # Prepares model weights by iterating through modules. It might some parameters that are NOT modules like model.param1
    @classmethod
    def serialize_weights(cls, model, verbose: bool = False) -> dict:
        weights = {}
        ignore_keys = cls.get_ignore_layers(model)
        for name, module in model.named_modules():
            if name in ignore_keys:
                continue
            try:
                # disable state_dict encoding for safetensors
                module.encoded_state_dict = False
                state_dict = module.state_dict()

                if len(state_dict) > 0:
                    weights[name] = dict(state_dict)
            except Exception:
                if verbose:
                    print("Skipping", name)

        return weights

    # Main function to save a quantized model
    @classmethod
    def save_quantized(cls, model, save_dir: str, verbose: bool = False):
        # Save config
        cls.cache_model(model, save_dir)

        # Serialization
        weights = cls.serialize_weights(model, verbose=verbose)

        # Save
        cls.save_weights(weights, save_dir)

    @classmethod
    def try_snapshot_download(
        cls, save_dir_or_hub: str, cache_dir: Union[str, None] = ""
    ):
        if cache_dir is None:
            save_dir = pjoin("", save_dir_or_hub)
        else:
            save_dir = pjoin(cache_dir, save_dir_or_hub)

        if not os.path.exists(save_dir):
            save_dir = snapshot_download(repo_id=save_dir_or_hub, cache_dir=cache_dir)
            save_dir = pjoin(save_dir)

        # Check
        if not os.path.exists(cls.get_weight_file(save_dir)):
            raise Exception("Weight file missing. Check your cache directory.")
        if not os.path.exists(cls.get_config_file(save_dir)):
            raise Exception("Config file missing. Check your cache directory.")

        return save_dir

    # This method is specfically designed in case we need to load some weights that are not part of any module
    @classmethod
    def post_module_load(cls, model, weights: dict):
        pass

    # Main function to load an HQQ quantized model from either HF hub or locally
    @classmethod
    def from_quantized(
        cls,
        save_dir_or_hub,
        compute_dtype: torch.dtype = float16,
        device="cuda",
        cache_dir: Union[str, None] = "",
        adapter: str = None,

        LoRC_dtype  = None,
        LoRC_weight_path = None,
        exp_rank = None,
        attn_rank = None,
        low_rank_only = False,
        lorc_tags = None,
        lorc_save_dir = None,
        ranks = None,
        **kwargs,
    ):
        # Get directory path
        save_dir = cls.try_snapshot_download(save_dir_or_hub, cache_dir)

        # Load model from config
        model = cls.create_model(save_dir, kwargs)

        # Track save directory
        model.save_dir = save_dir

        # Name the layers
        cls.setup_model(model)

        # Load weights
        try:
            weights = cls.load_weights(save_dir)
            if lorc_save_dir is not None:
                print(f"using partial lorc weights. tags are {lorc_tags}")
                lorc_weights = cls.load_weights(lorc_save_dir)
                for weight_name in weights:
                    if any(lorc_tag in weight_name for lorc_tag in lorc_tags):
                        weights[weight_name] = lorc_weights[weight_name]
                        print(f"{weight_name} uses lorc")
                del lorc_weights

        except Exception:
            print("Failed to load the weights")
            raise FileNotFoundError

        
            
        # load_state_dict() doesn't work with modules initialized with init_empty_weights(), so we need to do this manually
        @torch.no_grad()
        def _load_module(module, params=None):
            if module.name not in weights:
                return module.to(device=device, dtype=compute_dtype, non_blocking=True)

            state_dict = weights[module.name]
            if "W_q" in state_dict:
                module = HQQLinear(
                    linear_layer=None,
                    quant_config=None,
                    compute_dtype=compute_dtype,
                    device=device,
                )
                module.load_state_dict(state_dict)
                #LORC!!

                # U_path = "/u/bhuang4/mixtral_offloading/HQQ_LoRC/U_r8_half.safetensors"
                # V_path = "/u/bhuang4/mixtral_offloading/HQQ_LoRC/V_r8_half.safetensors"
                # with safe_open(U_path, framework="pt", device="cuda") as f:
                #     module.U = f.get_tensor(name).half()
                #     # print(f"Error is:{Error}")
                # with safe_open(V_path, framework="pt", device="cuda") as f:
                #     module.V = f.get_tensor(name).half()

            else:
                for key in state_dict:
                    setattr(
                        module,
                        key,
                        nn.Parameter(
                            state_dict[key].to(
                                device=device, dtype=compute_dtype, non_blocking=True
                            ),
                            requires_grad=False,
                        ),
                    )

            return module

        def unpack_3bit_32(W_q: Tensor, dtype=torch.int8) -> Tensor: #dtype = uint8?
            
            _step = W_q.shape[0]
            tmp = torch.empty([10 * _step, W_q.shape[1]], dtype=dtype, device="cuda")

            tmp[0 * _step : 1 * _step] = (W_q & 0b00111000000000000000000000000000) >> 27
            tmp[1 * _step : 2 * _step] = (W_q & 0b00000111000000000000000000000000) >> 24
            tmp[2 * _step : 3 * _step] = (W_q & 0b00000000111000000000000000000000) >> 21
            tmp[3 * _step : 4 * _step] = (W_q & 0b00000000000111000000000000000000) >> 18
            tmp[4 * _step : 5 * _step] = (W_q & 0b00000000000000111000000000000000) >> 15
            tmp[5 * _step : 6 * _step] = (W_q & 0b00000000000000000111000000000000) >> 12
            tmp[6 * _step : 7 * _step] = (W_q & 0b00000000000000000000111000000000) >> 9
            tmp[7 * _step : 8 * _step] = (W_q & 0b00000000000000000000000111000000) >> 6
            tmp[8 * _step : 9 * _step] = (W_q & 0b00000000000000000000000000111000) >> 3
            tmp[9 * _step : 10 * _step] = W_q & 0b00000000000000000000000000000111
            return tmp


        def UV_int3_dequantize(LoRC_weight_path, layer_name, orig_shape, UV, rank, LoRC_group_size,lorc_dtype):
            assert lorc_dtype in ['int3', 'int3_symm']
            if lorc_dtype == 'int3_symm':
                zero = 4
            else:
                with safe_open(f"{LoRC_weight_path}/{UV}_int3_zero.safetensors", framework="pt", device="cuda") as f:
                    zero = f.get_tensor(layer_name)
            with safe_open(f"{LoRC_weight_path}/{UV}_{lorc_dtype}_scale.safetensors", framework="pt", device="cuda") as f:
                scale = f.get_tensor(layer_name)
            with safe_open(f"{LoRC_weight_path}/{UV}_{lorc_dtype}_weight.safetensors", framework="pt", device="cuda") as f:
                weight = unpack_3bit_32(f.get_tensor(layer_name))

            if UV == 'U':
                weight = weight[:int(orig_shape[0] * (rank / LoRC_group_size)),:]
            else:
                weight = weight[:int(rank * orig_shape[1] / LoRC_group_size), :]

            if lorc_dtype == 'int3_symm':
                dequantized_weight = (weight - zero) * 2 * scale / 7
            else:
                dequantized_weight = (weight - zero) / scale

            if UV == 'U':
                dequantized_weight = dequantized_weight.reshape(orig_shape[0], -1)
            else:
                dequantized_weight = dequantized_weight.reshape(-1, orig_shape[1])

            return dequantized_weight.half()    

        def load_UV_int3(model, LoRC_weight_path, LoRC_group_size, ranks, lorc_dtype = 'int3'):
                if LoRC_weight_path == None: 
                    print("LoRC_weight_path is None. Not using LoRC.")
                    return
                # if model.config.model_type == "deepseek": fname = '/work/hdd/bcjw/yyuan6/hqq_lorc/sizes/deepseek_weight_size.pkl'
                # elif model.config.model_type == "mixtral": fname = '/work/hdd/bcjw/yyuan6/hqq_lorc/sizes/mixtral_weight_size.pkl'
                # with open(fname, 'rb') as f:
                #     loaded_layer_info = pickle.load(f)
                for name, module in model.named_modules():
                    if type(module) == HQQLinear: 
                        
                        # orig_shape = loaded_layer_info[name]
                        #     # print(orig_shape)
                        orig_shape=module.meta['shape']
                        # orig_shape = module.orig_shape
                        rank = next((value for key, value in ranks.items() if key in name), None)
                        if rank is not None:
                            if rank > 0:
                                module.U = UV_int3_dequantize(LoRC_weight_path, name, orig_shape, "U", rank, LoRC_group_size, lorc_dtype)
                                module.V = UV_int3_dequantize(LoRC_weight_path, name, orig_shape, "V", rank, LoRC_group_size, lorc_dtype)
                            module.name = name
                # del loaded_layer_info             
        
        # def UV_int3_dequantize(LoRC_weight_path, layer_name, orig_shape, UV, exp_rank, attn_rank, lorc_dtype):
        #     assert lorc_dtype in 
        #     if symm_flag:
        #         zero = 4
        #         with safe_open(f"{LoRC_weight_path}/{UV}_int3_symm_scale.safetensors", framework="pt", device="cuda") as f:
        #             scale = f.get_tensor(layer_name)
        #         with safe_open(f"{LoRC_weight_path}/{UV}_int3_symm_weight.safetensors", framework="pt", device="cuda") as f:
        #             weight = unpack_3bit_32(f.get_tensor(layer_name))
        #     else:
        #         with safe_open(f"{LoRC_weight_path}/{UV}_int3_zero.safetensors", framework="pt", device="cuda") as f:
        #             zero = f.get_tensor(layer_name)
        #         with safe_open(f"{LoRC_weight_path}/{UV}_int3_scale.safetensors", framework="pt", device="cuda") as f:
        #             scale = f.get_tensor(layer_name)
        #         with safe_open(f"{LoRC_weight_path}/{UV}_int3_weight.safetensors", framework="pt", device="cuda") as f:
        #             weight = unpack_3bit_32(f.get_tensor(layer_name))
        #     if UV == 'U':
        #         weight = weight[:orig_shape[0],:]
        #     else:
        #         if 'expert' in layer_name: rank = exp_rank
        #         elif 'self_attn.k_proj' in layer_name or 'self_attn.v_proj' in layer_name : rank = attn_rank
        #         elif 'self_attn.o_proj' in layer_name or 'self_attn.q_proj' in layer_name: rank = attn_rank
        #         weight = weight[:rank,:]
        #     if symm_flag:
        #         dequantized_weight = (weight - zero) * 2 * scale / 7
        #     else:
        #         dequantized_weight = (weight - zero) / scale #dequantize
        #     return dequantized_weight.half()    
            
        
        def UV_int8_dequantize(LoRC_weight_path,UV,layer_name):
            with safe_open(f"{LoRC_weight_path}/{UV}_int8_scale.safetensors", framework="pt", device="cuda") as f:
                scale = f.get_tensor(layer_name)
            with safe_open(f"{LoRC_weight_path}/{UV}_int8_zero.safetensors", framework="pt", device="cuda") as f:
                zero = f.get_tensor(layer_name)
            with safe_open(f"{LoRC_weight_path}/{UV}_int8_weight.safetensors", framework="pt", device="cuda") as f:
                weight = f.get_tensor(layer_name)
            dequantized_weight = (weight - zero) / scale #dequantize
            return dequantized_weight.half()

        @torch.no_grad()
        def load_UV_int8(model,LoRC_weight_path, low_rank_only = False):
                if LoRC_weight_path == None: 
                    print("LoRC_weight_path wrong")
                    return             
                print(low_rank_only)
                for name, module in model.named_modules():      
                    if type(module) == HQQLinear:             
                        # error_list = ['mlp']
                        lora_list = ['o_proj']
                        if lorc_tags is not None and all(lorc_tag not in name for lorc_tag in lorc_tags):
                            print(f"skipping {name}")
                            continue
                        if any(key in name for key in lora_list) or not low_rank_only:
                            parts = name.split('.')
                            if len(parts) > 2 and parts[1] == 'layers':
                                # print(name)
                                try:
                                    layer_idx = int(parts[2])
                                    if layer_idx > 100:
                                        continue
                                except ValueError:
                                    continue
                            try:
                                UV_int8_dequantize(LoRC_weight_path,'U',name)
                            except:
                                continue
                            module.U = UV_int8_dequantize(LoRC_weight_path,'U',name)    
                            module.V = UV_int8_dequantize(LoRC_weight_path,'V',name)
                            module.name = name
                            if low_rank_only:
                                print(name)

        # def load_UV_int3(model, LoRC_weight_path, low_rank_only = False, symm_flag = False):
        #         if LoRC_weight_path == None: 
        #             print("LoRC_weight_path wrong")
        #             return
                
        #         for name, module in model.named_modules():
        #             if type(module) == HQQLinear: 
        #                 if low_rank_only:
        #                     if ("w1" in name) or ("w3" in name): continue
        #                 with open('/u/bhuang4/MoE_quant/HQQ_LoRC/weight_size.pkl', 'rb') as f:
        #                     loaded_layer_info = pickle.load(f)
        #                     orig_shape = loaded_layer_info[name]
        #                     del loaded_layer_info             
        #                 module.U = UV_int3_dequantize(LoRC_weight_path,
        #                                               name,
        #                                               orig_shape,
        #                                               "U",
        #                                               exp_rank,
        #                                               attn_rank,
        #                                               symm_flag)
                        
        #                 module.V = UV_int3_dequantize(LoRC_weight_path,
        #                                               name,
        #                                               orig_shape,
        #                                               "V",
        #                                               exp_rank,
        #                                               attn_rank,
        #                                               symm_flag)
        #                 module.name = name

        #load UV
        @torch.no_grad()
        def load_UV_half(model,LoRC_weight_path):  #TO edit
            rank = re.search(r'\d{1,2}$', LoRC_weight_path).group()
            if LoRC_weight_path == None: 
                print("LoRC_weight_path wrong")
                return
            else:
                for name, module in model.named_modules():
                    if type(module) == HQQLinear:

                        with safe_open(f"{LoRC_weight_path}/U_r{rank}_half.safetensors", framework="pt", device="cuda") as f:
                            module.U = f.get_tensor(name)
                        with safe_open(f"{LoRC_weight_path}/V_r{rank}_half.safetensors", framework="pt", device="cuda") as f:
                            module.V = f.get_tensor(name)
            # if U_path != None & V_path != None:
            #     if module.name not in weights:
            #         return module.to(device=device, dtype=compute_dtype, non_blocking=True)

            #     state_dict = weights[module.name]
            #     name = module.name
            #     if "W_q" in state_dict:
            #         with safe_open(U_path, framework="pt", device="cuda") as f:
            #             module.U = f.get_tensor(name)
            #         with safe_open(V_path, framework="pt", device="cuda") as f:
            #             module.V = f.get_tensor(name)

        # Load modules
        cls.patch_model(
            model, _load_module, _load_module, {k: None for k in model.linear_tags}
        )

        #LoRC
        assert lorc_save_dir is None or ranks is not None 

        if LoRC_dtype == "half":
            load_UV_half(model, LoRC_weight_path)
        elif LoRC_dtype == "int8":
            load_UV_int8(model, LoRC_weight_path, low_rank_only=low_rank_only)
        else:
            load_UV_int3(model, LoRC_weight_path, 64, ranks, LoRC_dtype)
        

        # Load other weights that are not part of any module
        cls.post_module_load(model, weights)

        model.hqq_quantized = True

        # Set base class
        model.base_class = cls

        # Add adapter
        if adapter is not None:
            try:
                PeftUtils.load_lora_weights(model, filename=pjoin(save_dir, adapter))
                PeftUtils.cast_lora_weights(model, dtype=compute_dtype)
            except Exception as e:
                print("Skipping adapter loading...", str(e))

        return model
