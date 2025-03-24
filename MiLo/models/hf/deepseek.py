# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################
from ..base import BasePatch
from .base import BaseMiLoHFModel
from tqdm import tqdm


# Patch  functions
class DeepSeekMoEPatch(BasePatch):
    # These tags are used to specify the parameters of each layer type. For example, if you want to give different quantization parameters to different layers
    @classmethod
    def get_linear_tags(cls):
        return [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "mlp.experts.gate_proj",
            "mlp.experts.up_proj",
            "mlp.experts.down_proj",
            "mlp.shared_experts.gate_proj",
            "mlp.shared_experts.up_proj",
            "mlp.shared_experts.down_proj",
        ]

    @classmethod
    def patch_nonlinearlayers(cls, model, patch_fct, verbose=True):
        base_model = model.model
        model.lm_head = patch_fct(model.lm_head)  ###
        base_model.embed_tokens = patch_fct(base_model.embed_tokens)
        base_model.norm = patch_fct(base_model.norm)

        layers = base_model.layers
        for i in tqdm(range(len(base_model.layers)), disable=not verbose):
            layers[i].self_attn.rotary_emb = patch_fct(layers[i].self_attn.rotary_emb)
            layers[i].input_layernorm = patch_fct(layers[i].input_layernorm)
            layers[i].post_attention_layernorm = patch_fct(
                layers[i].post_attention_layernorm
            )

            if i == 0:
                layers[0].mlp.act_fn = patch_fct(layers[0].mlp.act_fn)
            else:
                n_experts = len(layers[i].mlp.experts)
                for k in range(n_experts):
                    layers[i].mlp.experts[k].act_fn = patch_fct(
                        layers[i].mlp.experts[k].act_fn
                    )
                layers[i].mlp.shared_experts.act_fn = patch_fct(
                    layers[i].mlp.shared_experts.act_fn
                )
                layers[i].mlp.gate = patch_fct(
                    layers[i].mlp.gate
                )  # Keep MOE gate as fp16 because it's small


    @classmethod
    def patch_linearlayers(cls, model, patch_fct, patch_params, verbose=True):
        base_model = model.model
        layers = base_model.layers
        for i in tqdm(range(len(layers)), disable=not verbose):
            layers[i].self_attn.q_proj = patch_fct(
                layers[i].self_attn.q_proj, patch_params["self_attn.q_proj"]
            )
            layers[i].self_attn.k_proj = patch_fct(
                layers[i].self_attn.k_proj, patch_params["self_attn.k_proj"]
            )
            layers[i].self_attn.v_proj = patch_fct(
                layers[i].self_attn.v_proj, patch_params["self_attn.v_proj"]
            )
            layers[i].self_attn.o_proj = patch_fct(
                layers[i].self_attn.o_proj, patch_params["self_attn.o_proj"]
            )

            if i == 0:
                layers[i].mlp.gate_proj = patch_fct(
                    layers[i].mlp.gate_proj,
                    patch_params["mlp.gate_proj"],
                )
                layers[i].mlp.up_proj = patch_fct(
                    layers[i].mlp.up_proj,
                    patch_params["mlp.up_proj"],
                )
                layers[i].mlp.down_proj = patch_fct(
                    layers[i].mlp.down_proj,
                    patch_params["mlp.down_proj"],
                )
            else:
                n_experts = len(layers[i].mlp.experts)
                
                for k in range(n_experts):
                    layers[i].mlp.experts[k].gate_proj = patch_fct(
                        layers[i].mlp.experts[k].gate_proj,
                        patch_params["mlp.experts.gate_proj"],
                    )
                    layers[i].mlp.experts[k].up_proj = patch_fct(
                        layers[i].mlp.experts[k].up_proj,
                        patch_params["mlp.experts.up_proj"],
                    )
                    layers[i].mlp.experts[k].down_proj = patch_fct(
                        layers[i].mlp.experts[k].down_proj,
                        patch_params["mlp.experts.down_proj"],
                    )
                layers[i].mlp.shared_experts.gate_proj = patch_fct(
                    layers[i].mlp.shared_experts.gate_proj,
                    patch_params["mlp.shared_experts.gate_proj"],
                )
                layers[i].mlp.shared_experts.up_proj = patch_fct(
                    layers[i].mlp.shared_experts.up_proj,
                    patch_params["mlp.shared_experts.up_proj"],
                )
                layers[i].mlp.shared_experts.down_proj = patch_fct(
                    layers[i].mlp.shared_experts.down_proj,
                    patch_params["mlp.shared_experts.down_proj"],
                )


class DeepSeekMoEMiLo(DeepSeekMoEPatch, BaseMiLoHFModel):
    pass
