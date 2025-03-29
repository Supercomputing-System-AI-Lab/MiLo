# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################
import transformers
from accelerate import init_empty_weights
from ..base import BaseMiLoModel, BasePatch


class BaseMiLoHFModel(BaseMiLoModel):
    # Save model architecture
    @classmethod
    def cache_model(cls, model, save_dir):
        model.config.save_pretrained(save_dir)

    # Create empty model from config
    @classmethod
    def create_model(cls, save_dir, kwargs):
        model_kwargs = {}
        for key in ["attn_implementation"]:
            if key in kwargs:
                model_kwargs[key] = kwargs[key]

        print(cls.get_config_file(save_dir))
        config = transformers.AutoConfig.from_pretrained(
            "deepseek-ai/deepseek-moe-16b-base" if "deepseek" in save_dir else cls.get_config_file(save_dir), trust_remote_code=True
        )

        auto_class = transformers.AutoModel

        # Todo: add support for other auto models
        archs = config.architectures
        if len(archs) == 1 and ("CausalLM" in archs[0]):
            auto_class = transformers.AutoModelForCausalLM

        with init_empty_weights():
            model = auto_class.from_config(config, **model_kwargs, trust_remote_code=True)

        return model


# Auto class used for HF models if no architecture was manually setup
class AutoMiLoHFModel(BaseMiLoHFModel, BasePatch):
    pass
