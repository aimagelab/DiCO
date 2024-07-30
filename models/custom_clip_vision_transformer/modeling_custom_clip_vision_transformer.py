import torch
from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled
from transformers.models.clip.modeling_clip import CLIPVisionTransformer
from transformers.utils import logging

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)


class CustomCLIPVisionTransformer(CLIPVisionTransformer):
    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.

        Args:
            torch_dtype (`torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype.
        """
        torch_dtype = kwargs.pop("torch_dtype", None)

        # override default dtype if needed
        dtype_orig = None
        if torch_dtype is not None:
            dtype_orig = cls._set_default_torch_dtype(torch_dtype)

        if is_deepspeed_zero3_enabled():
            import deepspeed

            logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
            # this immediately partitions the model across all gpus, to avoid the overhead in time
            # and memory copying it on CPU or each GPU first
            with deepspeed.zero.Init(config_dict_or_path=deepspeed_config()):
                model = cls(config)
        else:
            model = cls(config)

        # restore default dtype if it was modified
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        return model