import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig


logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)


class VisionEncoderEncoderConfig(PretrainedConfig):
    model_type = "vision-encoder-encoder"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "vision_encoder" not in kwargs or "encoder" not in kwargs:
            raise ValueError(
                f"A configuration of type {self.model_type} cannot be instantiated because "
                f"not both `vision_encoder` and `encoder` sub-configurations are passed, but only {kwargs}"
            )

        vision_encoder_config = kwargs.pop("vision_encoder")
        vision_encoder_model_type = vision_encoder_config.pop("model_type")
        encoder_config = kwargs.pop("encoder")
        encoder_model_type = encoder_config.pop("model_type")

        self.vision_encoder = AutoConfig.for_model(vision_encoder_model_type, **vision_encoder_config)
        self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
        self.is_encoder_decoder = False
        self.hidden_size = self.encoder.d_embed

    @classmethod
    def from_vision_encoder_encoder_configs(
        cls, vision_encoder_config: PretrainedConfig, encoder_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        return cls(vision_encoder=vision_encoder_config.to_dict(), encoder=encoder_config.to_dict(), **kwargs)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["vision_encoder"] = self.vision_encoder.to_dict()
        output["encoder"] = self.encoder.to_dict()
        output["model_type"] = self.__class__.model_type
        return output