# coding=utf-8
from typing import Optional
import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from . import VisionEncoderEncoderConfig

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)


class VisionEncoderEncoderModel(PreTrainedModel):
    config_class = VisionEncoderEncoderConfig
    base_model_prefix = "vision_encoder_encoder"
    main_input_name = "pixel_values"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        vision_encoder: Optional[PreTrainedModel] = None,
        encoder: Optional[PreTrainedModel] = None,
    ):
        if config is None and (vision_encoder is None or encoder is None):
            raise ValueError("Either a configuration or a vision encoder and an encoder has to be provided.")
        if config is None:
            config = VisionEncoderEncoderConfig.from_vision_encoder_encoder_configs(vision_encoder.config,
                                                                                    encoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        # initialize with config
        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        super().__init__(config)

        if vision_encoder is None:
            vision_encoder = AutoModel.from_config(config.vision_encoder)
        
        for param in vision_encoder.parameters():
            param.requires_grad = False
        vision_encoder.eval()

        if encoder is None:
            encoder = AutoModel.from_config(config.encoder)

        self.vision_encoder = vision_encoder
        self.encoder = encoder
        self.encoder.main_input_name = self.main_input_name

        if self.vision_encoder.config.to_dict() != self.config.vision_encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.vision_encoder.__class__} is overwritten by shared encoder config: {self.config.vision_encoder}"
            )
        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config: {self.config.encoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.vision_encoder.config = self.config.vision_encoder
        self.encoder.config = self.config.encoder
        # vision encoder outputs might need to be projected to different dimension for encoder
        if (
            self.vision_encoder.config.hidden_size != self.encoder.config.hidden_size
            and self.encoder.config.cross_attention_hidden_size is None
        ):
            self.vis_enc_to_enc_proj = nn.Linear(self.vision_encoder.config.hidden_size, self.encoder.config.hidden_size)

        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )
        
        self.require_vis_enc_to_enc_proj = self.vision_encoder.config.hidden_size != self.encoder.config.hidden_size


    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for VisionEncoderEncoderModel. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_vision_encoder_encoder_pretrained(
        cls,
        vision_encoder_pretrained_model_name_or_path: str = None,
        encoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs
    ) -> PreTrainedModel:
        kwargs_vision_encoder = {
            argument[len("vision_encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("vision_encoder_")
        }

        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        # remove vision encoder and encoder kwargs from kwargs
        for key in kwargs_vision_encoder.keys():
            del kwargs["vision_encoder_" + key]
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]

        # Load and initialize the vision encoder and encoder
        # The distinction between vision encoder and encoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        vision_encoder = kwargs_vision_encoder.pop("model", None)
        if vision_encoder is None:
            if vision_encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `vision_encoder_model` is not defined as an argument, a `vision_encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_vision_encoder:
                vision_encoder_config, kwargs_vision_encoder = AutoConfig.from_pretrained(
                    vision_encoder_pretrained_model_name_or_path, **kwargs_vision_encoder, return_unused_kwargs=True
                )

                if vision_encoder_config.is_decoder is True or vision_encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {vision_encoder_pretrained_model_name_or_path} as a vision_encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    vision_encoder_config.is_decoder = False
                    vision_encoder_config.add_cross_attention = False

                kwargs_vision_encoder["config"] = vision_encoder_config

            vision_encoder = AutoModel.from_pretrained(vision_encoder_pretrained_model_name_or_path, *model_args, **kwargs_vision_encoder)
        
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
                )

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)

        # instantiate config with corresponding kwargs
        config = VisionEncoderEncoderConfig.from_vision_encoder_encoder_configs(vision_encoder.config,
                                                                                encoder.config, **kwargs)

        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        return cls(vision_encoder=vision_encoder, encoder=encoder, config=config)

    def forward(
        self,
        pixel_values=None,
        vision_encoder_outputs=None,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        """
        Return:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_vision_encoder = {argument: value for argument, value in kwargs.items() if argument.startswith("vision_encoder_")
                                 and argument != 'gts'}
        
        kwargs_encoder = {argument: value for argument, value in kwargs.items() if argument.startswith("encoder_")
                          and argument != 'gts'}

        if vision_encoder_outputs is None:
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            self.vision_encoder.eval()
            with torch.no_grad():
                vision_encoder_outputs = self.vision_encoder(
                    pixel_values,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs_vision_encoder,
            )
        elif isinstance(vision_encoder_outputs, tuple):
            vision_encoder_outputs = BaseModelOutput(*vision_encoder_outputs)
        else:
            vision_encoder_outputs = (vision_encoder_outputs, )

        vision_encoder_hidden_states = vision_encoder_outputs[0]

        # optionally project vision_encoder_hidden_states
        if self.require_vis_enc_to_enc_proj and self.encoder.config.cross_attention_hidden_size is None:
            vision_encoder_hidden_states = vision_encoder_hidden_states.type(self.vis_enc_to_enc_proj.weight.dtype)
            vision_encoder_hidden_states = self.vis_enc_to_enc_proj(vision_encoder_hidden_states)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=vision_encoder_hidden_states,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
        )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        return encoder_outputs
