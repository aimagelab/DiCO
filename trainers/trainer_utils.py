
from transformers.utils import logging
import utils

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)


class TrainerRewardModel:
    def __init__(
        self, 
        args,
        custom_args,
        tokenizer,
    ):
        self.args = args
        self.custom_args = custom_args
        self.tokenizer = tokenizer
        self.clip_model, self.clip_preprocess = utils.load_pacs(custom_args.pacs_checkpoint)

        if self.clip_model is not None:
            self._model = self.clip_model

    
    def to(self, device_or_dtype):
        if self._model is not None:
            self._model.to(device_or_dtype)


    def eval(self):
        if self._model is not None:
            self._model.eval()                                       

    
    def compute_reward(self, generated_tokens, labels, pixel_values):
        rewards, _ = utils.compute_clip_score(
            generated_tokens, 
            labels, 
            pixel_values, 
            self.tokenizer, 
            self.clip_model, 
            w=2.0,
        )
        return rewards      


    def __call__(self, generated_tokens, labels, inputs, **kwargs):
        return self.compute_reward(generated_tokens, labels, inputs, **kwargs)
