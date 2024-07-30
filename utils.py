from transformers import AutoTokenizer, LogitsProcessorList, StoppingCriteriaList, Constraint, BeamSearchScorer, DisjunctiveConstraint, PhrasalConstraint, \
    ConstrainedBeamSearchScorer, BeamScorer, CLIPModel, CLIPFeatureExtractor
import nltk
import evaluate
import torch
import inspect
import json
import os
import uuid
import warnings
from collections import UserDict
from pathlib import Path
from typing import List, Dict, Optional, Union, Iterable, Callable, Tuple, NamedTuple
from tqdm import tqdm
import copy
from torch.cuda.amp import autocast
from transformers import CLIPFeatureExtractor, CLIPModel
import numpy as np
import torch.distributed as dist
from PIL import Image
from transformers.generation_beam_search import BeamHypotheses
from evaluation import clip
from transformers.generation_utils import GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, logger, BeamSearchEncoderDecoderOutput

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


METRICS = {
    'bleu': 'evaluation/bleu/bleu.py',
    'meteor': 'evaluation/meteor/meteor.py',
    'rouge': 'evaluation/rouge/rouge.py',
    'cider': 'evaluation/cider/cider_huggingface.py',
}


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def compute_metrics(eval_preds, tokenizer, do_train=False):
    metrics_results = dict()

    # default returns predictions, labels and img keys
    predictions, labels = eval_preds
    labels[labels == -100] = tokenizer.pad_token_id
    labels = labels.swapaxes(1, -1)

    decoded_predictions = tokenizer.batch_decode(
        predictions, skip_special_tokens=True)
    decoded_references = [tokenizer.batch_decode(
        label, skip_special_tokens=True) for label in labels]
    processed_predictions, processed_references = postprocess_text(
        decoded_predictions, decoded_references)

    metrics = METRICS
    if not do_train:
        metrics.update({'spice': 'evaluation/spice/spice_huggingface.py'})

    cache_dir = None

    for metric_name, metric_script in metrics.items():
        metric = evaluate.load(metric_script, experiment_id=str(
            uuid.uuid4()), cache_dir=cache_dir)

        if metric_name == 'meteor':
            metric_result = metric.compute(predictions=[prediction[0] for prediction in processed_predictions],
                                           references=processed_references, alpha=0.85, beta=0.2, gamma=0.6)
            metric_result['meteor'] = metric_result['meteor'].mean()
        elif metric_name in ['bleu', 'rouge']:
            metric_result = metric.compute(predictions=[prediction[0] for prediction in processed_predictions],
                                           references=processed_references)

            if metric_result is not None:
                if metric_name == 'bleu':
                    metric_result.update({
                        'bleu_1': metric_result['precisions'][0],
                        'bleu_2': metric_result['precisions'][1],
                        'bleu_3': metric_result['precisions'][2],
                        'bleu_4': metric_result['precisions'][3],
                    })
                    del metric_result['precisions']

                if metric_name == 'rouge':
                    metric_result = {'rouge': metric_result['rougeL']}
        else:
            metric_result = metric.compute(
                predictions=processed_predictions, references=processed_references)

        if metric_result is not None:
            metrics_results.update(metric_result)

    metrics_results = {name: value for name,
                       value in metrics_results.items() if value is not None}

    return metrics_results


@torch.enable_grad()
def generate_with_backpropagation(
    self,
    inputs: Optional[torch.Tensor] = None,
    max_length: Optional[int] = None,
    min_length: Optional[int] = None,
    do_sample: Optional[bool] = None,
    early_stopping: Optional[bool] = None,
    num_beams: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    typical_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    bad_words_ids: Optional[Iterable[int]] = None,
    force_words_ids: Optional[Union[Iterable[int],
                                    Iterable[Iterable[int]]]] = None,
    bos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    length_penalty: Optional[float] = None,
    no_repeat_ngram_size: Optional[int] = None,
    encoder_no_repeat_ngram_size: Optional[int] = None,
    num_return_sequences: Optional[int] = None,
    max_time: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    decoder_start_token_id: Optional[int] = None,
    use_cache: Optional[bool] = None,
    num_beam_groups: Optional[int] = None,
    diversity_penalty: Optional[float] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[
        int, torch.Tensor], List[int]]] = None,
    logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
    renormalize_logits: Optional[bool] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
    constraints: Optional[List[Constraint]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    forced_bos_token_id: Optional[int] = None,
    forced_eos_token_id: Optional[int] = None,
    remove_invalid_values: Optional[bool] = None,
    synced_gpus: Optional[bool] = False,
    exponential_decay_length_penalty: Optional[Tuple[Union[int, float]]] = None,
    use_vision_encoder_outputs: Optional[bool] = None,
    is_train: bool = True,
    **model_kwargs,
) -> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:
    r"""

    Generates sequences of token ids for models with a language modeling head. The method supports the following
    generation methods for text-decoder, text-to-text, speech-to-text, and vision-to-text models:

        - *greedy decoding* by calling [`~generation_utils.GenerationMixin.greedy_search`] if `num_beams=1` and
          `do_sample=False`.
        - *multinomial sampling* by calling [`~generation_utils.GenerationMixin.sample`] if `num_beams=1` and
          `do_sample=True`.
        - *beam-search decoding* by calling [`~generation_utils.GenerationMixin.beam_search`] if `num_beams>1` and
          `do_sample=False`.
        - *beam-search multinomial sampling* by calling [`~generation_utils.GenerationMixin.beam_sample`] if
          `num_beams>1` and `do_sample=True`.
        - *diverse beam-search decoding* by calling [`~generation_utils.GenerationMixin.group_beam_search`], if
          `num_beams>1` and `num_beam_groups>1`.
        - *constrained beam-search decoding* by calling
          [`~generation_utils.GenerationMixin.constrained_beam_search`], if `constraints!=None` or
          `force_words_ids!=None`.

    <Tip warning={true}>

    Apart from `inputs`, all the arguments below will default to the value of the attribute of the same name as
    defined in the model's config (`config.json`) which in turn defaults to the
    [`~modeling_utils.PretrainedConfig`] of the model.

    </Tip>

    Most of these parameters are explained in more detail in [this blog
    post](https://huggingface.co/blog/how-to-generate).

    Parameters:
        inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
            The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
            method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
            should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
            `input_ids`, `input_values`, `input_features`, or `pixel_values`.
        max_length (`int`, *optional*, defaults to `model.config.max_length`):
            The maximum length of the sequence to be generated.
        max_new_tokens (`int`, *optional*, defaults to None):
            The maximum numbers of tokens to generate, ignore the current number of tokens. Use either
            `max_new_tokens` or `max_length` but not both, they serve the same purpose.
        min_length (`int`, *optional*, defaults to 10):
            The minimum length of the sequence to be generated.
        do_sample (`bool`, *optional*, defaults to `False`):
            Whether or not to use sampling ; use greedy decoding otherwise.
        early_stopping (`bool`, *optional*, defaults to `False`):
            Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
        num_beams (`int`, *optional*, defaults to 1):
            Number of beams for beam search. 1 means no beam search.
        temperature (`float`, *optional*, defaults to 1.0):
            The value used to module the next token probabilities.
        top_k (`int`, *optional*, defaults to 50):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (`float`, *optional*, defaults to 1.0):
            If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or higher
            are kept for generation.
        typical_p (`float`, *optional*, defaults to 1.0):
            The amount of probability mass from the original distribution to be considered in typical decoding. If
            set to 1.0 it takes no effect. See [this paper](https://arxiv.org/pdf/2202.00666.pdf) for more details.
        repetition_penalty (`float`, *optional*, defaults to 1.0):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        bos_token_id (`int`, *optional*):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`int`, *optional*):
            The id of the *end-of-sequence* token.
        length_penalty (`float`, *optional*, defaults to 1.0):
             Exponential penalty to the length. 1.0 means that the beam score is penalized by the sequence length.
             0.0 means no penalty. Set to values < 0.0 in order to encourage the model to generate longer
             sequences, to a value > 0.0 in order to encourage the model to produce shorter sequences.
        no_repeat_ngram_size (`int`, *optional*, defaults to 0):
            If set to int > 0, all ngrams of that size can only occur once.
        encoder_no_repeat_ngram_size (`int`, *optional*, defaults to 0):
            If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the
            `decoder_input_ids`.
        bad_words_ids(`List[List[int]]`, *optional*):
            List of token ids that are not allowed to be generated. In order to get the token ids of the words that
            should not appear in the generated text, use `tokenizer(bad_words, add_prefix_space=True,
            add_special_tokens=False).input_ids`.
        force_words_ids(`List[List[int]]` or `List[List[List[int]]]`, *optional*):
            List of token ids that must be generated. If given a `List[List[int]]`, this is treated as a simple
            list of words that must be included, the opposite to `bad_words_ids`. If given `List[List[List[int]]]`,
            this triggers a [disjunctive constraint](https://github.com/huggingface/transformers/issues/14081),
            where one can allow different forms of each word.
        num_return_sequences(`int`, *optional*, defaults to 1):
            The number of independently computed returned sequences for each element in the batch.
        max_time(`float`, *optional*, defaults to None):
            The maximum amount of time you allow the computation to run for in seconds. generation will still
            finish the current pass after allocated time has been passed.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values are in `[0, 1]`, 1 for tokens
            that are not masked, and 0 for masked tokens. If not provided, will default to a tensor the same shape
            as `input_ids` that masks the pad token. [What are attention masks?](../glossary#attention-mask)
        decoder_start_token_id (`int`, *optional*):
            If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
        use_cache: (`bool`, *optional*, defaults to `True`):
            Whether or not the model should use the past last key/values attentions (if applicable to the model) to
            speed up decoding.
        num_beam_groups (`int`, *optional*, defaults to 1):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of
            beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
        diversity_penalty (`float`, *optional*, defaults to 0.0):
            This value is subtracted from a beam's score if it generates a token same as any beam from other group
            at a particular time. Note that `diversity_penalty` is only effective if `group beam search` is
            enabled.
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
            If provided, this function constraints the beam search to allowed tokens only at each step. If not
            provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
            `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
            on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
            for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
            Retrieval](https://arxiv.org/abs/2010.00904).
        logits_processor (`LogitsProcessorList`, *optional*):
             Custom logits processors that complement the default logits processors built from arguments and a
             model's config. If a logit processor is passed that is already created with the arguments or a model's
             config an error is thrown. This feature is intended for advanced users.
        renormalize_logits: (`bool`, *optional*, defaults to `False`):
            Whether to renormalize the logits after applying all the logits processors or warpers (including the
            custom ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the
            score logits are normalized but some logit processors or warpers break the normalization.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
             Custom stopping criteria that complement the default stopping criteria built from arguments and a
             model's config. If a stopping criteria is passed that is already created with the arguments or a
             model's config an error is thrown. This feature is intended for advanced users.
        constraints (`List[Constraint]`, *optional*):
             Custom constraints that can be added to the generation to ensure that the output will contain the use
             of certain tokens as defined by `Constraint` objects, in the most sensible way possible.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        forced_bos_token_id (`int`, *optional*):
            The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful
            for multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be
            the target language token.
        forced_eos_token_id (`int`, *optional*):
            The id of the token to force as the last generated token when `max_length` is reached.
        remove_invalid_values (`bool`, *optional*):
            Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to
            crash. Note that using `remove_invalid_values` can slow down generation.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        exponential_decay_length_penalty (`tuple(int, float)`, *optional*):
            This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been
            generated. The tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates
            where penalty starts and `decay_factor` represents the factor of exponential decay

        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If the model
            is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific kwargs
            should be prefixed with *decoder_*.

    Return:
        [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
        or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

            If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation_utils.GreedySearchDecoderOnlyOutput`],
                - [`~generation_utils.SampleDecoderOnlyOutput`],
                - [`~generation_utils.BeamSearchDecoderOnlyOutput`],
                - [`~generation_utils.BeamSampleDecoderOnlyOutput`]

            If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation_utils.GreedySearchEncoderDecoderOutput`],
                - [`~generation_utils.SampleEncoderDecoderOutput`],
                - [`~generation_utils.BeamSearchEncoderDecoderOutput`],
                - [`~generation_utils.BeamSampleEncoderDecoderOutput`]

    Examples:

    Greedy Decoding:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

    >>> prompt = "Today I believe we can finally"
    >>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    >>> # generate up to 30 tokens
    >>> outputs = model.generate_with_backpropagation(input_ids, do_sample=False, max_length=30)
    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ['Today I believe we can finally get to the point where we can make a difference in the lives of the people of the United States of America.\n']
    ```

    Multinomial Sampling:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

    >>> prompt = "Today I believe we can finally"
    >>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    >>> # sample up to 30 tokens
    >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
    >>> outputs = model.generate_with_backpropagation(input_ids, do_sample=True, max_length=30)
    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ['Today I believe we can finally get rid of discrimination," said Rep. Mark Pocan (D-Wis.).\n\n"Just look at the']
    ```

    Beam-search decoding:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    >>> tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")

    >>> sentence = "Paris is one of the densest populated areas in Europe."
    >>> input_ids = tokenizer(sentence, return_tensors="pt").input_ids

    >>> outputs = model.generate_with_backpropagation(input_ids)
    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ['Paris ist eines der dichtesten besiedelten Gebiete Europas.']
    ```"""
    if is_train:
        torch.set_grad_enabled(True)
        if not use_vision_encoder_outputs:
            inputs.requires_grad = True

    # 1. Set generation parameters if not already defined
    bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
    num_beams = num_beams if num_beams is not None else self.config.num_beams
    length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
    early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
    num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
    do_sample = do_sample if do_sample is not None else self.config.do_sample
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
    )

    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

    if eos_token_id is None and hasattr(self.config, "decoder"):
        eos_token_id = self.config.decoder.eos_token_id

    if pad_token_id is None and eos_token_id is not None:
        if model_kwargs.get("attention_mask", None) is None:
            logger.warning(
                "The attention mask and the pad token id were not set. As a consequence, you may observe "
                "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
            )
        logger.warning(
            f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        pad_token_id = eos_token_id

    output_scores = output_scores if output_scores is not None else self.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
    )

    # 2. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, bos_token_id, model_kwargs)
    batch_size = inputs_tensor.shape[0]

    # 3. Define other model kwargs
    model_kwargs["output_attentions"] = output_attentions
    model_kwargs["output_hidden_states"] = output_hidden_states
    model_kwargs["use_cache"] = use_cache

    accepts_attention_mask = "attention_mask" in set(
        inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, pad_token_id, eos_token_id
        )

    if use_vision_encoder_outputs:
        model_input_name = 'vision_encoder_outputs'

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created
        # and added to `model_kwargs`
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )
        # WORKAROUND
        if hasattr(self, 'is_ensemble') and self.is_ensemble:
            model_kwargs[model_input_name] = inputs_tensor

    # 4. Prepare `input_ids` which will be used for auto-regressive generation
    if self.config.is_encoder_decoder:
        input_ids = self._prepare_decoder_input_ids_for_generation(
            batch_size,
            decoder_start_token_id=decoder_start_token_id,
            bos_token_id=bos_token_id,
            model_kwargs=model_kwargs,
            device=inputs_tensor.device,
        )
    else:
        # if decoder-only then inputs_tensor has to be `input_ids`
        input_ids = inputs_tensor
    input_ids_seq_length = input_ids.shape[-1]

    # 5. Prepare `max_length` depending on other stopping criteria
    # if `max_new_tokens` is passed, but not `max_length` -> set `max_length = max_new_tokens`
    if max_length is None and max_new_tokens is not None:
        max_length = max_new_tokens + input_ids_seq_length
    elif max_length is not None and max_new_tokens is not None:
        # Both are set, this is odd, raise a warning
        warnings.warn(
            "Both `max_length` and `max_new_tokens` have been set "
            f"but they serve the same purpose. `max_length` {max_length} "
            f"will take priority over `max_new_tokens` {max_new_tokens}.",
            UserWarning,
        )
    # default to config if still None
    max_length = max_length if max_length is not None else self.config.max_length
    min_length = min_length if min_length is not None else self.config.min_length

    if min_length is not None and min_length > max_length:
        raise ValueError(
            f"Unfeasable length constraints: the minimum length ({min_length}) is larger than the maximum "
            f"length ({max_length})"
        )
    if input_ids_seq_length >= max_length:
        input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, but ``max_length`` is set to"
            f" {max_length}. This can lead to unexpected behavior. You should consider increasing"
            " ``config.max_length`` or ``max_length``."
        )

    # 6. determine generation mode
    is_constraint_gen_mode = constraints is not None or force_words_ids is not None
    is_greedy_gen_mode = (
        (num_beams == 1) and (num_beam_groups ==
                              1) and do_sample is False and not is_constraint_gen_mode
    )
    is_sample_gen_mode = (
        (num_beams == 1) and (num_beam_groups ==
                              1) and do_sample is True and not is_constraint_gen_mode
    )
    is_beam_gen_mode = (
        (num_beams > 1) and (num_beam_groups ==
                             1) and do_sample is False and not is_constraint_gen_mode
    )
    is_beam_sample_gen_mode = (
        (num_beams > 1) and (num_beam_groups ==
                             1) and do_sample is True and not is_constraint_gen_mode
    )
    is_group_beam_gen_mode = (num_beams > 1) and (
        num_beam_groups > 1) and not is_constraint_gen_mode

    if num_beam_groups > num_beams:
        raise ValueError(
            "`num_beam_groups` has to be smaller or equal to `num_beams`")
    if is_group_beam_gen_mode and do_sample is True:
        raise ValueError(
            "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
        )

    # 7. prepare distribution pre_processing samplers
    logits_processor = self._get_logits_processor(
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
        input_ids_seq_length=input_ids_seq_length,
        # shouldn't break if inputs_tensor is `vision_encoder_outputs`
        encoder_input_ids=inputs_tensor,
        bad_words_ids=bad_words_ids,
        min_length=min_length,
        max_length=max_length,
        eos_token_id=eos_token_id,
        forced_bos_token_id=forced_bos_token_id,
        forced_eos_token_id=forced_eos_token_id,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
        remove_invalid_values=remove_invalid_values,
        exponential_decay_length_penalty=exponential_decay_length_penalty,
        logits_processor=logits_processor,
        renormalize_logits=renormalize_logits,
    )

    # 8. prepare stopping criteria
    stopping_criteria = self._get_stopping_criteria(
        max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
    )

    # 9. go into different generation modes
    if is_greedy_gen_mode:
        if num_return_sequences > 1:
            raise ValueError(
                f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
            )

        # 10. run greedy search
        return self.greedy_search(
            input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif is_sample_gen_mode:
        # 10. prepare logits warper
        logits_warper = self._get_logits_warper(
            top_k=top_k,
            top_p=top_p,
            typical_p=typical_p,
            temperature=temperature,
            num_beams=num_beams,
            renormalize_logits=renormalize_logits,
        )

        # 11. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids,
            expand_size=num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 12. run sample
        return self.sample(
            input_ids,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif is_beam_gen_mode:
        if num_return_sequences > num_beams:
            raise ValueError(
                "`num_return_sequences` has to be smaller or equal to `num_beams`.")

        if stopping_criteria.max_length is None:
            raise ValueError(
                "`max_length` needs to be a stopping_criteria for now.")

        # 10. prepare beam search scorer
        beam_scorer = CustomBeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=inputs_tensor.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
            num_beam_hyps_to_keep=num_return_sequences,
        )
        # 11. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
        )

        # WORKAROUND
        if hasattr(self, 'is_ensemble') and self.is_ensemble:
            del model_kwargs['encoder_outputs']

        # 12. run beam search
        return self.beam_search(
            input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif is_beam_sample_gen_mode:
        # 10. prepare logits warper
        logits_warper = self._get_logits_warper(
            top_k=top_k,
            top_p=top_p,
            typical_p=typical_p,
            temperature=temperature,
            num_beams=num_beams,
            renormalize_logits=renormalize_logits,
        )

        if stopping_criteria.max_length is None:
            raise ValueError(
                "`max_length` needs to be a stopping_criteria for now.")
        # 11. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size * num_return_sequences,
            num_beams=num_beams,
            device=inputs_tensor.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
        )

        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids,
            expand_size=num_beams * num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 13. run beam sample
        return self.beam_sample(
            input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif is_group_beam_gen_mode:
        if num_return_sequences > num_beams:
            raise ValueError(
                "`num_return_sequences` has to be smaller or equal to `num_beams`.")

        if num_beams % num_beam_groups != 0:
            raise ValueError(
                "`num_beams` should be divisible by `num_beam_groups` for group beam search.")

        if stopping_criteria.max_length is None:
            raise ValueError(
                "`max_length` needs to be a stopping_criteria for now.")

        # 10. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            max_length=stopping_criteria.max_length,
            device=inputs_tensor.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
            num_beam_hyps_to_keep=num_return_sequences,
            num_beam_groups=num_beam_groups,
        )
        # 11. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
        )
        # 12. run beam search
        return self.group_beam_search(
            input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif is_constraint_gen_mode:
        if num_return_sequences > num_beams:
            raise ValueError(
                "`num_return_sequences` has to be smaller or equal to `num_beams`.")

        if stopping_criteria.max_length is None:
            raise ValueError(
                "`max_length` needs to be a stopping_criteria for now.")

        if num_beams <= 1:
            raise ValueError(
                "`num_beams` needs to be greater than 1 for constrained genertation.")

        if do_sample:
            raise ValueError(
                "`do_sample` needs to be false for constrained generation.")

        if num_beam_groups is not None and num_beam_groups > 1:
            raise ValueError(
                "`num_beam_groups` not supported yet for constrained generation.")

        final_constraints = []
        if constraints is not None:
            final_constraints = constraints

        if force_words_ids is not None:

            def typeerror():
                raise ValueError(
                    "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]`"
                    f"of positive integers, but is {force_words_ids}."
                )

            if not isinstance(force_words_ids, list) or len(force_words_ids) == 0:
                typeerror()

            for word_ids in force_words_ids:
                if isinstance(word_ids[0], list):
                    if not isinstance(word_ids, list) or len(word_ids) == 0:
                        typeerror()
                    if any(not isinstance(token_ids, list) for token_ids in word_ids):
                        typeerror()
                    if any(
                        any((not isinstance(token_id, int) or token_id < 0)
                            for token_id in token_ids)
                        for token_ids in word_ids
                    ):
                        typeerror()

                    constraint = DisjunctiveConstraint(word_ids)
                else:
                    if not isinstance(word_ids, list) or len(word_ids) == 0:
                        typeerror()
                    if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                        typeerror()

                    constraint = PhrasalConstraint(word_ids)
                final_constraints.append(constraint)

        # 10. prepare beam search scorer
        constrained_beam_scorer = ConstrainedBeamSearchScorer(
            constraints=final_constraints,
            batch_size=batch_size,
            num_beams=num_beams,
            device=inputs_tensor.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
            num_beam_hyps_to_keep=num_return_sequences,
        )
        # 11. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
        )
        # 12. run beam search
        return self.constrained_beam_search(
            input_ids,
            constrained_beam_scorer=constrained_beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )


class CustomBeamSearchScorer(BeamScorer):
    r"""
    [`BeamScorer`] implementing standard beam search decoding.

    Adapted in part from [Facebook's XLM beam search
    code](https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529).

    Reference for the diverse beam search algorithm and implementation [Ashwin Kalyan's DBS
    implementation](https://github.com/ashwinkalyan/dbs/blob/master/dbs/beam_utils.lua)

    Args:
        batch_size (`int`):
            Batch Size of `input_ids` for which standard beam search decoding is run in parallel.
        max_length (`int`):
            The maximum length of the sequence to be generated.
        num_beams (`int`):
            Number of beams for beam search.
        device (`torch.device`):
            Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be
            allocated.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
            model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
            sequences.
        do_early_stopping (`bool`, *optional*, defaults to `False`):
            Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
        num_beam_hyps_to_keep (`int`, *optional*, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            [`~transformer.BeamSearchScorer.finalize`].
        num_beam_groups (`int`):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
    """

    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[bool] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        **kwargs,
    ):
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
            )
            for _ in range(batch_size)
        ]
        self._done = torch.tensor([False for _ in range(
            batch_size)], dtype=torch.bool, device=self.device)

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1,"
                " one should make use of `greedy_search` instead."
            )

        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                "`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` has to be"
                f" divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

        if "max_length" in kwargs:
            warnings.warn(
                "Passing `max_length` to BeamSearchScorer is deprecated and has no effect. "
                "`max_length` should be passed directly to `beam_search(...)`, `beam_sample(...)`"
                ", or `group_beam_search(...)`."
            )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        beam_indices: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        if not (batch_size == (input_ids.shape[0] // self.group_size)):
            if self.num_beam_groups > 1:
                raise ValueError(
                    f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                    f"size of {self.group_size} is expected by the beam scorer."
                )
            else:
                raise ValueError(
                    f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                    f"{self.group_size} is expected by the beam scorer."
                )

        device = input_ids.device
        next_beam_scores = torch.zeros(
            (batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros(
            (batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros(
            (batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                if self.num_beams < len(beam_hyp):
                    raise ValueError(
                        f"Batch can only be done if at least {self.num_beams} beams have been generated")
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError(
                        "Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx],
                    next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    if beam_indices is not None:
                        beam_index = beam_indices[batch_beam_idx]
                        beam_index = beam_index + (next_index,)
                    else:
                        beam_index = None

                    beam_hyp.add(
                        input_ids[batch_beam_idx].clone(),
                        next_score,  # Workaround to back propagate the gradient
                        # next_score.item(),
                        beam_indices=beam_index,
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id:"
                    f" {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        beam_indices: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.LongTensor]:
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                # Workaround to back propagate the gradient
                # final_score = final_beam_scores[batch_beam_idx].item()
                final_score = final_beam_scores[batch_beam_idx]
                final_tokens = input_ids[batch_beam_idx]
                beam_index = beam_indices[batch_beam_idx] if beam_indices is not None else None
                beam_hyp.add(final_tokens, final_score,
                             beam_indices=beam_index)

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_indices = []
        best_scores = torch.zeros(
            batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                best_index = best_hyp_tuple[2]
                sent_lengths[self.num_beam_hyps_to_keep *
                             i + j] = len(best_hyp)

                # append hyp to lists
                best.append(best_hyp)

                # append indices to list
                best_indices.append(best_index)

                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_lengths_max = sent_lengths.max().item() + 1
        sent_max_len = min(
            sent_lengths_max, max_length) if max_length is not None else sent_lengths_max
        decoded: torch.LongTensor = input_ids.new(
            batch_size * self.num_beam_hyps_to_keep, sent_max_len)

        if len(best_indices) > 0 and best_indices[0] is not None:
            indices: torch.LongTensor = input_ids.new(
                batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        else:
            indices = None

        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        if indices is not None:
            indices.fill_(-1)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, (hypo, best_idx) in enumerate(zip(best, best_indices)):
            decoded[i, : sent_lengths[i]] = hypo

            if indices is not None:
                indices[i, : len(best_idx)] = torch.tensor(best_idx)

            if sent_lengths[i] < sent_max_len:
                decoded[i, sent_lengths[i]] = eos_token_id

        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
                "beam_indices": indices,
            }
        )


class CustomEvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    keys: Optional[List[str]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]


def postprocess_text(predictions, references=None):
    punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-",
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"]

    if predictions:
        predictions = [nltk.word_tokenize(pred.strip())
                       for pred in predictions]
        predictions = [
            [' '.join([w.lower() for w in v if w not in punctuations])] for v in predictions]

    if references:
        references = [[nltk.word_tokenize(ref.strip())
                       for ref in it] for it in references]
        references = [[' '.join([w.lower() for w in v if w not in punctuations])
                       for v in it] for it in references]

    return predictions, references


def save_predictions(output, tokenizer, trainer, split, checkpoint_id=None):
    training_args = trainer.args
    predictions = output.predictions
    keys = output.keys
    metrics = output.metrics

    trainer.log_metrics(split, metrics)
    trainer.save_metrics(split, metrics)

    decoded_predictions = tokenizer.batch_decode(
        predictions, skip_special_tokens=True)
    processed_predictions, _ = postprocess_text(
        predictions=decoded_predictions)

    try:
        ids = [int(key.split('_')[-1]) for key in keys]
    except:
        ids = keys

    coco_format = [{'image_id': idx, "caption": caption[0]}
                   for idx, caption in zip(ids, processed_predictions)]
    filenames_format = {key: caption[0] for key,
                        caption in zip(keys, processed_predictions)}

    checkpoint_id = checkpoint_id or trainer.state.global_step

    with open(Path(training_args.output_dir).joinpath(f'{split}_predictions_coco_format_{checkpoint_id}.json'), 'w') as f:
        json.dump(coco_format, f)
    with open(Path(training_args.output_dir).joinpath(f'{split}_predictions_filenames_{checkpoint_id}.json'), 'w') as f:
        json.dump(filenames_format, f)


@torch.no_grad()
def compute_clip_score(generated_tokens, reference_tokens, images, tokenizer, model, ref=False, w=2.5):
    if reference_tokens[0, 0, 0].item() != tokenizer.bos_token_id:
        reference_tokens = torch.cat([
            torch.full((reference_tokens.shape[0], reference_tokens.shape[1], 1),
                       tokenizer.bos_token_id, device=reference_tokens.device),
            reference_tokens,
        ], dim=-1)

    generated_tokens = copy.deepcopy(generated_tokens)
    if w == 2.5:
        # CLIPModel uses EOS as padding too
        # https://github.com/huggingface/transformers/blob/b71f20a7c9f3716d30f6738501559acf863e2c5c/src/transformers/models/clip/tokenization_clip.py#L309C9-L309C18
        generated_tokens[generated_tokens ==
                         tokenizer.pad_token_id] = tokenizer.eos_token_id
        pad_token_id = tokenizer.eos_token_id
    elif w == 2.0:
        generated_tokens[generated_tokens == tokenizer.pad_token_id] = 0
        pad_token_id = 0

    n_ref_captions = reference_tokens.shape[1]
    n_candidate_captions = len(generated_tokens) // len(images)

    try:
        device = model.device
    except AttributeError:
        device = model.token_embedding.weight.device

    if images.device != device:
        images = images.to(device)
    if generated_tokens.device != device:
        generated_tokens = generated_tokens.to(device)
    if reference_tokens.device != device:
        reference_tokens = reference_tokens.to(device)

    with autocast(dtype=torch.float16):
        image_features = model.get_image_features(images)
        image_features = image_features.repeat_interleave(
            n_candidate_captions, dim=0)
        if generated_tokens.shape[-1] != 77:
            dif_len = 77 - generated_tokens.shape[-1]
            generated_tokens[generated_tokens ==
                             tokenizer.pad_token_id] = pad_token_id
            generated_tokens = torch.cat([
                generated_tokens,
                torch.full((generated_tokens.shape[0], dif_len), pad_token_id,
                           dtype=generated_tokens.dtype, device=generated_tokens.device),
            ], dim=-1)
        cand_features = model.get_text_features(generated_tokens)
        image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)
        cand_features = cand_features / \
            cand_features.norm(dim=-1, keepdim=True)
        clips = w * torch.diagonal(image_features @ cand_features.t(), 0).cpu()

        ref_clips = None
        if ref:
            reference_tokens = copy.deepcopy(reference_tokens)
            if w == 2.5:
                reference_tokens[reference_tokens ==
                                 tokenizer.pad_token_id] = tokenizer.eos_token_id
            elif w == 2.0:
                reference_tokens[reference_tokens ==
                                 tokenizer.pad_token_id] = 0
            reference_tokens = reference_tokens.flatten(0, 1)
            ref_features = model.get_text_features(reference_tokens)
            ref_features = ref_features / \
                ref_features.norm(dim=-1, keepdim=True)

            hidden_size = ref_features.shape[-1]

            ref_features = ref_features.view(-1,
                                             1, n_ref_captions, hidden_size)
            cand_features = cand_features.view(-1,
                                               n_candidate_captions, 1, hidden_size)
            ref_scores = (ref_features * cand_features).sum(-1)

            ref_scores = ref_scores.max(-1)[0]

            ref_scores = torch.clip(ref_scores.view(-1), 0).cpu()
            no_ref_scores = clips
            ref_clips = 2 * no_ref_scores * \
                ref_scores / (no_ref_scores + ref_scores)

    return clips, ref_clips


# TODO
def load_images(keys, preprocess):
    path_ims = Path('/nas/softechict-nas-2/datasets/coco/val2014')
    images = list()
    with autocast(dtype=torch.float16):
        for img_idx in keys:
            image = preprocess(Image.open(path_ims.joinpath(f'{img_idx}.jpg')))
            images.append(image)
    return torch.stack(images)


def load_pacs(pacs_checkpoint: str):
    logger.info("Loading CLIP model for PAC-S...")
    model, clip_preprocess = clip.load("ViT-B/32", device='cuda')
    model.eval()
    checkpoint = torch.load(pacs_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    logger.info("Loaded CLIP model for PAC-S.")

    # workaroud to match huggingface's API for CLIP
    model.get_image_features = model.encode_image
    model.get_text_features = model.encode_text
    return model, clip_preprocess


def load_clips():
    logger.info("Loading CLIP model for PAC-S...")
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32").to(torch.float16)
    preprocess = CLIPFeatureExtractor.from_pretrained(
        "openai/clip-vit-base-patch32")
    model = model.cuda()
    model.eval()
    logger.info("Loaded CLIP model for CLIP-S.")
    return model, preprocess


def clip_score_from_predictions(all_predictions, all_labels, all_keys, tokenizer, pacs_checkpoint, mini_batch=8):
    tot_clips, tot_ref_clips, tot_pacs, tot_ref_pacs = [], [], [], []

    clip_model, clip_preprocess = load_clips()
    w = 2.5
    def preprocess_fn(x): return clip_preprocess(
        x, return_tensors='pt')['pixel_values'][0]
    for i in tqdm(
        range(0, len(all_predictions), mini_batch),
        mininterval=1.0,
        desc="Computing CLIP score..."
    ):
        images = load_images(all_keys[i:i + mini_batch], preprocess_fn)
        predictions = all_predictions[i:i + mini_batch]
        labels = all_labels[i:i + mini_batch]
        clips, ref_clips = compute_clip_score(
            predictions, labels, images, tokenizer, clip_model, ref=True, w=w)
        tot_clips.append(clips)
        tot_ref_clips.append(ref_clips)

    del clip_model
    del clip_preprocess
    clip_model, clip_preprocess = load_pacs(pacs_checkpoint)

    w = 2.0
    def preprocess_fn(x): return clip_preprocess(x)
    for i in tqdm(
        range(0, len(all_predictions), mini_batch),
        mininterval=1.0,
        desc="Computing PAC score..."
    ):
        images = load_images(all_keys[i:i + mini_batch], preprocess_fn)
        predictions = all_predictions[i:i + mini_batch]
        labels = all_labels[i:i + mini_batch]
        decoded_predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True)
        predictions = clip.tokenize(decoded_predictions)
        decoded_labels = tokenizer.batch_decode(
            labels.flatten(0, 1), skip_special_tokens=True)
        try:
            labels = clip.tokenize(decoded_labels, truncate=True).view(
                *labels.shape[:2], -1)
        except:
            continue
        pacs, ref_pacs = compute_clip_score(
            predictions, labels, images, tokenizer, clip_model, ref=True, w=w)
        tot_pacs.append(pacs)
        tot_ref_pacs.append(ref_pacs)

    del clip_model
    del clip_preprocess

    clips = torch.cat(tot_clips).mean().item()
    ref_clips = torch.cat(tot_ref_clips).mean().item()
    pacs = torch.cat(tot_pacs).mean().item()
    ref_pacs = torch.cat(tot_ref_pacs).mean().item()
    return clips, ref_clips, pacs, ref_pacs
