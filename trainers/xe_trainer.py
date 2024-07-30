import copy
import math
import sys
import time
from argparse import Namespace
from typing import Optional, List, Callable, Dict, Union, Any, Tuple
import numpy as np
import torch
import torch.distributed as dist
import webdataset as wds
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, get_scheduler
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.trainer_pt_utils import find_batch_size, IterableDatasetShard, nested_numpify, nested_concat
from transformers.trainer_utils import has_length, EvalPrediction, denumpify_detensorize, PredictionOutput, \
    speed_metrics
from transformers.utils import logging
from evaluation.cider.cider_huggingface import Cider
import utils
import os


is_debug = 'pydevd' in sys.modules
logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)


class XETrainer(Seq2SeqTrainer):
    def __init__(self, test_dataset: Optional[Dataset] = None, cider: Optional[Cider] = None, custom_args: Optional[Namespace] = None, **kwargs):
        self.test_dataset = test_dataset
        self.cider = cider
        self.custom_args = custom_args

        super().__init__(**kwargs)


    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        train_dataloader = wds.WebLoader(
            train_dataset,
            batch_size=None,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        return train_dataloader


    def get_eval_dataloader(self, eval_dataset: Optional[wds.DataPipeline] = None) -> wds.WebLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataloader = wds.WebLoader(
            eval_dataset,
            batch_size=None,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )

        return eval_dataloader


    def get_test_dataloader(self, test_dataset: Optional[wds.DataPipeline] = None) -> wds.WebLoader:
        if test_dataset is None and self.test_dataset is None:
            raise ValueError("Trainer: testuation requires an test_dataset.")
        test_dataset = test_dataset if test_dataset is not None else self.test_dataset

        test_dataloader = wds.WebLoader(
            test_dataset,
            batch_size=None,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )

        return test_dataloader


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get(
                "global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        gen_kwargs.update({'__key__': inputs.get('__key__')})
        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(
                        outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(
                        outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(
                    labels, gen_kwargs["max_length"])
        else:
            labels = None

        return loss, generated_tokens, labels

    def evaluation_loop(
        self,
        dataloader: wds.WebLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> utils.CustomEvalLoopOutput:
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        # eval_dataset = getattr(dataloader, "dataset", None)
        eval_dataset = getattr(dataloader, "dataset", None)

        # if is_torch_tpu_available():
        #     dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        keys_host = list()
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_keys = list()
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        pad_token_id = self.model.config.pad_token_id
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            inputs['output_dir'] = self.args.output_dir
            if is_debug and step > 5:
                break
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            keys = copy.copy(inputs['__key__'])
            labels = copy.copy(inputs['labels'])

            # during evaluation compute loss only wrt the first reference caption
            inputs['labels'] = inputs['labels'][:, 0, :]

            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].type(
                    model.dtype)
            if 'vision_encoder_outputs' in inputs:
                inputs['vision_encoder_outputs'] = inputs['vision_encoder_outputs'].type(
                    model.dtype)
            if inputs.get('main_input_name'):
                main_input_name = inputs.get('main_input_name')
                model.main_input_name = main_input_name
                if hasattr(model, 'encoder'):
                    model.encoder.main_input_name = main_input_name
            loss, logits, _ = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = inputs["input_ids"] if args.include_inputs_for_metrics else None
            labels = labels.to(model.device)
            # pad anc concat methods work only on the 1st dimension, workaround: double swapaxes(1, -1)
            labels = labels.swapaxes(1, -1)
            labels = labels.contiguous()

            # Update containers on host
            if keys is not None:
                gathered_keys = [None for _ in range(utils.get_world_size())]
                if dist.is_initialized():
                    dist.all_gather_object(gathered_keys, keys)
                else:
                    gathered_keys = keys
                keys_host.extend(kk for k in gathered_keys for kk in k)
                # keys_host.extend(gathered_keys)
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat(
                    (losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(
                    labels, pad_index=pad_token_id)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels,
                                                                               padding_index=pad_token_id)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(
                    preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate(
                        (all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(
                        all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(
                            all_labels, labels, padding_index=pad_token_id)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

            if (step % self.args.logging_steps == 0 or is_debug) and utils.is_main_process():
                world_size_observed_num_examples = observed_num_examples * utils.get_world_size()
                logger.info(f'{step=}, {world_size_observed_num_examples=}')

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if keys_host:
            all_keys = keys_host
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate(
                (all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(
                all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(
                    all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(
                all_labels, labels, padding_index=pad_token_id)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.

        # original code, doesn't work in distributed setting

        # if all_keys is not None:
        #     all_keys = all_keys[::get_world_size()]
        # if all_losses is not None:
        #     all_losses = all_losses[:num_samples]
        # if all_preds is not None:
        #     all_preds = nested_truncate(all_preds, num_samples)
        # if all_labels is not None:
        #     all_labels = nested_truncate(all_labels, num_samples)
        # if all_inputs is not None:
        #     all_inputs = nested_truncate(all_inputs, num_samples)

        # Metrics!
        if self.compute_metrics is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds,
                                   label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(
                    predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        output =  utils.CustomEvalLoopOutput(predictions=all_preds, keys=all_keys, label_ids=all_labels, metrics=metrics,
                                    num_samples=num_samples)
        utils.save_predictions(output, self.tokenizer, trainer=self, split=metric_key_prefix)
        
        if os.path.exists(os.path.join(self.args.output_dir, "eval_results.json")):
            os.remove(os.path.join(self.args.output_dir, "eval_results.json"))
        if os.path.exists(os.path.join(self.args.output_dir, "all_results.json")):
            os.rename(os.path.join(self.args.output_dir, "all_results.json"), os.path.join(self.args.output_dir, "last_checkpoint_results.json"))
        
        return output
