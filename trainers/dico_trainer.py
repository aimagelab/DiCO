import copy
import math
import sys
import time
from argparse import Namespace
from typing import Optional, List, Callable, Union, Tuple
import copy
import numpy as np
import torch
import torch.distributed as dist
import webdataset as wds
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.trainer_pt_utils import find_batch_size, IterableDatasetShard, nested_numpify, nested_concat
from transformers.trainer_utils import has_length, EvalPrediction, denumpify_detensorize, PredictionOutput, \
    speed_metrics
from transformers.utils import logging
from torch.nn import functional as F
from torch.utils.data import Dataset
from models.vision_encoder_decoder import VisionEncoderDecoderModel
from .trainer_utils import TrainerRewardModel
import utils
import os

is_debug = 'pydevd' in sys.modules
logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

TRAINER_STATE_NAME = "trainer_state.json"


class DiCOTrainer(Seq2SeqTrainer):

    def __init__(self, custom_args: Optional[Namespace] = None, **kwargs):
        self.ref_model = None
        
        self.custom_args = custom_args

        super().__init__(**kwargs)
        self.decoder_pad_token = self.model.decoder.config.pad_token_id
        
        ref_model = VisionEncoderDecoderModel.from_pretrained(self.args.resume_from_checkpoint)
        ref_model = self._wrap_model(ref_model, training=False)
        ref_model.eval()
        self.ref_model = ref_model
        
        self.reward_model = TrainerRewardModel(self.args, custom_args, self.tokenizer)


    def _wrap_model(self, model, training=True, dataloader=None):
        # workaround for distributed data parallel
        generate = getattr(model, 'generate', None)
        generate_wb = getattr(model, 'generate_with_backpropagation', None)
        compute_transition_beam_scores = getattr(model, 'compute_transition_beam_scores', None)
        vision_encoder_forward = getattr(model, 'vision_encoder_forward', None)
        model = super()._wrap_model(model, training=training, dataloader=dataloader)
        
        try:
            inner_model = model.module
        except AttributeError:
            inner_model = model

        if self.ref_model is not None:
            self.ref_model = self.ref_model.to(inner_model.dtype).to(inner_model.device)

        if generate is not None:
            model.generate = generate
        if generate_wb is not None:
            model.generate_with_backpropagation = generate_wb
        if compute_transition_beam_scores is not None:
            model.compute_transition_beam_scores = compute_transition_beam_scores
        if vision_encoder_forward is not None:
            model.vision_encoder_forward = vision_encoder_forward   
        return model


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
            num_workers=0,  # awful workaround!
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
            )

        return test_dataloader
    
    
    def compute_beam_search(self, model, inputs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        max_length = self.args.generation_max_length
        num_beams = self.args.generation_num_beams
        if "labels" in inputs:
            labels = inputs.get("labels")
        else:
            labels = None
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": max_length if max_length is not None else model.config.max_length,
            "num_beams": num_beams if num_beams is not None else model.config.num_beams,
            'num_return_sequences': num_beams if num_beams is not None else model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
            "output_scores": True,
            "return_dict_in_generate": True
        }

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if inputs.get('main_input_name'):
            main_input_name = inputs.get('main_input_name')
            model.main_input_name = main_input_name
            if hasattr(model, 'encoder'):
                model.encoder.main_input_name = main_input_name
        elif hasattr(model, "encoder") and model.encoder.main_input_name != model.main_input_name:
            main_input_name = model.encoder.main_input_name
        else:
            main_input_name = model.main_input_name

        try:
            generation_inputs = inputs[main_input_name]
            generation_inputs.requires_grad = True
        except KeyError:
            generation_inputs = inputs['vision_encoder_outputs']
            gen_kwargs['use_vision_encoder_outputs'] = True
            generation_inputs.requires_grad = False

        output = model.generate_with_backpropagation(
            generation_inputs,
            **gen_kwargs,
        )
        generated_tokens = output['sequences']
        #log_probs = output['sequences_scores']
        log_probs = model.compute_transition_beam_scores(output.sequences, output.scores, output.beam_indices, self.tokenizer.eos_token_id)
        log_probs = log_probs.view(labels.shape[0], num_beams, -1)

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        
        output['generated_tokens'] = generated_tokens
        output['log_probs'] = log_probs
        return output


    def predict(
        self,
        test_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> PredictionOutput:
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        if hasattr(self.state, 'best_model_checkpoint') and self.state.best_model_checkpoint is not None and self.args.load_best_model_at_end:
            output.metrics['best_model_checkpoint'] = self.state.best_model_checkpoint

        utils.save_predictions(output, self.tokenizer, trainer=self, split=metric_key_prefix)

        return output


    def prepare_dico_inputs(self, model, inputs):  
        inputs['output_dir'] = self.args.output_dir
        n_candidates_per_sample = self.args.generation_num_beams
        with torch.no_grad():
            vision_encoder_outputs = model.vision_encoder_forward(inputs['pixel_values']).last_hidden_state
        inputs['vision_encoder_outputs'] = vision_encoder_outputs
        inputs['dico_repeat_interleave_encoder_hidden_states'] = n_candidates_per_sample
        pixel_values = inputs.pop('pixel_values')
        optim_outputs = self.compute_beam_search(model, inputs)
        generated_tokens = optim_outputs.generated_tokens
        candidate_policy = optim_outputs.log_probs.sum(-1)
        return inputs, pixel_values, generated_tokens, n_candidates_per_sample, candidate_policy
    

    def compute_dico_logps(self, output_logits, labels, loss_mask):
        per_token_logps = torch.gather(output_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        return (per_token_logps * loss_mask).sum(-1)


    def weight_quality_distances(self, rewards, policy_rejected_logps, reference_rejected_logps, chosen_mask, rejected_mask, tau):
        max_rewards = rewards.view_as(chosen_mask)[chosen_mask].unsqueeze(1)
        num_negatives = chosen_mask.shape[1]
        diff_measure = (max_rewards - rewards.reshape(-1, num_negatives))[rejected_mask] * tau
        gamma = torch.nn.functional.softmax(diff_measure.reshape(-1, num_negatives-1).cuda(), dim=-1).reshape(-1)
        policy_rejected_logps *= gamma
        reference_rejected_logps *= gamma

        return policy_rejected_logps, reference_rejected_logps


    def dico_loss(self, candidate_policy, ref_policy, chosen_mask, rejected_mask, rewards):
        bsz = chosen_mask.shape[0]

        reference_chosen_logps = ref_policy[chosen_mask]
        reference_rejected_logps = ref_policy[rejected_mask]

        policy_chosen_logps = candidate_policy[chosen_mask]
        policy_rejected_logps = candidate_policy[rejected_mask]

        policy_rejected_logps, reference_rejected_logps = self.weight_quality_distances(rewards, policy_rejected_logps, reference_rejected_logps, chosen_mask, rejected_mask, self.custom_args.dico_tau)

        policy_rejected_logps = policy_rejected_logps.view(bsz, -1).sum(-1)
        reference_rejected_logps = reference_rejected_logps.view(bsz, -1).sum(-1)
            
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        loss = -F.logsigmoid(self.custom_args.dico_beta * logits)
        chosen_rewards = self.custom_args.dico_beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.custom_args.dico_beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss.mean(), chosen_rewards.mean().item(), rejected_rewards.mean().item()


    def compute_rank(self, rewards, num_beams):
        rewards_batch = rewards.reshape(-1, num_beams)
        positive_indexes = torch.argmax(rewards_batch, dim=-1, keepdim=True)
        positive_mask = torch.zeros_like(rewards_batch, dtype=torch.bool)

        return torch.scatter(
            positive_mask,
            dim=1,
            index=positive_indexes,
            src=torch.ones_like(positive_indexes, dtype=torch.bool)
        )


    def compute_dico_loss(self, model, inputs, return_outputs=False):
        inputs, pixel_values, generated_tokens, n_candidates_per_sample, candidate_policy = self.prepare_dico_inputs(model, inputs)
        labels = inputs.pop('labels')
        
        with torch.no_grad():
            ref_inputs = {**inputs}
            ref_decoder_input_ids = generated_tokens.clone()

            # drop eos
            ref_decoder_input_ids[ref_decoder_input_ids == self.tokenizer.eos_token_id] = self.decoder_pad_token

            ref_inputs['decoder_input_ids'] = ref_decoder_input_ids
            ref_inputs['dico'] = True
            ref_inputs['shift_token_right'] = False

            ref_outputs = self.ref_model(**ref_inputs)

        rewards = self.reward_model(generated_tokens, labels, pixel_values)
        chosen_mask = self.compute_rank(rewards, n_candidates_per_sample)
        rejected_mask = ~chosen_mask

        # drop bos
        ref_labels = generated_tokens[:, 1:]

        loss_mask = (ref_labels != self.decoder_pad_token)

        # loss of ref model is computed using the token ids generated by the optim model
        ref_policy = self.compute_dico_logps(ref_outputs.logits, ref_labels, loss_mask).view(chosen_mask.shape)
        model_outputs = model(**ref_inputs)
        candidate_policy = self.compute_dico_logps(model_outputs.logits, ref_labels, loss_mask).view(chosen_mask.shape)
        
        loss, chosen_rewards, rejected_rewards = self.dico_loss(candidate_policy, ref_policy, chosen_mask, rejected_mask, rewards)

        reward_dict = dict()
        if chosen_rewards is not None:
            reward_dict['chosen_rewards'] = chosen_rewards
            reward_dict['rejected_rewards'] = rejected_rewards

        return (loss, reward_dict) if return_outputs else loss


    def compute_loss(self, model, inputs, return_outputs=False):
        if model.training:
            loss_and_outputs = self.compute_dico_loss(model, inputs, return_outputs=True)
            if (self.state.global_step * self.args.gradient_accumulation_steps) % self.args.logging_steps == 0:
                self.callback_handler.on_log(self.args, self.state, self.control, loss_and_outputs[1])
            return loss_and_outputs if return_outputs else loss_and_outputs[0]
        
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        inputs['output_dir'] = self.args.output_dir
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    

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
            inputs['labels'] = inputs['labels'][:, 0, :]
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].type(model.dtype)
            if 'vision_encoder_outputs' in inputs:
                inputs['vision_encoder_outputs'] = inputs['vision_encoder_outputs'].type(model.dtype)
            if inputs.get('main_input_name'):
                main_input_name = inputs.get('main_input_name')
                model.main_input_name = main_input_name
                if hasattr(model, 'encoder'):
                    model.encoder.main_input_name = main_input_name
            loss, logits, _ = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
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
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels, pad_index=pad_token_id)
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
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
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
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=pad_token_id)
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
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=pad_token_id)

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

        # Metrics!
        if self.compute_metrics is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
                
            all_preds_pt, all_labels_pt = torch.from_numpy(all_preds), torch.from_numpy(all_labels).swapaxes(-1, -2)
            del self.reward_model
            clips, ref_clips, pacs, ref_pacs = utils.clip_score_from_predictions(all_preds_pt, all_labels_pt, all_keys, self.tokenizer, self.custom_args.pacs_checkpoint)
            metrics['CLIPScore'] = clips
            metrics['ref-CLIPScore'] = ref_clips
            metrics['PACScore'] = pacs
            metrics['ref-PACScore'] = ref_pacs
            self.reward_model = TrainerRewardModel(self.args, self.custom_args, self.tokenizer)
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
    