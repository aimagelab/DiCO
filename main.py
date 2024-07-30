from functools import partial
import sys
from pathlib import Path
from types import MethodType
import torch
from transformers import (
    logging,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    AutoModel,
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    CLIPModel,
    set_seed,
    GPT2Config,
    GPT2LMHeadModel,
    BertConfig,
    BertModel,
    CLIPVisionConfig
)
from transformers.trainer_utils import get_last_checkpoint
from models import (
    VisionEncoderEncoderConfig,
    VisionEncoderEncoderModel,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
    CustomCLIPVisionTransformer
)
from trainers.xe_trainer import XETrainer
from trainers.dico_trainer import DiCOTrainer
from utils import compute_metrics, generate_with_backpropagation, save_predictions
import json
import webdataset as wds
from braceexpand import braceexpand
import random
import logging as python_logging

python_logging.basicConfig(level=python_logging.INFO)
logging.set_verbosity_info()
logging.enable_default_handler()
logging.enable_explicit_format()
logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)
is_debug = 'pydevd' in sys.modules

CLIP_BACKBONE = 'openai/clip-vit-large-patch14'


def get_parser():
    parser = HfArgumentParser(Seq2SeqTrainingArguments)

    # model
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_embd', type=int, default=512)
    parser.add_argument('--n_head', type=int, default=8)

    # dico training
    parser.add_argument('--dico', action='store_true')
    parser.add_argument('--dico_beta', type=float, default=0.2)
    parser.add_argument('--dico_tau', type=float, default=300)
    parser.add_argument('--pacs_checkpoint', type=str,
                        default='clip_ViT-B-32.pth')

    # dataset
    parser.add_argument('--train_dataset', type=str,
                        default='coco_training_xe')
    parser.add_argument('--validation_dataset', type=str,
                        default='coco_validation')
    parser.add_argument('--test_dataset', type=str, default=None)

    return parser


def register_models():
    AutoConfig.register('clip_vision_model', CLIPVisionConfig)
    AutoModel.register(CLIPVisionConfig, CustomCLIPVisionTransformer)

    AutoConfig.register('vision-encoder-encoder', VisionEncoderEncoderConfig)
    AutoModel.register(VisionEncoderEncoderConfig, VisionEncoderEncoderModel)

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    CONFIG_MAPPING_NAMES.pop('vision-encoder-decoder')
    AutoConfig.register('vision-encoder-decoder', VisionEncoderDecoderConfig)
    AutoModel.register(VisionEncoderDecoderConfig, VisionEncoderDecoderModel)


def get_vision_encoder():
    clip_model = CLIPModel.from_pretrained(CLIP_BACKBONE)
    clip_model.vision_model.main_input_name = "pixel_values"
    vision_encoder = clip_model.vision_model
    return vision_encoder


def get_model(tokenizer, custom_args, vision_encoder):
    vocab_size = tokenizer.vocab_size + 1
    custom_configuration = GPT2Config()
    custom_configuration.add_cross_attention = True
    custom_configuration.n_layer = custom_args.n_layer or custom_configuration.n_layer
    custom_configuration.n_embd = custom_args.n_embd or custom_configuration.n_embd
    custom_configuration.n_head = custom_args.n_head or custom_configuration.n_head
    custom_configuration.vocab_size = vocab_size
    custom_configuration.bos_token_id = tokenizer.bos_token_id  # 49406
    custom_configuration.eos_token_id = tokenizer.eos_token_id  # 49407
    custom_configuration.pad_token_id = tokenizer.pad_token_id  # 49408
    gpt2_decoder = GPT2LMHeadModel(custom_configuration)

    config_encoder = BertConfig(hidden_size=custom_configuration.n_embd, d_embed=custom_configuration.n_embd,
                                num_hidden_layers=custom_configuration.n_layer, intermediate_size=2048,
                                num_attention_heads=custom_configuration.n_head)
    transformer_encoder = BertModel(config=config_encoder)

    encoder = VisionEncoderEncoderModel(
        vision_encoder=vision_encoder, encoder=transformer_encoder)

    model_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config=encoder.config,
                                                                           decoder_config=gpt2_decoder.config)
    model = VisionEncoderDecoderModel(
        config=model_config, encoder=encoder, decoder=gpt2_decoder)

    model.config.add_cross_attention = True
    model.config.is_encoder_decoder = True
    model.config.decoder_start_token_id = model_config.decoder.decoder_start_token_id if model_config.decoder.decoder_start_token_id is not None else 49406
    model.config.vocab_size = vocab_size
    model.config.bos_token_id = tokenizer.bos_token_id  # 49406
    model.config.eos_token_id = tokenizer.eos_token_id  # 49407
    model.config.pad_token_id = tokenizer.pad_token_id  # 49408
    model.config.hidden_size = model.decoder.config.n_embd  # for DeepSpeed compatibility

    return model


def process_sample(sample, tokenizer, feature_extractor):
    image = feature_extractor(sample['jpg'], return_tensors='pt')

    gts = sample['json'] if 'json' in sample else sample['txt']
    text = {'gts': gts}
    text.update(tokenizer(
        gts, return_tensors='pt', padding='max_length', max_length=77))

    text['__key__'] = sample['__key__']

    return image, text


def create_dataset(dataset_name, map_fn=None, batch_size=8,
                   shuffle=False, repeat=False) -> wds.DataPipeline:
    datasets_file = 'datasets.json'
    with open(datasets_file, 'r') as f:
        available_datasets = json.load(f)
    available_dataset_names = list(available_datasets.keys())
    available_datasets = {name: braceexpand(path) for name, path in available_datasets.items()
                          if name == dataset_name}
    if not available_datasets:
        raise ValueError(
            f'{dataset_name} not in {available_dataset_names}')

    logger.info(f'Loading {dataset_name}...')

    shards = [shard for dataset in list(
        available_datasets.values()) for shard in dataset]

    if shuffle:
        random.shuffle(shards)

    ds = wds.DataPipeline(
        wds.ResampledShards(shards) if repeat else wds.SimpleShardList(shards),
        wds.split_by_worker,
        wds.split_by_node,
        wds.tarfile_to_samples(),
        wds.shuffle(1000) if shuffle else None,
        wds.decode('pil'),
        wds.map(map_fn) if map_fn else None,
        wds.batched(batch_size),
    )

    return ds


def get_datasets(training_args, custom_args, feature_extractor, tokenizer):
    partial_process_sample = partial(
        process_sample,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor
    )

    train_dataset = eval_dataset = test_dataset = None

    if training_args.do_train:
        train_dataset = create_dataset(custom_args.train_dataset, map_fn=partial_process_sample,
                                       batch_size=training_args.per_device_train_batch_size, shuffle=True, repeat=True)

    if training_args.do_eval:
        eval_dataset = create_dataset(custom_args.validation_dataset, map_fn=partial_process_sample,
                                      batch_size=training_args.per_device_eval_batch_size, shuffle=False, repeat=False)

    if training_args.do_predict:
        test_dataset = create_dataset(custom_args.test_dataset, map_fn=partial_process_sample,
                                      batch_size=training_args.per_device_eval_batch_size, shuffle=False, repeat=False)

    return train_dataset, eval_dataset, test_dataset


def get_trainer(custom_args, **kwargs):
    is_inference = kwargs.pop('is_inference', False)
    if custom_args.dico or is_inference:
        trainer_cls = DiCOTrainer
    else:
        trainer_cls = XETrainer

    return trainer_cls(custom_args=custom_args, **kwargs)


def collate_fn(samples, tokenizer):
    out = dict()

    out['pixel_values'] = torch.vstack(
        [sample["pixel_values"] for sample in samples[0]])
    out['main_input_name'] = 'pixel_values'

    labels = torch.nn.utils.rnn.pad_sequence([sample['input_ids'].T for sample in samples[1]],
                                             padding_value=tokenizer.pad_token_id).permute(1, 2, 0)

    # remove BOS (will be added by model)
    out['labels'] = labels[..., 1:]

    out['__key__'] = [sample['__key__'] for sample in samples[1]]

    return out


def main():
    parser = get_parser()

    training_args, custom_args = parser.parse_args_into_dataclasses()

    output_dir = Path(training_args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(training_args.seed)

    register_models()

    feature_extractor = AutoFeatureExtractor.from_pretrained(CLIP_BACKBONE)

    tokenizer = AutoTokenizer.from_pretrained(CLIP_BACKBONE)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    checkpoint = get_last_checkpoint(training_args.output_dir)

    is_inference = training_args.do_predict and not training_args.do_train

    if is_inference or (not checkpoint and training_args.resume_from_checkpoint):
        model = AutoModel.from_pretrained(training_args.resume_from_checkpoint)
    elif checkpoint:
        model = AutoModel.from_pretrained(checkpoint)
    else:
        vision_encoder = get_vision_encoder()
        model = get_model(tokenizer, custom_args, vision_encoder)

    dtype = torch.float16 if training_args.fp16 else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dtype=dtype, device=device)

    train_dataset, eval_dataset, test_dataset = get_datasets(
        training_args,
        custom_args,
        feature_extractor,
        tokenizer
    )

    # will be set by deepspeed
    optimizer, scheduler = None, None

    if custom_args.dico:
        model.generate = MethodType(generate_with_backpropagation, model)
        model.generate_with_backpropagation = MethodType(
            generate_with_backpropagation, model)

    partial_collate_fn = partial(collate_fn, tokenizer=tokenizer)
    partial_compute_metrics = partial(compute_metrics, tokenizer=tokenizer)

    trainer = get_trainer(
        custom_args=custom_args,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial_collate_fn,
        compute_metrics=partial_compute_metrics,
        optimizers=(optimizer, scheduler),
        tokenizer=tokenizer,
        is_inference=is_inference,
    )

    if training_args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=checkpoint, ignore_keys_for_eval=['input_ids'])
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        trainer.save_model()

    if training_args.do_predict:
        prediction_output = trainer.predict(
            test_dataset=test_dataset, num_beams=training_args.generation_num_beams)

        checkpoint_id = training_args.resume_from_checkpoint.split(
            '/checkpoint-')[-1].replace('/', '')
        save_predictions(prediction_output, tokenizer, trainer,
                         split='test', checkpoint_id=checkpoint_id)

    logger.info('END')


if __name__ == "__main__":
    main()
