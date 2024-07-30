import gradio as gr
import torch
from main import register_models, CLIP_BACKBONE
import argparse
from transformers import AutoModel, AutoFeatureExtractor, AutoTokenizer
from functools import partial


def generate_caption(image, model, feature_extractor, tokenizer, num_beams=1):
    inputs = feature_extractor(image, return_tensors='pt')
    inputs['pixel_values'] = inputs['pixel_values'].to(dtype=model.dtype, device=device)
    inputs['num_beams'] = num_beams
    
    with torch.no_grad():
        output = model.generate(**inputs)
    
    caption = tokenizer.decode(output[0], skip_special_tokens=True)
    return caption


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='dico-ViTL14')
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--num_beams', type=int, default=5)
    args = parser.parse_args()

    register_models()

    model = AutoModel.from_pretrained(args.checkpoint)
    feature_extractor = AutoFeatureExtractor.from_pretrained(CLIP_BACKBONE)
    tokenizer = AutoTokenizer.from_pretrained(CLIP_BACKBONE)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32 if args.fp32 else torch.float16

    model.to(dtype=dtype, device=device)
    model.eval()

    iface = gr.Interface(
        fn=partial(
            generate_caption,
            model=model,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            num_beams=args.num_beams
        ),
        inputs=[gr.Image(type='pil')],
        outputs=['text'],
        live=True,
        title='DiCO - Revisiting Image Captioning Training Paradigm via Direct CLIP-based Optimization',
        description='Upload an image, and the model will generate a caption.',
        submit_btn=gr.Button('Generate caption', interactive=True),
        allow_flagging='never'
    )

    iface.launch()
