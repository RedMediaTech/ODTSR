"""
    单步模型 任意分辨率超分 推理代码
    注意分块大小要和训练时一致
"""
import os
import argparse
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
import glob
from PIL import Image
from generator import Generator
from wavelet_color_fix import adain_color_fix, wavelet_color_fix
import gradio as gr

def adaptive_pad(img, tilesize, stride):
    w, h = img.size
    pad_h = (tilesize - h) if h <= tilesize else ((h - tilesize + stride - 1) // stride) * stride + tilesize - h
    pad_w = (tilesize - w) if w <= tilesize else ((w - tilesize + stride - 1) // stride) * stride + tilesize - w

    new_w = w + pad_w
    new_h = h + pad_h

    new_img = Image.new(img.mode, (new_w, new_h), color=0)
    new_img.paste(img, (0, 0))
    return new_img

@torch.no_grad()
def build_inference_fn():
    tilesize = 64
    tile_stride = tilesize - tilesize // 4
    
    weight_dtype = torch.bfloat16

    pretrained_qwen_path = os.environ["qwen_path"]

    sd_safe_tensor_path_json_format = f'''[
        [
            "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
            "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
            "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
            "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
            "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
            "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
            "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
            "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
            "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00009-of-00009.safetensors"
        ],
        [
            "{pretrained_qwen_path}/text_encoder/model-00001-of-00004.safetensors",
            "{pretrained_qwen_path}/text_encoder/model-00002-of-00004.safetensors",
            "{pretrained_qwen_path}/text_encoder/model-00003-of-00004.safetensors",
            "{pretrained_qwen_path}/text_encoder/model-00004-of-00004.safetensors"
        ],
        "{pretrained_qwen_path}/vae/diffusion_pytorch_model.safetensors"
    ]'''

    pretrained_ckpt_path_gen = os.environ["lora_pth_path"]

    model = Generator(
        torch_dtype = torch.bfloat16,
        pretrained_weights=sd_safe_tensor_path_json_format,
        tokenizer_path = f"{pretrained_qwen_path}/tokenizer",
        learning_rate=0,
        use_gradient_checkpointing=False,
        pretrained_ckpt_path_gen = pretrained_ckpt_path_gen
    )
    
    model.pipe.requires_grad_(False)
    model = model.to(device='cuda')
    model.device = next(model.parameters()).device
    model.pipe.device = model.device
    
    def inference(
        img: Image.Image,
        scale: float,
        prompt: str,
        neg_prompt: str,
        fidelity: float,
    ):
        img = img.convert('RGB')
        w,h = img.size
        w_desti = round(w * scale)
        h_desti = round(h * scale)

        upsampled_img = img.resize((w_desti, h_desti), Image.BICUBIC)
        img = adaptive_pad(upsampled_img, tilesize=tilesize * 8, stride=tile_stride * 8)
        
        res_img = model.infer(
            prompt = prompt,
            negative_prompt = neg_prompt,
            condition_image = img,
            cfg_scale = 1.0,
            fidelity = fidelity,
            tiled = True,
            tile_size = tilesize,
            tile_stride = tile_stride
        )
        cropped_image = res_img.crop((0, 0, w_desti, h_desti))
        output_pil = wavelet_color_fix(target=cropped_image, source=upsampled_img)
        return output_pil
    
    return inference

        
if __name__ == '__main__':
    real_fn = build_inference_fn()
    inputs = [
        gr.Image(label="Input image", type="pil"),      # 直接拿到 PIL.Image
        gr.Slider(label="sr scale", minimum=0.5, maximum=4.0, step=0.5, value=2.0),
        gr.Textbox(label="Prompt", placeholder="selectable, default is empty", value=""),
        gr.Textbox(label="Negative prompt", placeholder="selectable, default is empty", value=""),
        gr.Slider(label="fidelity", minimum=0.0, maximum=1.0, step=0.05, value=1.0),
    ]
    outputs = gr.Image(label="SR result", type="pil", format='png') 
    demo = gr.Interface(
        fn=real_fn,
        inputs=inputs,
        outputs=outputs
    )
    demo.launch(server_name="0.0.0.0", server_port=7861)