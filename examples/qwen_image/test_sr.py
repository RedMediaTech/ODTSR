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
def test(args):
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

    model = Generator(
        torch_dtype = torch.bfloat16,
        pretrained_weights=sd_safe_tensor_path_json_format,
        tokenizer_path = f"{pretrained_qwen_path}/tokenizer",
        learning_rate=0,
        use_gradient_checkpointing=False,
        pretrained_ckpt_path_gen = args.trained_ckpt
    )
    
    model.pipe.requires_grad_(False)
    model = model.to(device='cuda')
    model.device = next(model.parameters()).device
    model.pipe.device = model.device
    
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.input_path, '**', ext), recursive=True))
    
    image_paths = sorted(image_paths)
    print("total len:", len(image_paths))
    
    range_parts = args.start_end.split(',')
    start_idx = int(range_parts[0]) if range_parts[0] else 0
    end_idx = int(range_parts[1]) if len(range_parts) > 1 and range_parts[1] else None

    if end_idx is not None:
        image_paths = image_paths[start_idx:end_idx]
    else:
        image_paths = image_paths[start_idx:]

    os.makedirs(args.output_path, exist_ok=True)
    
    image_paths.insert(0, image_paths[0])

    for idx,img_path in enumerate(image_paths):
        if idx == 1:
            start = time.perf_counter()

        # 读取并转换图像
        print(f'deal {img_path}')
        filename_without_extension = os.path.splitext(os.path.basename(img_path))[0]
        res_path = os.path.join(args.output_path, f"{filename_without_extension}.png")
        print(f"will save to {res_path} \n")

        img = Image.open(img_path).convert('RGB')
        w,h = img.size
        w_desti = round(w * args.scale)
        h_desti = round(h * args.scale)

        upsampled_img = img.resize((w_desti, h_desti), Image.BICUBIC)
        img = adaptive_pad(upsampled_img, tilesize=tilesize * 8, stride=tile_stride * 8)
        
        # get caption
        prompt = """High Contrast, hyper detailed photo, 2k UHD"""
        
        # def get_prompt_given_name(imgpath):
        #     name = os.path.splitext(os.path.basename(imgpath))[0]
        #     txtpath = os.path.join('xxxxxxx/datasets/realsr/1/prompt', f"{name}.txt")
        #     # 获得txt内容
        #     with open(txtpath, 'r') as f:
        #         prompt = f.read()
        #     return prompt

        # prompt = get_prompt_given_name(img_path)
        print("prompt:", prompt)

        res_img = model.infer(
            prompt = prompt,
            negative_prompt = "",
            condition_image = img,
            cfg_scale = args.cfg,
            fidelity = 1.0,
            tiled = True,
            tile_size = tilesize,
            tile_stride = tile_stride
        )

        cropped_image = res_img.crop((0, 0, w_desti, h_desti))

        if args.align_method == 'adain':
            output_pil = adain_color_fix(target=cropped_image, source=upsampled_img)
            print("use adain color fix")
        elif args.align_method == 'wavelet':
            output_pil = wavelet_color_fix(target=cropped_image, source=upsampled_img)
            print("use wavelet color fix")
        else:
            output_pil = cropped_image
            print("no color fix")
        output_pil.save(res_path)
        
    end = time.perf_counter()
    print(f"总耗时: {end - start:.6f} 秒")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a test script.")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input images.",
    )
    parser.add_argument(
        "--trained_ckpt",
        type=str,
        required=True,
        help="Path to trained_ckpt.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./test_outputs",
        help="Path to save the results.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="sr scale",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=1.0,
        help="cfg",
    )
    parser.add_argument(
        "--start_end",
        type=str,
        default="0,",
        help="index range",
    )
    parser.add_argument(
        "--align_method",
        type=str,
        default="adain",
        help="color align method, adain or wavelet",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    test(args)