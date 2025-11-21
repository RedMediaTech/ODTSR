"""
copy from https://github.com/csslc/PiSA-SR/blob/main/src/datasets/dataset.py
change record: https://www.diffchecker.com/wEIJtoyR/
"""
import os
import random
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from pathlib import Path

import numpy as np
from .realesrgan import RealESRGAN_degradation

import pillow_heif
pillow_heif.options.DISABLE_SECURITY_LIMITS = True

def get_prompt_given_img_path(img_path):
    txt_path = img_path.replace("/img/", "/prompt/").replace(".png", ".txt")
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()  # 读取内容并去除首尾空白字符
    except FileNotFoundError:
        print(f"Warning: Prompt file not found for {img_path}")
        return ""
    except Exception as e:
        print(f"Error reading prompt file {txt_path}: {str(e)}")
        return ""

def tensor_01_to_pil(tensor):
    tensor = tensor.detach().cpu()
    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
    tensor = tensor.permute(1, 2, 0)
    image = Image.fromarray(tensor.numpy())
    return image

class PairedSROnlineTxtDataset(torch.utils.data.Dataset):
    """
        args should have:
        * deg_file_path
        * dataset_txt_paths
        * use_qwen
        * highquality_dataset_txt_paths
        * null_text_ratio
    """
    def __init__(self, split=None, args=None):
        super().__init__()

        self.args = args
        self.split = split
        if split == 'train':
            self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
            self.crop_preproc = transforms.Compose([
                # transforms.Resize(1440),
                transforms.RandomCrop((512, 512)),
                # transforms.RandomHorizontalFlip(),
            ])
            
            self.gt_list = []
            for txt in args.dataset_txt_paths.split():   # 按空格切分成多个文件路径
                with open(txt, 'r') as f:
                    self.gt_list.extend([line.strip() for line in f])
            if args.highquality_dataset_txt_paths is not None:
                with open(args.highquality_dataset_txt_paths, 'r') as f:
                    self.hq_gt_list = [line.strip() for line in f.readlines()]

        elif split == 'test':
            self.input_folder = os.path.join(args.dataset_test_folder, "test_SR_bicubic")
            self.output_folder = os.path.join(args.dataset_test_folder, "test_HR")
            self.lr_list = []
            self.gt_list = []
            lr_names = os.listdir(os.path.join(self.input_folder))
            gt_names = os.listdir(os.path.join(self.output_folder))
            assert len(lr_names) == len(gt_names)
            for i in range(len(lr_names)):
                self.lr_list.append(os.path.join(self.input_folder, lr_names[i]))
                self.gt_list.append(os.path.join(self.output_folder,gt_names[i]))
            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop((args.resolution_ori, args.resolution_ori)),
                transforms.Resize((args.resolution_tgt, args.resolution_tgt)),
            ])
            assert len(self.lr_list) == len(self.gt_list)

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        def safe_open(idx, paths):
            # 尝试使用指定索引
            if 0 <= idx < len(paths) and os.path.exists(paths[idx]):
                try:
                    img = Image.open(paths[idx]).convert('RGB')
                    return img, paths[idx]
                except:
                    pass

            # 随机选择可用路径
            while True:
                p = random.choice(paths)
                if os.path.exists(p):
                    try:
                        img = Image.open(p).convert('RGB')
                        return img, p
                    except:
                        continue

        if self.split == 'train':
            if self.args.highquality_dataset_txt_paths is not None:
                if np.random.uniform() < self.args.prob:
                    gt_img = Image.open(self.gt_list[idx]).convert('RGB')
                else:
                    idx = random.sample(range(0, len(self.hq_gt_list)), 1)
                    gt_img = Image.open(self.hq_gt_list[idx[0]]).convert('RGB')
            else:
                gt_img, gt_img_path = safe_open(idx, self.gt_list)
            gt_img = self.crop_preproc(gt_img)

            output_t, img_t = self.degradation.degrade_process(np.asarray(gt_img)/255., resize_bak=True)
            output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)

            # input images scaled to -1,1
            # img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            # output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            # [0,1] c,h,w  to uint8 pil IMage 


            example = {}
            # example["prompt"] = caption
            # example["neg_prompt"] = self.args.neg_prompt_csd
            # example["null_prompt"] = ""
            example["gt"] = tensor_01_to_pil(output_t)
            example["lq"] = tensor_01_to_pil(img_t)
            
            if not self.args.use_qwen:
                random_value = random.random()
                if random_value < self.args.null_text_ratio:
                    text = ""
                else:
                    text = get_prompt_given_img_path(gt_img_path)
                    
                example['text'] = text
            
            return example
            
        elif self.split == 'test':
            input_img = Image.open(self.lr_list[idx]).convert('RGB')
            output_img = Image.open(self.gt_list[idx]).convert('RGB')
            img_t = self.crop_preproc(input_img)
            output_t = self.crop_preproc(output_img)
            # input images scaled to -1, 1
            img_t = F.to_tensor(img_t)
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = F.to_tensor(output_t)
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_csd
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t
            example["base_name"] = os.path.basename(self.lr_list[idx])

            return example