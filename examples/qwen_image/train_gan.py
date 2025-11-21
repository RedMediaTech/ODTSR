"""
生成器概率使用prompt
判别器使用prompt
prompt来自qwen

单步 gan训练
"""
import os
import gc
import random
import shutil
import torch
from datetime import datetime
from contextlib import redirect_stdout, nullcontext
from utils import yaml_load, parse_args
from PIL import Image, ImageDraw, ImageFont

from ganloss import GANLoss
from generator import Generator
from discriminator import Discriminator
from dinov3loss import DINOv3PerceptualLoss

from transformers import AutoModel
from diffsynth.extensions.realesrgan.dataset import PairedSROnlineTxtDataset
from copy import deepcopy
from accelerate import Accelerator
from accelerate.utils import set_seed
from lightning.fabric.loggers import TensorBoardLogger

import lpips
import matplotlib.pyplot as plt
import numpy as np

def get_gan_weight(now_sigma, start_sigma, min_weight = 1.0, max_weight = 5.0):
    ratio = now_sigma / start_sigma
    weight = min_weight + (max_weight - min_weight) * ratio
    return weight

def deal_discriminator_condition(input1, input2, use_dual_condition_flag):
    if use_dual_condition_flag:
        return torch.stack([input1, input2], dim=2)
    else:
        return input1

def adaptive_relaxed_mean(tensor, output_size=10):
    if output_size >= 100:
        return tensor
    else:
        # 使用自适应平均池化然后上采样
        orig_size = tensor.shape[2:]
        pooled = torch.nn.functional.adaptive_avg_pool2d(tensor, (output_size, output_size))
        return torch.nn.functional.interpolate(pooled, size=orig_size, mode='bilinear')

def add_gaussian_noise(input_tensor, variance=0.01):
    # 计算标准差
    std = variance ** 0.5  # sqrt(0.01) = 0.1
    
    # 生成相同形状的噪声
    noise = torch.randn_like(input_tensor) * std
    
    # 添加到原始输入
    return input_tensor + noise


def random_crop_pair(data_dict, crop_h, crop_w):
    """
    对包含 'gt' 和 'lq' 两个 PIL.Image 的字典进行相同的随机裁剪
    :param data_dict: {'gt': PIL.Image, 'lq': PIL.Image}
    :param crop_h: 裁剪高度
    :param crop_w: 裁剪宽度
    :return: {'gt': PIL.Image, 'lq': PIL.Image}
    """
    gt_img = data_dict['gt']
    lq_img = data_dict['lq']

    assert gt_img.size == lq_img.size, "GT 和 LQ 的尺寸不一致"
    w, h = gt_img.size
    assert crop_w <= w and crop_h <= h, "裁剪尺寸不能大于原图尺寸"

    # 随机生成裁剪起始点
    left = random.randint(0, w - crop_w)
    top = random.randint(0, h - crop_h)

    # 裁剪区域 (left, top, right, bottom)
    box = (left, top, left + crop_w, top + crop_h)

    return gt_img.crop(box), lq_img.crop(box)
    


def crop_for_bchw(model_pred, cropsize, gt_rgb):
    batch_size, channels, height, width = model_pred.shape
    assert cropsize < height and cropsize < width
    
    start_h = np.random.randint(0, height - cropsize + 1)
    start_w = np.random.randint(0, width - cropsize + 1)
    
    cropped_latent = model_pred[:, :, start_h:start_h + cropsize, start_w:start_w + cropsize]
    
    gt_start_h = start_h * 8
    gt_start_w = start_w * 8
    gt_cropsize = cropsize * 8
    
    cropped_gt = gt_rgb[:, :, gt_start_h:gt_start_h + gt_cropsize, gt_start_w:gt_start_w + gt_cropsize]
    
    return cropped_latent, cropped_gt


def train(args):
    dataset_yaml = yaml_load(args.mmaigc_dataset_yml)
    gradient_accumulation_steps = dataset_yaml['accumulate_grad_batches']
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='bf16',
        log_with='tensorboard',
        project_dir = os.path.join("./experiments", dataset_yaml['exp_tag'])
    )
    
    set_seed(42)

    dataset = PairedSROnlineTxtDataset(
        split="train",
        args=args
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=2,
        collate_fn=lambda x: x[0]
    )
    
    pretrained_qwen_path = os.environ["qwen_path"]
    pretrained_wan_path = os.environ["wan_path"]

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

    generator = Generator(
        torch_dtype = torch.bfloat16,
        pretrained_weights=sd_safe_tensor_path_json_format,
        tokenizer_path = f"{pretrained_qwen_path}/tokenizer",
        learning_rate=dataset_yaml['learning_rate'],
        use_gradient_checkpointing=dataset_yaml['use_gradient_checkpointing'],
        pretrained_ckpt_path_gen = dataset_yaml['pretrained_ckpt_path_gen']
    )
    
    if args.offload_dis_t5:
        dis_pretrained_weights = f"""[
            "{pretrained_wan_path}/diffusion_pytorch_model.safetensors",
            "{pretrained_wan_path}/Wan2.1_VAE.pth"
        ]"""
    else:
        dis_pretrained_weights = f"""[
            "{pretrained_wan_path}/diffusion_pytorch_model.safetensors",
            "{pretrained_wan_path}/models_t5_umt5-xxl-enc-bf16.pth",
            "{pretrained_wan_path}/Wan2.1_VAE.pth"
        ]"""

    discriminator = Discriminator(
        torch_dtype = torch.bfloat16,
        pretrained_weights=dis_pretrained_weights,
        dis_tokenizer_path = f'{pretrained_wan_path}/google/umt5-xxl',
        learning_rate=dataset_yaml['learning_rate_dis'],
        use_gradient_checkpointing=dataset_yaml['use_gradient_checkpointing'],
        pretrained_ckpt_path_dis = dataset_yaml['pretrained_ckpt_path_dis']
    )
    
    cri_gan = GANLoss(gan_type = dataset_yaml['gan_type'], loss_weight = dataset_yaml['gan_loss_weight'], 
                      real_label_val=dataset_yaml['real_label_val'], fake_label_val = dataset_yaml['fake_label_val'])
    
    optimizer_g = generator.configure_optimizers()
    optimizer_d = discriminator.configure_optimizers()
  
    net_lpips = lpips.LPIPS(net='vgg')

    # dino_model = AutoModel.from_pretrained(
    #     "xxxxxxxxxxxxxx/aigc_pretrained/dinov3-vith16plus-pretrain-lvd1689m", 
    #     torch_dtype=torch.bfloat16,
    #     device_map="cpu", 
    # )
    # net_lpips = DINOv3PerceptualLoss(dino_model, layers=[7, 15, 23, 31], weights=[4.0, 3.0, 2.0, 1.0])
    net_lpips.requires_grad_(False)
    net_lpips.eval()
    
    generator.pipe.device = accelerator.device
    discriminator.pipe.device = accelerator.device
    
    generator, discriminator, net_lpips, optimizer_g, optimizer_d, dataloader = accelerator.prepare(
        generator, discriminator, net_lpips, optimizer_g, optimizer_d, dataloader
    )
    
    if accelerator.is_main_process:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        workdir = os.path.join("./experiments", dataset_yaml['exp_tag'], current_time)
        os.makedirs(workdir, exist_ok=False)
        trt_logger = TensorBoardLogger(workdir, name="tensorboard")

        # backup config file
        yaml_file_name = os.path.basename(args.mmaigc_dataset_yml)
        target_path = os.path.join(workdir, yaml_file_name)
        shutil.copy2(args.mmaigc_dataset_yml, target_path)
        
        # plot model structure
        filename = os.path.join(workdir, 'generator_model_structure.txt')
        with open(filename, "a") as f:
            with redirect_stdout(f):
                print(generator.module.pipe)
                
        filename = os.path.join(workdir, 'discriminator_model_structure.txt')
        with open(filename, "a") as f:
            with redirect_stdout(f):
                print(discriminator.module.pipe)
        
        # 获取生成器可训练参数名字并写入文件
        filename = os.path.join(workdir, 'generator_trainable_parameters.txt')
        with open(filename, "a") as f:
            with redirect_stdout(f):
                for name, param in generator.named_parameters():
                    if param.requires_grad:
                        print(name)

        # 获取判别器可训练参数名字并写入文件
        filename = os.path.join(workdir, 'discriminator_trainable_parameters.txt')
        with open(filename, "a") as f:
            with redirect_stdout(f):
                print("Trainable parameters in discriminator:")
                for name, param in discriminator.named_parameters():
                    if param.requires_grad:
                        print(name)
        
        # plot generator.pipe_G.scheduler
        data_np = generator.module.pipe.scheduler.sigmas.numpy()
        plt.plot(data_np, marker='o')
        plt.title('Line Plot of sigmas')
        plt.xlabel('timesteps_id')
        plt.ylabel('sigma')
        # 显示网格（可选）
        plt.grid(True)
        # 保存图像到硬盘
        plt.savefig(os.path.join(workdir, 'line_plot_G.png')) 
        
        plt.clf()
        
        data_np = discriminator.module.pipe.scheduler.sigmas.numpy()
        plt.plot(data_np, marker='o')
        plt.title('Line Plot of sigmas')
        plt.xlabel('timesteps_id')
        plt.ylabel('sigma')
        # 显示网格（可选）
        plt.grid(True)
        # 保存图像到硬盘
        plt.savefig(os.path.join(workdir, 'line_plot_D.png')) 
    


    # if generator.module.pipe.text_encoder is not None:
    #     del generator.module.pipe.text_encoder
    #     # generator.module.pipe.text_encoder = None
    #     gc.collect()
    #     torch.cuda.empty_cache()
    
    # null_text_ratio = dataset_yaml['null_text_ratio']
    crop_latent_for_rgb_size = 60
    # assert crop_latent_for_rgb_size <= 44
    rgb_w = dataset_yaml['rgb_w']
    lpips_w = dataset_yaml['lpips_w']

    iteration = 0
    log_iteration = 0
    
    # 记录判别器需要训练的参数的名字
    dis_trainable_param_names = [
        name for name, param in discriminator.named_parameters() 
        if param.requires_grad
    ]

    for epoch in range(dataset_yaml['max_epochs']):
        for batch_idx, data in enumerate(dataloader, 0):
            accelerator.print(batch_idx, iteration)
            """
                一些预处理
                vae_new(lq_rgb) = gt_latents
            """
            with torch.no_grad():
                gt_pil, lq_pil = random_crop_pair(data, 512, 512)

                inputs_posi = {
                    "prompt": data["text"] if random.random() < 0.75 else ""
                }
                inputs_nega = {"negative_prompt": ""}
                inputs_shared = {
                    # Assume you are using this pipeline for inference,
                    # please fill in the input parameters.
                    "input_image": gt_pil,
                    "height": gt_pil.size[1],
                    "width": gt_pil.size[0],
                    "condition_image": lq_pil,
                    # Please do not modify the following parameters
                    # unless you clearly know what this will cause.
                    "cfg_scale": 1,
                    "rand_device": accelerator.device
                }
                
                for unit in generator.module.pipe.units:
                    inputs_shared, inputs_posi, inputs_nega = generator.module.pipe.unit_runner(unit, generator.module.pipe, inputs_shared, inputs_posi, inputs_nega)
        
                lq_rgb = inputs_shared['condition_rgb'].detach()
                lq_latents = inputs_shared['condition_latents'].detach()
                gt_rgb = inputs_shared['input_rgb'].detach()
                gt_latents = inputs_shared['input_latents'].detach()
                
                pre_saved_prompt_emb = {}
                pre_saved_prompt_emb['prompt_emb'] = inputs_posi['prompt_emb']
                pre_saved_prompt_emb['prompt_emb_mask'] = inputs_posi['prompt_emb_mask']

                caption = data['text']
                # get prompt embding using caption
                if args.offload_dis_t5:
                    # get prompt_emb from dataloader
                    raise NotImplementedError("not support offload_dis_t5")
                else:
                    prompt_emb = discriminator.module.pipe.prompter.encode_prompt(caption, positive=True, device=accelerator.device)
                
                # print(prompt_emb['context'].shape, prompt_emb['context'].dtype, prompt_emb['context'].device)
                # torch.Size([1, 512, 4096]) torch.bfloat16 cuda:0
                
            """
                train G
            """
            # D训练的参数 不要产生梯度 否则影响D的梯度累积                
            for name, param in discriminator.named_parameters():
                if name in dis_trainable_param_names:
                    param.requires_grad = False

            sync_gradients = ((batch_idx + 1) % gradient_accumulation_steps == 0) or (batch_idx == len(dataloader) - 1)
            ctx_g = accelerator.no_sync(generator) if not sync_gradients else nullcontext()

            with ctx_g:
                new_lq_latents = generator.module.pipe.new_vae.encode(lq_rgb, tiled=False)

                start_point = 0
                gen_start_point = 750 # 500 250
                random_timestep_id_for_dis = torch.randint(start_point, discriminator.module.pipe.scheduler.num_train_timesteps, (1,))
                random_timestep = discriminator.module.pipe.scheduler.timesteps[random_timestep_id_for_dis].to(device=accelerator.device)
                
                fixed_timestep_id = torch.randint(gen_start_point, gen_start_point + 1, (1,))
                fixed_timestep = generator.module.pipe.scheduler.timesteps[fixed_timestep_id].to(device=accelerator.device)
                
                random_timestep_id_for_lq_stream = torch.randint(gen_start_point, 1000, (1,))
                random_timestep_for_lq_stream = generator.module.pipe.scheduler.timesteps[random_timestep_id_for_lq_stream].to(device=accelerator.device)
                
                one_step_sigma = generator.module.pipe.scheduler.sigmas[fixed_timestep_id].to(dtype=torch.bfloat16, device=accelerator.device)
                one_step_sigma_lq_stream = generator.module.pipe.scheduler.sigmas[random_timestep_id_for_lq_stream].to(dtype=torch.bfloat16, device=accelerator.device)
                
                noise = torch.randn_like(lq_latents)
                noisy_latents = generator.module.pipe.scheduler.add_noise(new_lq_latents.detach(), noise, fixed_timestep)
                noisy_lq_latents = generator.module.pipe.scheduler.add_noise(lq_latents.detach(), noise, random_timestep_for_lq_stream)
                gan_w_ratio = get_gan_weight(one_step_sigma_lq_stream, one_step_sigma)
                
                # 一定概率使用纯lq，不使用噪声，为了保真指标
                if random.random() < dataset_yaml['lq_no_noise_prob']:
                    noisy_lq_latents = lq_latents
                    gan_w_ratio = get_gan_weight(0, one_step_sigma)

                noise_pred = generator(noisy_latents, noisy_lq_latents, fixed_timestep, **pre_saved_prompt_emb).to(dtype=torch.bfloat16)
                training_pred = noisy_latents + (0 - one_step_sigma) * noise_pred

                # select part of model_pred
                # target_training_pred, target_gt_rgb = crop_for_bchw(training_pred, crop_latent_for_rgb_size, gt_rgb)
                training_pred_rgb = generator.module.pipe.vae.decode(training_pred)
                new_lq_latents_rgb = generator.module.pipe.vae.decode(new_lq_latents)
                    
                loss_rgb_mse = torch.nn.functional.mse_loss(training_pred_rgb.float(), gt_rgb.detach().float())
                loss_rgb_mse = rgb_w * loss_rgb_mse
                
                loss_lpips  = net_lpips(training_pred_rgb.float(), gt_rgb.detach().float()).mean()
                loss_lpips = lpips_w * loss_lpips
                
                # noise = torch.randn_like(lq_latents)
                # noisy_latents_fake = discriminator.module.pipe.scheduler.add_noise(training_pred, noise, random_timestep)
                # noisy_latents_fake = training_pred

                with torch.no_grad():
                    real_d_pred = discriminator(deal_discriminator_condition(gt_latents, lq_latents, dataset_yaml['use_dual_condition_flag']), random_timestep, prompt_emb).detach().clone()
                fake_g_pred = discriminator(deal_discriminator_condition(training_pred, lq_latents, dataset_yaml['use_dual_condition_flag']), random_timestep, prompt_emb)
                tmp1 = cri_gan(real_d_pred - adaptive_relaxed_mean(fake_g_pred, output_size = dataset_yaml['relaxed_mean_size']), False, is_disc=False)
                tmp2 = cri_gan(fake_g_pred - adaptive_relaxed_mean(real_d_pred, output_size = dataset_yaml['relaxed_mean_size']), True, is_disc=False)
                loss_g_gan = (tmp1 + tmp2) / 2

                loss_new_vae_lq = torch.nn.functional.mse_loss(new_lq_latents_rgb.float(), gt_rgb.float())
                
                
                total_loss = (loss_rgb_mse + loss_lpips + loss_g_gan * gan_w_ratio + loss_new_vae_lq)
                total_loss = total_loss / gradient_accumulation_steps
                accelerator.backward(total_loss)
            
            if sync_gradients:
                optimizer_g.step()
                optimizer_g.zero_grad()
            
            ################################################################################################################################################################
            """
                train D
            """
            for name, param in discriminator.named_parameters():
                if name in dis_trainable_param_names:
                    param.requires_grad = True

            # for name, param in discriminator.named_parameters():
            #     if param.requires_grad and param.grad is not None:
            #         accelerator.print(f"[Warning] after G training: {name} has non-None grad")

            ctx_d = accelerator.no_sync(discriminator) if not sync_gradients else nullcontext()
                
            with ctx_d:
                fake_d_pred = fake_g_pred.detach().clone()
                real_d_pred = discriminator(deal_discriminator_condition(gt_latents, lq_latents, dataset_yaml['use_dual_condition_flag']), random_timestep, prompt_emb)
                tmp_3 = cri_gan(real_d_pred - adaptive_relaxed_mean(fake_d_pred, output_size = dataset_yaml['relaxed_mean_size']), True, is_disc=True)
                # cal R1 loss 
                noised_gt_latents = add_gaussian_noise(gt_latents, variance=dataset_yaml['variance'])
                real_d_pred_R1 = discriminator(deal_discriminator_condition(noised_gt_latents, lq_latents, dataset_yaml['use_dual_condition_flag']), random_timestep, prompt_emb)
                tmp_4 = torch.nn.functional.mse_loss(real_d_pred.float(), real_d_pred_R1.float())
                tmp_4 = dataset_yaml['r1_regularization'] * tmp_4
                
                # accelerator.backward(tmp_3 + tmp_4)

                fake_d_pred = discriminator(deal_discriminator_condition(training_pred.detach(), lq_latents, dataset_yaml['use_dual_condition_flag']), random_timestep, prompt_emb)
                tmp_5 = cri_gan(fake_d_pred - adaptive_relaxed_mean(real_d_pred.detach(), output_size = dataset_yaml['relaxed_mean_size']), False, is_disc=True)
                # cal R2 loss
                # fake_d_pred_R2 = discriminator(add_gaussian_noise(training_pred.detach(), variance=dataset_yaml['variance']), random_timestep, prompt_emb)
                # tmp_6 = torch.nn.functional.mse_loss(fake_d_pred.detach().float(), fake_d_pred_R2.float())
                # tmp_6 = dataset_yaml['r2_regularization'] * tmp_6
                tmp_6 = 0

                accelerator.backward((tmp_3*0.5 + tmp_4 + tmp_5*0.5 + tmp_6) / gradient_accumulation_steps)
            
            if accelerator.is_main_process:
                with torch.no_grad():
                    print(real_d_pred - adaptive_relaxed_mean(fake_d_pred, output_size = dataset_yaml['relaxed_mean_size']))

            if sync_gradients:
                optimizer_d.step()
                optimizer_d.zero_grad()

            if accelerator.is_main_process:
                trt_logger.log_metrics({
                    "loss_rgb_mse": loss_rgb_mse.item(), 
                    "loss_lpips": loss_lpips.item(),
                    "loss_g_gan_tmp1": tmp1.item(),
                    "loss_g_gan_tmp2": tmp2.item(),
                    "loss_new_vae_lq": loss_new_vae_lq.item(),
                    "loss_d_real": tmp_3.item(),
                    "loss_d_real_r1": tmp_4.item() if torch.is_tensor(tmp_4) else tmp_4,
                    "loss_d_fake": tmp_5.item(),
                    "loss_d_fake_r2": tmp_6.item() if torch.is_tensor(tmp_6) else tmp_6
                    }, step = log_iteration)
                log_iteration += 1
            
            if sync_gradients:
                iteration += 1

                if accelerator.is_main_process:
                    if iteration % dataset_yaml['viz_iters'] == 1:
                        print(caption)
                        
                        # generator.module.save_to_disk(iteration, lq = lq_rgb, gt = gt_rgb, pred_latent=training_pred,
                        #                     savedir = workdir)
                        with torch.no_grad():
                            res_img = generator.module.pipe.vae_output_to_image(training_pred_rgb)
                            res_img2 = generator.module.pipe.vae_output_to_image(new_lq_latents_rgb)
                        imgs = [gt_pil, lq_pil, res_img, res_img2]
                        h = max(im.height for im in imgs)
                        total_w = sum(im.width for im in imgs)
                        canvas = Image.new('RGB', (total_w, h))
                        x = 0
                        for im in imgs:
                            canvas.paste(im, (x, 0))
                            x += im.width
                        canvas.save(os.path.join(workdir, f"iter_{iteration}.png"))

                        # save prompt to txt
                        with open(os.path.join(workdir, f"iter_{iteration}.txt"), "w") as f:
                            f.write(inputs_posi['prompt'])

                    if iteration % dataset_yaml['save_ckpt_iters'] == 1:
                        generator.module.save_ckpt(
                            os.path.join(workdir, 'checkpoints'), iter = iteration, tag = "gen"
                        )

                    if iteration % dataset_yaml['save_ckpt_iters'] == 1:
                        discriminator.module.save_ckpt(
                            os.path.join(workdir, 'checkpoints'), iter = iteration, tag = "dis"
                        )
            
        

if __name__ == '__main__': 
    args = parse_args()
    if args.task == "data_process":
        raise NotImplementedError("")
    elif args.task == "train":
        train(args)