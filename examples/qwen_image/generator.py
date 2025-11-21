import os
import torch
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from base import BaseModelForT2ILoRA
from model_training.train import replace_linear_with_duallora, DualLoRALinear
from einops import rearrange
import json
from copy import deepcopy

class Generator(BaseModelForT2ILoRA):
    def __init__(
        self,
        torch_dtype=torch.float16, pretrained_weights=None, tokenizer_path = None,
        learning_rate=1e-4, use_gradient_checkpointing=True,
        pretrained_ckpt_path_gen = None
    ):
        super().__init__(learning_rate=learning_rate, use_gradient_checkpointing=use_gradient_checkpointing)
        use_offload = False

        model_configs = []
        if pretrained_weights is not None:
            pretrained_weights = json.loads(pretrained_weights)
            for path in pretrained_weights:
                if use_offload and isinstance(path, list) and ('transformer' in path[0]):
                    model_configs += [ModelConfig(path=path, offload_dtype=torch.float8_e4m3fn)]
                else:
                    model_configs += [ModelConfig(path=path)]
    
        if tokenizer_path is not None:
            self.pipe = QwenImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, tokenizer_config=ModelConfig(tokenizer_path))
        else:
            self.pipe = QwenImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)

        # Reset training scheduler (do it in each training step)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        self.pipe.freeze_except([])

        # copy a new vae
        self.pipe.new_vae = deepcopy(self.pipe.vae)
        self.unfrozen(self.pipe.new_vae.encoder, type(self.pipe.new_vae.encoder.conv_in))

        # 结构修改 & fp8降低显存
        lora_base_model = 'dit' # hard core
        lora_rank = 128
        self.add_custom_dual_lora(
            getattr(self.pipe, lora_base_model),
            lora_rank=lora_rank
        )

        # 推理prompt用于初始化可训练的embeding
        # inputs_posi = {"prompt": "High Contrast, highly detailed, hyper detailed photo - realistic maximum detail, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations."}
        # pre_saved_prompt_emb = self.pipe.units[-1].process(self.pipe, **inputs_posi)
        # key: prompt_emb, prompt_emb_mask
        # print(pre_saved_prompt_emb.keys())

        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        if pretrained_ckpt_path_gen:
            state_dict = torch.load(pretrained_ckpt_path_gen, map_location='cpu')
            self.load_state_dict(state_dict, strict = False)

    def unfrozen(self, model, target_cls):
        for name, module in model.named_modules():
            # time_conv单帧用不到 在不开启find_unused_parameters的情况下会报错
            if isinstance(module, target_cls) and 'time_conv' not in name:
                for p in module.parameters():
                    p.requires_grad = True

    def add_custom_dual_lora(self, model, lora_rank):
        patterns = [
            "img_in",
            "img_mod.1",
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "to_out.0",
            "img_mlp.net.0.proj",
            "img_mlp.net.2",
        ]
        replace_linear_with_duallora(model, patterns, rank=lora_rank, alpha1=0, alpha2=lora_rank, use_fp8 = True)

    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.parameters())
        # opt = torch.optim.AdamW(trainable_modules, lr=self.learning_rate, betas=(0.9, 0.95), weight_decay=0)
        # opt = torch.optim.Adam(trainable_modules, lr=self.learning_rate, betas=(0.0, 0.9))
        opt = torch.optim.RMSprop(
            trainable_modules,
            lr=self.learning_rate,
            alpha=0.9,  # 对应 Adam 的 beta2
            momentum=0.0  # 对应 Adam 的 beta1=0.0
        )
        return opt
    
    def forward(self, noisy_latents, condition_latent, timestep, prompt_emb, prompt_emb_mask):
        b,c,h,w = noisy_latents.shape
        out = self.pipe.model_fn(self.pipe.dit, 
                                 noisy_latents,
                                 condition_latent,
                                 timestep,
                                 prompt_emb,
                                 prompt_emb_mask,
                                 h*8,
                                 w*8,
                                 use_gradient_checkpointing=True
                                 )
        return out

    @torch.no_grad()
    def infer(self,
              prompt,
              negative_prompt,
              condition_image, # already paded
              cfg_scale,
              fidelity,
              tiled,
              tile_size,
              tile_stride):
        
        # Parameters
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "input_image": None,
            "condition_image": condition_image,
            "height": condition_image.size[1], "width": condition_image.size[0],
            "seed": 42, "rand_device": self.device,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
        }
        
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        
        lq_latents = inputs_shared['condition_latents']
        new_lq_latents = self.pipe.new_vae.encode(inputs_shared['condition_rgb'], tiled=tiled, tile_size=tile_size * 8, tile_stride=tile_stride* 8)
        noise = inputs_shared['noise']
        posi_prompt_emb = {}
        posi_prompt_emb['prompt_emb'] = inputs_posi['prompt_emb']
        posi_prompt_emb['prompt_emb_mask'] = inputs_posi['prompt_emb_mask']
        nega_prompt_emb = {}
        nega_prompt_emb['prompt_emb'] = inputs_nega['prompt_emb']
        nega_prompt_emb['prompt_emb_mask'] = inputs_nega['prompt_emb_mask']

        start_timestep = 750
        fixed_timestep_id = torch.randint(start_timestep, start_timestep + 1, (1,))
        fixed_timestep = self.pipe.scheduler.timesteps[fixed_timestep_id].to(device=self.device)
        one_step_sigma = self.pipe.scheduler.sigmas[fixed_timestep_id].to(dtype=torch.bfloat16, device=self.device)

        fidelity_timestep_id = int(start_timestep + fidelity * (1000 - start_timestep) + 0.5)
        if fidelity_timestep_id == 1000:
            pass
        else:
            fidelity_timestep_id = torch.randint(fidelity_timestep_id, fidelity_timestep_id + 1, (1,))
            fidelity_timestep = self.pipe.scheduler.timesteps[fidelity_timestep_id].to(device=self.device)
            lq_latents = self.pipe.scheduler.add_noise(lq_latents.detach(), noise, fidelity_timestep)
        
        print(fidelity_timestep_id)

        noisy_latents = self.pipe.scheduler.add_noise(new_lq_latents.detach(), noise, fixed_timestep)
        b,c,h,w = noisy_latents.shape
            
        # Inference
        noise_pred_posi = self.pipe.model_fn(self.pipe.dit, noisy_latents, lq_latents, fixed_timestep,
                                             **posi_prompt_emb, height=h * 8, width=w * 8, tiled = tiled, tile_size=tile_size, tile_stride=tile_stride)
        if cfg_scale != 1.0:
            noise_pred_nega = self.pipe.model_fn(self.pipe.dit, noisy_latents, lq_latents, fixed_timestep,
                                             **nega_prompt_emb, height=h * 8, width=w * 8, tiled = tiled, tile_size=tile_size, tile_stride=tile_stride)
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        else:
            noise_pred = noise_pred_posi

        # one step prediction
        training_pred = noisy_latents + (0 - one_step_sigma) * noise_pred
        
        # Decode
        image = self.pipe.vae.decode(training_pred, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.pipe.vae_output_to_image(image)

        return image