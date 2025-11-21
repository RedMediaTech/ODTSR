import os
import torch
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from base import BaseModelForT2ILoRA
from einops import rearrange
import json


class Discriminator(BaseModelForT2ILoRA):
    def __init__(
        self,
        torch_dtype=torch.float16, pretrained_weights=None,
        dis_tokenizer_path = None,
        learning_rate=1e-4, use_gradient_checkpointing=True,
        pretrained_ckpt_path_dis = None
    ):
        super().__init__(learning_rate=learning_rate, use_gradient_checkpointing=use_gradient_checkpointing)
        
        model_configs = []

        if pretrained_weights is not None:
            pretrained_weights = json.loads(pretrained_weights)
            model_configs += [ModelConfig(path=path) for path in pretrained_weights]

        
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, tokenizer_config=ModelConfig(dis_tokenizer_path))
        
        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True, shift=3.0)

        self.pipe.freeze_except([])
        
        # dit训练
        self.pipe.dit.train()
        self.pipe.dit.requires_grad_(True)

        self.D_last_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.PReLU(num_parameters=32),
            torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        )
        self.D_last_conv.to(dtype=torch.bfloat16)
        self.D_last_conv.requires_grad_(True)

        self.use_gradient_checkpointing = use_gradient_checkpointing

        if pretrained_ckpt_path_dis:
            state_dict = torch.load(pretrained_ckpt_path_dis, map_location='cpu')
            self.load_state_dict(state_dict, strict = False)
        
    def configure_optimizers(self):
        param_groups = []

        denoising_params = [
            p for p in self.pipe.dit.parameters() 
            if p.requires_grad
        ]
        param_groups.append({
            'params': denoising_params,
            'lr': self.learning_rate
        })
        
        d_last_conv_params = [
            p for p in self.D_last_conv.parameters() 
            if p.requires_grad
        ]
        param_groups.append({
            'params': d_last_conv_params,
            'lr': self.learning_rate * 1.0
        })
        
        # opt = torch.optim.Adam(
        #     param_groups,
        #     lr=self.learning_rate,  # 注意：这里的基础学习率将被参数组覆盖
        #     betas=(0.0, 0.9)
        # )
        opt = torch.optim.RMSprop(
            param_groups,
            lr=self.learning_rate,
            alpha=0.9,  # 对应 Adam 的 beta2
            momentum=0.0  # 对应 Adam 的 beta1=0.0
        )
        return opt
    
    def forward(self, latents, timestep, context):
        """
            latents: [1,16,h,w]   or   [1,16,2,h,w]
            timestep: 1,
            context: 1,len,4096
        """
        if len(latents.shape) == 4:
            out = self.pipe.model_fn(self.pipe.dit, 
                            latents = latents.unsqueeze(2),
                            timestep = timestep,
                            context = context,
                            use_gradient_checkpointing = True
                            )
            # 1,16,1,h,w -> 1,16,h,w
            out = out[0].permute(1,0,2,3)
        else:
            assert len(latents.shape) == 5 and latents.shape[2] == 2
            out = self.pipe.model_fn(self.pipe.dit, 
                            latents = latents,
                            timestep = timestep,
                            context = context,
                            use_gradient_checkpointing = True
                            )
            # 1,16,2,h,w -> 1,16,h,w
            out = out[0, :, 0:1, :, :].permute(1,0,2,3)
        out = self.D_last_conv(out)
        return out


# test
if __name__ == "__main__":
    json_paths = """[
        "path2wan2.1/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "path2wan2.1/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "path2wan2.1/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
    ]"""

    model = Discriminator(pretrained_weights = json_paths)

    latents = torch.zeros((1,16,1,100,100), dtype=torch.bfloat16)
    timestep = torch.zeros((1,), dtype=torch.bfloat16)
    context = torch.zeros((1,100,4096), dtype=torch.bfloat16)
    out = model(latents, timestep, context)