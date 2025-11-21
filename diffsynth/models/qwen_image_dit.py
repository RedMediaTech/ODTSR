import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, List
from einops import rearrange
from .sd3_dit import TimestepEmbeddings, RMSNorm
from .flux_dit import AdaLayerNorm
import os
import matplotlib.pyplot as plt
import numpy as np

debug = False

class ApproximateGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)

def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]]
):
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    return x_out.type_as(x)


class QwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(1024)
        neg_index = torch.arange(1024).flip(0) * -1 - 1
        self.pos_freqs = torch.cat([
            self.rope_params(pos_index, self.axes_dim[0], self.theta),
            self.rope_params(pos_index, self.axes_dim[1], self.theta),
            self.rope_params(pos_index, self.axes_dim[2], self.theta),
        ], dim=1)
        self.neg_freqs = torch.cat([
            self.rope_params(neg_index, self.axes_dim[0], self.theta),
            self.rope_params(neg_index, self.axes_dim[1], self.theta),
            self.rope_params(neg_index, self.axes_dim[2], self.theta),
        ], dim=1)
        self.rope_cache = {}
        self.scale_rope = scale_rope
        
    def rope_params(self, index, dim, theta=10000):
        """
            Args:
                index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(
            index,
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim))
        )
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs
    
    def forward(self, video_fhw, txt_seq_lens, device):
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        frame, height, width = video_fhw
        rope_key = f"{frame}_{height}_{width}"

        if rope_key not in self.rope_cache:
            seq_lens = frame * height * width
            freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
            freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)
            freqs_frame = freqs_pos[0][:frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
            if self.scale_rope:
                freqs_height = torch.cat(
                    [
                        freqs_neg[1][-(height - height//2):],
                        freqs_pos[1][:height//2]
                    ], 
                    dim=0
                )
                freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
                freqs_width = torch.cat(
                    [
                        freqs_neg[2][-(width - width//2):],
                        freqs_pos[2][:width//2]
                    ], 
                    dim=0
                )
                freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
                
            else:
                freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
                freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)
            
            freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
            self.rope_cache[rope_key] = freqs.clone().contiguous()
        vid_freqs = self.rope_cache[rope_key]

        if self.scale_rope:
            max_vid_index = max(height // 2, width // 2)
        else:
            max_vid_index = max(height, width)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index: max_vid_index + max_len, ...]
        return vid_freqs, txt_freqs


class QwenFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(dim * 4)
        self.net = nn.ModuleList([])
        self.net.append(ApproximateGELU(dim, inner_dim))
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

def compute_and_save_attention(joint_q, joint_k, save_dir="attn_debug"):
    """
    自动保存每次调用的 attention q@k^T，并生成可视化热力图（拼成 4x6 的head网格图）。
    设定：text=0:18, noise=18:18+1024。
    """
    if not hasattr(compute_and_save_attention, "layer_counter"):
        compute_and_save_attention.layer_counter = 0
    layer_idx = compute_and_save_attention.layer_counter

    os.makedirs(save_dir, exist_ok=True)

    # 1. scaled dot-product attention
    scale = joint_q.size(-1) ** -0.5
    attn_scores = torch.matmul(joint_q, joint_k.transpose(-2, -1)) * scale
    attn_scores = attn_scores.softmax(dim=-1)
    
    # 2. 取 noise->text 注意力
    text_start, text_end = 0, 18
    noise_start, noise_end = 18, 18 + 1024
    attn_noise_to_text = attn_scores[:, :, noise_start:noise_end, text_start:text_end]  # [B,H,1024,18]

    # 3. 对 text keys 求平均
    avg_key = attn_noise_to_text.mean(dim=-1)  # [B,H,1024]

    # 4. 为所有 head 生成热力图
    fig, axes = plt.subplots(4, 6, figsize=(12, 8))
    axes = axes.flatten()

    for head_index in range(24):
        avg_head = avg_key[:, head_index, :]  # [B,1024]
        heatmap = avg_head[0].reshape(32, 32).detach().cpu().float().numpy()

        ax = axes[head_index]
        im = ax.imshow(heatmap, cmap='viridis')
        ax.set_title(f"H{head_index}", fontsize=8)
        ax.axis('off')

    # 去掉多余子图（如果head<24）
    for i in range(24, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    png_save_path = os.path.join(save_dir, f"attn_heatmap_layer{layer_idx:02d}_allheads.png")
    plt.savefig(png_save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()

    print(f"✅ Saved combined attention map (layer {layer_idx}) at: {png_save_path}")

    # 5. 更新计数器
    compute_and_save_attention.layer_counter += 1
    

class QwenDoubleStreamAttention(nn.Module):
    def __init__(
        self,
        dim_a,
        dim_b,
        num_heads,
        head_dim,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(dim_a, dim_a)
        self.to_k = nn.Linear(dim_a, dim_a)
        self.to_v = nn.Linear(dim_a, dim_a)
        self.norm_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_k = RMSNorm(head_dim, eps=1e-6)

        self.add_q_proj = nn.Linear(dim_b, dim_b)
        self.add_k_proj = nn.Linear(dim_b, dim_b)
        self.add_v_proj = nn.Linear(dim_b, dim_b)
        self.norm_added_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_added_k = RMSNorm(head_dim, eps=1e-6)

        self.to_out = torch.nn.Sequential(nn.Linear(dim_a, dim_a))
        self.to_add_out = nn.Linear(dim_b, dim_b)

    def forward(
        self,
        image: torch.FloatTensor,
        text: torch.FloatTensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        img_q, img_k, img_v = self.to_q(image), self.to_k(image), self.to_v(image)
        txt_q, txt_k, txt_v = self.add_q_proj(text), self.add_k_proj(text), self.add_v_proj(text)
        seq_txt = txt_q.shape[1]

        img_q = rearrange(img_q, 'b s (h d) -> b h s d', h=self.num_heads)
        img_k = rearrange(img_k, 'b s (h d) -> b h s d', h=self.num_heads)
        img_v = rearrange(img_v, 'b s (h d) -> b h s d', h=self.num_heads)

        txt_q = rearrange(txt_q, 'b s (h d) -> b h s d', h=self.num_heads)
        txt_k = rearrange(txt_k, 'b s (h d) -> b h s d', h=self.num_heads)
        txt_v = rearrange(txt_v, 'b s (h d) -> b h s d', h=self.num_heads)

        img_q, img_k = self.norm_q(img_q), self.norm_k(img_k)
        txt_q, txt_k = self.norm_added_q(txt_q), self.norm_added_k(txt_k)
        
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_q = apply_rotary_emb_qwen(img_q, img_freqs)
            img_k = apply_rotary_emb_qwen(img_k, img_freqs)
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs)

        joint_q = torch.cat([txt_q, img_q], dim=2)
        joint_k = torch.cat([txt_k, img_k], dim=2)
        joint_v = torch.cat([txt_v, img_v], dim=2)

        if debug:
            L_txt = txt_q.size(2)
            L_img = img_q.size(2)
            L_joint = L_txt + L_img
            print(f"lengths: txt-{L_txt}, L_img-{L_img}")

            gamma = float(os.environ.get("attention_gamma", 1.0))
            eps = 1e-5

            # compute_and_save_attention(joint_q, joint_k)
            attn_bias = torch.zeros(L_joint, L_joint, device=joint_q.device, dtype=joint_q.dtype)
            tmp =  torch.log(torch.tensor(gamma, dtype=joint_q.dtype, device=joint_q.device) + eps)
            print(tmp)
            attn_bias[L_txt:, :L_txt] = tmp
            attn_bias[:L_txt, L_txt:] = tmp
            joint_attn_out = torch.nn.functional.scaled_dot_product_attention(joint_q, joint_k, joint_v, attn_mask=attn_bias)
        else:
            joint_attn_out = torch.nn.functional.scaled_dot_product_attention(joint_q, joint_k, joint_v)

        joint_attn_out = rearrange(joint_attn_out, 'b h s d -> b s (h d)').to(joint_q.dtype)

        txt_attn_output = joint_attn_out[:, :seq_txt, :]
        img_attn_output = joint_attn_out[:, seq_txt:, :]

        img_attn_output = self.to_out(img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_attention_heads: int, 
        attention_head_dim: int, 
        eps: float = 1e-6,
    ):    
        super().__init__()
        
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim), 
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenDoubleStreamAttention(
            dim_a=dim,
            dim_b=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = QwenFeedForward(dim=dim, dim_out=dim)

        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True), 
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = QwenFeedForward(dim=dim, dim_out=dim)
    
    def _modulate(self, x, mod_params):
        if mod_params.ndim == 2:
            shift, scale, gate = mod_params.chunk(3, dim=-1)
            return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1) 
        else:
            # [B, 2L, 3*dim]
            shift, scale, gate = mod_params.chunk(3, dim=-1)
            return x * (1 + scale) + shift, gate   

    def forward(
        self,
        image: torch.Tensor,  
        text: torch.Tensor,
        temb: torch.Tensor, 
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b,L,c = image.shape
        L = L//2
        img_mod_attn, img_mod_mlp = self.img_mod(temb).chunk(2, dim=-1)  # [B, 3*dim] each
        # after replace img_mod's linear, [B, 2, 3*dim] each
        if img_mod_attn.ndim == 3:
            assert img_mod_attn.shape[1] == 2 # 这里是2 是因为img_mod用两套参数区别对待了temb输入
            # broadcast to [B, 2L, 3*dim]
            img_mod_attn = torch.repeat_interleave(img_mod_attn, repeats=L, dim=1)
            img_mod_mlp  = torch.repeat_interleave(img_mod_mlp, repeats=L, dim=1)
            
        txt_mod_attn, txt_mod_mlp = self.txt_mod(temb).chunk(2, dim=-1)  # [B, 3*dim] each

        img_normed = self.img_norm1(image)
        img_modulated, img_gate = self._modulate(img_normed, img_mod_attn)

        txt_normed = self.txt_norm1(text)
        txt_modulated, txt_gate = self._modulate(txt_normed, txt_mod_attn)

        img_attn_out, txt_attn_out = self.attn(
            image=img_modulated,
            text=txt_modulated,
            image_rotary_emb=image_rotary_emb,
        )
        
        image = image + img_gate * img_attn_out
        text = text + txt_gate * txt_attn_out

        img_normed_2 = self.img_norm2(image)
        img_modulated_2, img_gate_2 = self._modulate(img_normed_2, img_mod_mlp)

        txt_normed_2 = self.txt_norm2(text)
        txt_modulated_2, txt_gate_2 = self._modulate(txt_normed_2, txt_mod_mlp)

        img_mlp_out = self.img_mlp(img_modulated_2)
        txt_mlp_out = self.txt_mlp(txt_modulated_2)

        image = image + img_gate_2 * img_mlp_out
        text = text + txt_gate_2 * txt_mlp_out

        return text, image


class QwenImageDiT(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 60,
    ):
        super().__init__()

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=[16,56,56], scale_rope=True) 

        self.time_text_embed = TimestepEmbeddings(256, 3072, diffusers_compatible_format=True, scale=1000, align_dtype_to_timestep=True)
        self.txt_norm = RMSNorm(3584, eps=1e-6)

        self.img_in = nn.Linear(64, 3072)
        self.txt_in = nn.Linear(3584, 3072)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=3072,
                    num_attention_heads=24,
                    attention_head_dim=128,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = AdaLayerNorm(3072, single=True)
        self.proj_out = nn.Linear(3072, 64)


    def forward(
        self,
        latents=None,
        timestep=None,
        prompt_emb=None,
        prompt_emb_mask=None,
        height=None,
        width=None,
    ):
        img_shapes = [(latents.shape[0], latents.shape[2]//2, latents.shape[3]//2)]
        txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
        
        image = rearrange(latents, "B C (H P) (W Q) -> B (H W) (P Q C)", H=height//16, W=width//16, P=2, Q=2)
        image = self.img_in(image)
        text = self.txt_in(self.txt_norm(prompt_emb))

        conditioning = self.time_text_embed(timestep, image.dtype)

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=latents.device)

        for block in self.transformer_blocks:
            text, image = block(
                image=image,
                text=text,
                temb=conditioning,
                image_rotary_emb=image_rotary_emb,
            )
        
        image = self.norm_out(image, conditioning)
        image = self.proj_out(image)
        
        latents = rearrange(image, "B (H W) (P Q C) -> B C (H P) (W Q)", H=height//16, W=width//16, P=2, Q=2)
        return image
    
    @staticmethod
    def state_dict_converter():
        return QwenImageDiTStateDictConverter()



class QwenImageDiTStateDictConverter():
    def __init__(self):
        pass

    def from_civitai(self, state_dict):
        return state_dict
