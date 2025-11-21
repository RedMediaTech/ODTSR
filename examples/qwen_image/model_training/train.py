import torch, os, json
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ImageDataset, ModelLogger, launch_training_task, qwen_image_parser
from diffsynth.extensions.realesrgan.dataset import PairedSROnlineTxtDataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def replace_linear_with_duallora(model, patterns, rank, alpha1, alpha2, use_fp8=False):
    """
    先冻结model所有参数，
    将匹配patterns的nn.Linear替换为DualLoRALinear
    不匹配的替换为 LinearFP8Wrapper
    """
    def to_fp8(tensor):
        # 仅在 use_fp8 时转换
        if use_fp8:
            return tensor.to(dtype=torch.float8_e4m3fn)
        return tensor

    # 冻结全部参数
    for p in model.parameters():
        p.requires_grad = False

    def _replace_module(parent, name_prefix=""):
        for name, module in list(parent.named_children()):
            full_name = f"{name_prefix}{name}"
            if isinstance(module, torch.nn.Linear):
                module.weight.data = to_fp8(module.weight.data)
                if module.bias is not None:
                    module.bias.data = to_fp8(module.bias.data)

                if any(p in full_name for p in patterns):
                    # 替换为 DualLoRALinear
                    new_module = DualLoRALinear(module, rank, alpha1, alpha2)
                    setattr(parent, name, new_module)
                    print(f"[lora] {full_name} -> DualLoRALinear")
                else: 
                    # 替换为自动精度转换的 FP8LinearWrapper
                    new_module = LinearFP8Wrapper(module)
                    setattr(parent, name, new_module)
                    print(f"[cast] {full_name} -> LinearFP8Wrapper (fp8={use_fp8})")
            else:
                _replace_module(module, name_prefix=full_name + ".")
    
    _replace_module(model)

class LinearFP8Wrapper(torch.nn.Module):
    """普通 Linear forward 时自动 cast 到输入 dtype"""
    def __init__(self, linear):
        super().__init__()
        assert isinstance(linear, torch.nn.Linear)
        self.weight = linear.weight
        self.bias = linear.bias
        # 冻结参数
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x):
        w = self.weight.to(dtype=x.dtype)
        b = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(x, w, b)

class DualLoRALinear(torch.nn.Module):
    """
    对序列前一半/后一半分别使用不同 LoRA 的 Linear 包装器。
    原 Linear 权重原地共享，不额外复制。
    """
    def __init__(self, linear, rank, alpha1, alpha2):
        super().__init__()
        assert isinstance(linear, torch.nn.Linear)
        self.rank    = rank
        self.alpha1  = alpha1
        self.alpha2  = alpha2
        self.linear  = linear

        dev = linear.weight.device
        dt  = torch.bfloat16

        # 两套低秩矩阵，参数量 2 * (in * rank + rank * out)
        self.lora_A1 = torch.nn.Linear(linear.in_features, rank, bias=False).to(device=dev, dtype=dt)
        self.lora_B1 = torch.nn.Linear(rank, linear.out_features, bias=False).to(device=dev, dtype=dt)
        self.lora_A2 = torch.nn.Linear(linear.in_features, rank, bias=False).to(device=dev, dtype=dt)
        self.lora_B2 = torch.nn.Linear(rank, linear.out_features, bias=False).to(device=dev, dtype=dt)

        self.scaling1 = alpha1 /  max(1, rank)
        self.scaling2 = alpha2 /  max(1, rank)

        torch.nn.init.normal_(self.lora_A1.weight, std=1.0 / rank)
        torch.nn.init.zeros_(self.lora_B1.weight)
        torch.nn.init.normal_(self.lora_A2.weight, std=1.0 / rank)
        torch.nn.init.zeros_(self.lora_B2.weight)

    def _base_linear(self, x):
        # 仅在计算里把 float8 权重/偏置转成 x.dtype（bf16）
        w = self.linear.weight.detach().to(dtype=x.dtype)
        b = None
        if self.linear.bias is not None:
            b = self.linear.bias.detach().to(dtype=x.dtype)
        return torch.nn.functional.linear(x, w, b)

    def forward(self, x):
        if x.ndim == 2:
            y = self._base_linear(x)  # (B, C_out)
            delta1 = self.lora_B1(self.lora_A1(x)) * self.scaling1
            delta2 = self.lora_B2(self.lora_A2(x)) * self.scaling2
            y1 = y + delta1
            y2 = y + delta2
            return torch.stack([y1, y2], dim=1)  # (B, 2, C_out)
        else:
            B, L2, _ = x.shape
            assert L2 % 2 == 0, "sequence length must be even"
            L = L2 // 2

            y = self._base_linear(x)                           # [B, 2L, C_out]
            x1 = x[:, :L, :]                             # [B, L, C_in]
            x2 = x[:, L:, :]                             # [B, L, C_in]

            delta1 = self.lora_B1(self.lora_A1(x1)) * self.scaling1   # [B, L, C_out]
            delta2 = self.lora_B2(self.lora_A2(x2)) * self.scaling2   # [B, L, C_out]

            y[:, :L] += delta1
            y[:, L:] += delta2
            return y

class QwenImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
    ):
        super().__init__()
        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        if tokenizer_path is not None:
            self.pipe = QwenImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, tokenizer_config=ModelConfig(tokenizer_path))
        else:
            self.pipe = QwenImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        
        # Reset training scheduler (do it in each training step)
        self.pipe.scheduler.set_timesteps(1000, training=True, shift=1.0)
        
        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            # model = self.add_lora_to_model(
            #     getattr(self.pipe, lora_base_model),
            #     target_modules=lora_target_modules.split(","),
            #     lora_rank=lora_rank
            # )
            model = self.add_custom_dual_lora(
                getattr(self.pipe, lora_base_model),
                lora_rank=lora_rank
            )
            
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        
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
        replace_linear_with_duallora(model, patterns, rank=lora_rank, alpha1=0, alpha2=lora_rank)

    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["text"]}
        inputs_nega = {"negative_prompt": ""}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": data["gt"],
            "height": data["gt"].size[1],
            "width": data["gt"].size[0],
            "condition_image": data["lq"],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss



if __name__ == "__main__":
    parser = qwen_image_parser()
    args = parser.parse_args()
    dataset = PairedSROnlineTxtDataset(
        split="train",
        args = args
    )

    model = QwenImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path="path2qwenimage/Qwen-Image/tokenizer",
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt, # "pipe.dit."
    )
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    launch_training_task(
        dataset, model, model_logger, optimizer, scheduler,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
