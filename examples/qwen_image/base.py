import torch
import os

class BaseModelForT2ILoRA(torch.nn.Module):
    def __init__(
        self,
        learning_rate=1e-4,
        use_gradient_checkpointing=True,
    ):
        super().__init__()
        # Set parameters
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing


    @torch.no_grad()
    def save_ckpt(self, ckpt_workdir, iter, tag):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.state_dict()
        lora_state_dict = {}
        
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        
        if not os.path.exists(ckpt_workdir):
            os.makedirs(ckpt_workdir, exist_ok=True)
            
        torch.save(lora_state_dict, os.path.join(ckpt_workdir, f'net_{tag}_iter_{iter}.pth'))