import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv3PerceptualLoss(nn.Module):
    def __init__(self, model, layers, weights=None, criterion_type = 'l1'):
        super().__init__()
        self.criterion_type = criterion_type
        self.model = model.eval()
        self.layers = layers
        self.weights = weights if weights is not None else [1.0] * len(layers)
        assert len(self.layers) == len(self.weights)
        self.sum_weights = sum(self.weights)

        self.activations = {}
        # 注册 hook
        for i, layer_idx in enumerate(self.layers):
            self.model.layer[layer_idx].register_forward_hook(
                self._get_hook(layer_idx)
            )

        # 避免模型参数被更新
        for p in self.model.parameters():
            p.requires_grad_(False)

        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError(f'Unknown criterion type: {self.criterion_type}')

    def _get_hook(self, layer_idx):
        def hook(module, input, output):
            # output 就是 hidden_states (B, L, C)
            self.activations[layer_idx] = output
        return hook

    def forward(self, pred, target):
        """
        pred, target: 图像张量 (B, C, H, W)，范围 [-1,1]
        """
        pred = (pred + 1.0) / 2.0
        target = (target + 1.0) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device, dtype=pred.dtype).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device, dtype=pred.dtype).view(1,3,1,1)
        pred = (pred - mean) / std
        target = (target - mean) / std

        # 通过 DINOv3 提取特征
        _ = self.model(pred)
        pred_feats = {k: v for k, v in self.activations.items()}

        _ = self.model(target)
        target_feats = {k: v for k, v in self.activations.items()}

        # 计算加权 L2 损失
        loss = 0.0

        for idx, w in zip(self.layers, self.weights):
            f_pred = pred_feats[idx]
            f_target = target_feats[idx]
            loss = loss + w / self.sum_weights * self.criterion(f_pred, f_target)

        return loss
