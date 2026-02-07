import torch
import torch.nn as nn
from mmdet3d.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule

class ChamferDistanceLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, source, target):
        """
        计算两组点云的 Chamfer Distance
        source: [B, N, 3] (Predicted Control Points)
        target: [B, M, 3] (Ground Truth Sampled Points)
        """
        # 简单的 L2 Chamfer 实现
        # source -> target
        dists1 = torch.cdist(source, target) # [B, N, M]
        min_dists1, _ = torch.min(dists1, dim=2) # [B, N]
        term1 = torch.mean(min_dists1, dim=1) # [B]

        # target -> source
        min_dists2, _ = torch.min(dists1, dim=1) # [B, M]
        term2 = torch.mean(min_dists2, dim=1) # [B]

        return self.loss_weight * (torch.mean(term1) + torch.mean(term2))

@HEADS.register_module()
class PolyHead(BaseModule):
    def __init__(self, 
                 in_channels=128, 
                 num_polys=2,     # 左右两根轨道
                 num_control_points=5, # 每根轨道5个控制点
                 hidden_dim=256,
                 loss_poly=dict(type='ChamferDistanceLoss', loss_weight=1.0),
                 init_cfg=None):
        super(PolyHead, self).__init__(init_cfg)
        
        self.num_polys = num_polys
        self.num_points = num_control_points
        self.out_dim = num_polys * num_control_points * 3 # 输出 (x,y,z)
        
        # 1. 几何特征提取 (Global Context)
        # 轨道是贯穿整个场景的，需要全局感受野
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 2. MLP 回归器
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.out_dim)
        )
        
        # 3. 损失函数
        if loss_poly['type'] == 'ChamferDistanceLoss':
            self.loss_func = ChamferDistanceLoss(loss_weight=loss_poly['loss_weight'])
        else:
            self.loss_func = nn.MSELoss()

    def forward(self, x):
        """
        Input: [B, C, H, W]
        Output: [B, num_polys, num_points, 3]
        """
        B = x.shape[0]
        
        # Global Pooling -> [B, C, 1, 1] -> [B, C]
        feat = self.global_pool(x).view(B, -1)
        
        # MLP Regression
        points_pred = self.mlp(feat)
        
        # Reshape to [B, num_polys, num_points, 3]
        points_pred = points_pred.view(B, self.num_polys, self.num_points, 3)
        
        return points_pred

    def loss(self, points_pred, gt_poly_3d):
        """
        Input:
            points_pred: [B, 2, 5, 3]
            gt_poly_3d: List[List[Tensor]] (B list of polys)
        """
        loss_dict = dict()
        total_loss = 0
        
        # 由于 batch 内每帧轨道数量可能不同 (虽然通常是2根), 且 GT 格式需要对齐
        # 这里假设 gt_poly_3d 已经在 Data Pipeline 里转换成了 Point Sets
        
        for i in range(len(points_pred)):
            pred = points_pred[i].view(-1, 3) # [10, 3] 合并左右轨道的点
            
            # 获取 GT (假设 transforms.py 已经将 coefficients 转为了 points)
            # gt_poly_3d[i] 应该是一个 [M, 3] 的 tensor，包含了所有轨道的采样点
            target = gt_poly_3d[i] 
            
            if isinstance(target, list):
                # 如果还是列表，尝试堆叠
                if len(target) > 0:
                     # 确保 target 是 tensor
                    if torch.is_tensor(target[0]):
                        target = torch.cat(target, dim=0)
                    else:
                        target = torch.tensor(target, device=pred.device)
                else:
                    continue # 没有轨道真值，跳过
            
            if not torch.is_tensor(target):
                 target = torch.tensor(target, device=pred.device)

            # 计算 Loss
            # 注意: 如果 target 点数很少，Chamfer 也能工作
            loss = self.loss_func(pred.unsqueeze(0), target.unsqueeze(0))
            total_loss += loss
            
        loss_dict['loss_poly'] = total_loss / len(points_pred)
        return loss_dict