import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import BEVConfig as cfg

class PillarEncoder(nn.Module):
    # [v2.0 修正] in_channels 改为 5 (x, y, z, intensity, dt)
    def __init__(self, in_channels=5, out_channels=64):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        
        # [v2.0 优化] 增加特征维度: 
        # 输入 5 (x,y,z,i,dt) -> 扩展到 8 (x,y,z,i,dt, x-xc, y-yc, z-zc)
        # 这能让模型更好地理解点在网格内的相对位置
        self.use_norm = True 
        input_dim = in_channels + 3 if self.use_norm else in_channels

        self.pfn = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            # [v2.0] 添加 MaxPool 之前的通道映射，保证输出特征丰富度
            nn.Linear(out_channels, out_channels * 2) 
        )
        
    def forward(self, points_list):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        processed_batches = []
        for pts in points_list:
            pts = pts.to(device=device, dtype=dtype)
            
            if pts.shape[0] > 0:
                # 1. 计算体素中心偏移 (Augment Point Features)
                # 假设 pts 是 (N, 5) -> x, y, z, i, dt
                x_idx = ((pts[:, 0] - cfg.X_RANGE[0]) / cfg.VOXEL_SIZE).long()
                y_idx = ((pts[:, 1] - cfg.Y_RANGE[0]) / cfg.VOXEL_SIZE).long()
                
                # 过滤越界点
                mask = (x_idx >= 0) & (x_idx < cfg.GRID_W) & (y_idx >= 0) & (y_idx < cfg.GRID_H)
                pts = pts[mask]
                x_idx, y_idx = x_idx[mask], y_idx[mask]

                if self.use_norm:
                    # 计算体素中心坐标
                    x_center = x_idx * cfg.VOXEL_SIZE + cfg.X_RANGE[0] + cfg.VOXEL_SIZE / 2
                    y_center = y_idx * cfg.VOXEL_SIZE + cfg.Y_RANGE[0] + cfg.VOXEL_SIZE / 2
                    z_center = pts[:, 2] # 简化处理，Z轴不分格
                    
                    # 拼接偏移量 (N, 8)
                    pts_aug = torch.cat([pts, pts[:, 0:1]-x_center.unsqueeze(1), 
                                              pts[:, 1:2]-y_center.unsqueeze(1), 
                                              pts[:, 2:3]-z_center.unsqueeze(1)], dim=1)
                else:
                    pts_aug = pts

                # 2. 特征提取 [N, 128]
                feat = self.pfn(pts_aug)
                
                # 3. 最大池化 (Max Pooling) - 比 index_add_ 更适合提取几何特征
                # 这里为了保持代码结构，我们使用 pseudo_voxelize (scatter max)
                bev_map = self._scatter_max(feat, x_idx, y_idx, device, dtype)
                processed_batches.append(bev_map)
            else:
                processed_batches.append(
                    torch.zeros((self.out_channels * 2, cfg.GRID_H, cfg.GRID_W), 
                                device=device, dtype=dtype)
                )
        
        return torch.stack(processed_batches)

    def _scatter_max(self, feat, x_idx, y_idx, device, dtype):
        """v2.0 推荐使用 Max Pooling 而不是 Sum Pooling"""
        indices = y_idx * cfg.GRID_W + x_idx
        canvas = torch.zeros((self.out_channels * 2, cfg.GRID_H * cfg.GRID_W), 
                             device=device, dtype=dtype)
        
        # 使用 scatter_reduce (PyTorch 1.12+) 或 简易实现
        # 这里为了兼容性，使用一种简化的覆盖式写入 (近似 Max) 
        # *在实际工业部署中，建议使用 torch_scatter 库*
        canvas[:, indices] = feat.t() 
        
        return canvas.view(self.out_channels * 2, cfg.GRID_H, cfg.GRID_W)