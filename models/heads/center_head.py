import torch
import torch.nn as nn
from mmdet3d.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule

@HEADS.register_module()
class CenterHead(BaseModule):
    def __init__(self, 
                 in_channels=128, 
                 tasks=None, 
                 common_heads=dict(), 
                 share_conv_channel=64, 
                 bbox_coder=None, 
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'), 
                 loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25), 
                 norm_bbox=True, 
                 init_cfg=None):
        super(CenterHead, self).__init__(init_cfg)
        
        self.in_channels = in_channels
        self.tasks = tasks
        self.common_heads = common_heads
        self.norm_bbox = norm_bbox
        self.bbox_coder = bbox_coder
        
        # 建立共享卷积层
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, share_conv_channel, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True)
        )
        
        # 建立各个任务头 (Task Heads)
        self.task_heads = nn.ModuleList()
        for task in tasks:
            heads = nn.ModuleDict()
            # 1. 类别热图头 (Heatmap)
            heads['hm'] = nn.Sequential(
                nn.Conv2d(share_conv_channel, share_conv_channel, 3, padding=1),
                nn.BatchNorm2d(share_conv_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(share_conv_channel, task['num_class'], 1)
            )
            # 2. 回归头 (Regression: center, dim, rot, vel)
            for head_name, head_channels in common_heads.items():
                heads[head_name] = nn.Sequential(
                    nn.Conv2d(share_conv_channel, share_conv_channel, 3, padding=1),
                    nn.BatchNorm2d(share_conv_channel),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(share_conv_channel, head_channels[-1], 1)
                )
            self.task_heads.append(heads)

        # 损失函数
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

    def forward(self, x):
        """
        Input x: [B, C, H, W] (BEV Features)
        Output: List of dicts (one per task)
        """
        ret_dicts = []
        x = self.shared_conv(x)
        
        for task_head in self.task_heads:
            task_dict = {}
            for head_name, head_layer in task_head.items():
                task_dict[head_name] = head_layer(x)
                
                # 对 Heatmap 进行 Sigmoid 归一化 (CenterPoint 标准做法)
                if head_name == 'hm':
                    task_dict[head_name] = torch.sigmoid(task_dict[head_name])
                    # 裁剪边界值以防数值不稳定
                    task_dict[head_name] = torch.clamp(task_dict[head_name], min=1e-4, max=1-1e-4)
            ret_dicts.append(task_dict)
            
        return ret_dicts

    def loss(self, preds_dicts, gt_bboxes_3d, gt_labels_3d):
        """
        计算 CenterPoint 损失
        """
        loss_dict = dict()
        
        # 生成 GT Heatmap (通常需要调用特定的 target generator，这里简化示意)
        # 实际代码中需要 GaussianTargetGenerator 生成 heatmap
        # 假设 targets 已经在外部生成或通过 bbox_coder 获取
        
        # ... (此处省略复杂的 Target 生成逻辑，假设使用 self.bbox_coder.encode) ...
        # 注意：在完整实现中，你需要 Gaussian Focal Loss 的具体实现
        
        # 为了演示代码结构完整性，我们展示 Loss 计算流程
        for task_id, preds in enumerate(preds_dicts):
            # 模拟 Target
            # heatmap_target, bbox_target... = self.get_targets(...)
            pass 
            
        # 注意: 由于 target generation 代码量较大，建议复用 mmdet3d 的 utils
        # 这里返回一个占位 Loss 以保证运行
        loss_dict['loss_heatmap'] = torch.tensor(0.0, device=preds_dicts[0]['hm'].device, requires_grad=True)
        loss_dict['loss_bbox'] = torch.tensor(0.0, device=preds_dicts[0]['hm'].device, requires_grad=True)
        
        return loss_dict