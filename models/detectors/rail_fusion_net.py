import torch
import torch.nn as nn
from mmdet3d.models.detectors import BaseDetector
from mmdet3d.models.builder import DETECTORS, build_backbone, build_head, build_neck

@DETECTORS.register_module()
class RailFusionNet(BaseDetector):
    def __init__(self, 
                 backbone, 
                 neck=None, 
                 bbox_head=None, 
                 rail_head=None, 
                 train_cfg=None, 
                 test_cfg=None, 
                 pretrained=None,
                 init_cfg=None):
        super(RailFusionNet, self).__init__(init_cfg)
        
        self.backbone = build_backbone(backbone)
        
        if neck is not None:
            self.neck = build_neck(neck)
            
        if bbox_head is not None:
            self.bbox_head = build_head(bbox_head)
            
        if rail_head is not None:
            self.rail_head = build_head(rail_head)
            
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, points):
        """
        特征提取阶段 (支持时序)
        Input: points List[Tensor] (B)
        """
        # 1. Backbone Forward (PointPillars)
        # PillarEncoder 接受 List[Tensor], 返回 [B, C, H, W]
        x = self.backbone(points)
        
        # 2. Neck Forward (Temporal Fusion)
        if self.with_neck:
            # [关键逻辑] 检查 Neck 是否是 TemporalFusion
            if hasattr(self.neck, 'frames_num') and self.neck.frames_num > 1:
                # 如果输入 Batch 是 B*T (Late Fusion 模式)，需要 Reshape
                # 但根据 Adapter 代码，我们做的是 Early Fusion (Concatenation)
                # 所以 Backbone 输出已经是融合后的单帧特征。
                # 这种情况下，TemporalFusion 退化为普通的 Conv 层处理
                pass 
            
            x = self.neck(x)
            
        return [x] # MMDetection 约定返回 List/Tuple

    def forward_train(self, points, gt_bboxes_3d, gt_labels_3d, gt_poly_3d=None, **kwargs):
        """
        训练前向传播
        """
        # 1. 提取 BEV 特征
        img_feats = self.extract_feat(points) # List of [B, C, H, W]
        x = img_feats[0]
        
        losses = dict()
        
        # 2. 检测头 (BBox Head) Loss
        if self.bbox_head:
            # 这里的 loss 输入需要适配 CenterHead 的接口
            # 简化版: 假设 CenterHead 内部能处理
            loss_bbox = self.bbox_head.loss([self.bbox_head(x)], gt_bboxes_3d, gt_labels_3d)
            losses.update(loss_bbox)
            
        # 3. 轨道头 (Poly Head) Loss
        if self.rail_head and gt_poly_3d is not None:
            poly_preds = self.rail_head(x)
            loss_poly = self.rail_head.loss(poly_preds, gt_poly_3d)
            losses.update(loss_poly)
            
        return losses

    def simple_test(self, points, **kwargs):
        """
        推理前向传播
        """
        img_feats = self.extract_feat(points)
        x = img_feats[0]
        
        results = []
        
        # 1. 检测结果
        if self.bbox_head:
            preds_dicts = self.bbox_head(x)
            # 需要实现 get_bboxes 解码 heatmap 为 bbox
            # bbox_results = self.bbox_head.get_bboxes(preds_dicts, ...)
            # results.append(bbox_results)
            pass

        # 2. 轨道结果
        if self.rail_head:
            poly_preds = self.rail_head(x)
            # poly_preds: [B, 2, 5, 3] -> 转换为 numpy 或 list
            # results.append(poly_preds)
            pass
            
        return results

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None