# configs/osdar23_temporal.py

_base_ = [
    './_base_/dataset.py',
    './_base_/model.py',
    './_base_/schedule.py'
]

# [实验元数据]
work_dir = './work_dirs/osdar23_temporal_v2'
experiment_name = 'v2.0_phase2_temporal_finetune'

# [SOTA 策略] 加载 SOSDaR 几何预训练权重 (Source 4, 27)
# 请在运行完 sosdar_geometry 后，将最佳权重路径填入此处
load_from = './work_dirs/sosdar_geometry_pretrain/latest.pth' 

# [模型微调配置]
model = dict(
    type='RailFusionNet',
    
    # [微调策略] Source 174: 仅在微调阶段控制冻结
    # 建议冻结底层 Backbone 的前几层，保留几何特征，让高层适应真实域噪声
    backbone=dict(
        frozen_stages=1, 
    ),
    
    # [时序模块] 确保开启
    neck=dict(
        type='TemporalFusion',
        frames_num=4,
        fusion_method='conv_gru' # GRU 对时序建模效果最好
    ),
    
    # [检测头] 真实域误报较多，调整 Loss 权重 (Source 23)
    bbox_head=dict(
        loss_cls=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25)
    )
)

# [数据覆盖] 
# 覆盖 _base_，专注于 OSDaR23 真实域 (Source 18)
data_root = '/root/autodl-tmp/FOD/data'
dataset_type = 'RailDataset'
class_names = ['pedestrian', 'car', 'obstacle', 'signal', 'buffer_stop']

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_poly_3d=True),
    
    # [v2.0 核心] 时序对齐 (Source 22: 利用 odom 解决稀疏性)
    dict(type='LoadMultiViewTemporalPoints', 
         frames_num=4, 
         use_odom=True,
         file_client_args=dict(backend='disk')),
    
    # 真实域增强需要更谨慎
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.3925, 0.3925], # +/- 22.5度，比预训练保守
         scale_ratio_range=[0.95, 1.05],
         translation_std=[0, 0, 0],
         update_poly3d=True), 

    dict(type='PointsRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(type='ObjectRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_poly_3d'])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/osdar23_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=dict(use_lidar=True, use_camera=True), # 真实域开启多模态融合
        test_mode=False,
        box_type_3d='LiDAR'),
    val=dict(
        # 验证集配置保持 _base_ 中的 OSDaR 设置即可，无需覆盖
    ),
    test=dict(
        # 测试集配置保持 _base_ 中的 OSDaR 设置即可，无需覆盖
    )
)

# [训练策略优化]
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01) # 微调使用较小的 LR
runner = dict(max_epochs=12) # 微调轮数可以减少