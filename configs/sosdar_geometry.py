# configs/sosdar_geometry.py

_base_ = [
    './_base_/dataset.py',
    './_base_/model.py',
    './_base_/schedule.py'
]

# [实验元数据]
work_dir = './work_dirs/sosdar_geometry_pretrain'
experiment_name = 'v2.0_phase1_geometry_pretrain'

# [模型配置]
model = dict(
    # 预训练阶段不冻结任何参数，全量学习几何特征
    backbone=dict(
        frozen_stages=-1, 
    ),
    # 几何头配置 (强调几何学习)
    rail_head=dict(
        loss_poly=dict(type='ChamferDistanceLoss', loss_weight=2.0) # [策略] 加大几何Loss权重 (Source 27)
    )
)

# [数据覆盖] 
# 覆盖 _base_/dataset.py 中的 ConcatDataset，仅使用 SOSDaR 进行专项训练
data_root = '/root/autodl-tmp/FOD/SOSDaR24'
dataset_type = 'RailDataset'
class_names = ['pedestrian', 'car', 'obstacle', 'signal', 'buffer_stop']

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_poly_3d=True),
    # SOSDaR 数据通常非常完美，可以不做复杂的时序对齐，或者为了保持一致性保留
    dict(type='LoadMultiViewTemporalPoints', frames_num=4, use_odom=True), 
    
    # [增强] 几何预训练阶段，可以使用更激进的旋转
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.785, 0.785], # +/- 45度
         scale_ratio_range=[0.95, 1.05],
         translation_std=[0, 0, 0],
         update_poly3d=True), # [关键] 必须确保transforms.py已修复

    dict(type='PointsRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(type='ObjectRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_poly_3d'])
]

data = dict(
    samples_per_gpu=4, # VGPU 48G 显存充足
    workers_per_gpu=4,
    train=dict(
        _delete_=True, # 删除 base 中的 ConcatDataset 定义
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/sosdar24_infos_train.pkl', # 需先运行 create_data.py
        pipeline=train_pipeline,
        classes=class_names,
        modality=dict(use_lidar=True, use_camera=False), # 预训练可仅用雷达，加快速度
        test_mode=False,
        box_type_3d='LiDAR')
)

# [调度配置]
runner = dict(max_epochs=24) # 充分预训练