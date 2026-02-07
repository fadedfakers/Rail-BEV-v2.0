# configs/_base_/dataset.py

# 数据集类型定义（对应 data/osdar23_adapter.py 中定义的类）
dataset_type = 'RailDataset'

# AutoDL 路径配置
data_root_osdar = '/root/autodl-tmp/FOD/data'      # OSDaR23 (Real)
data_root_sosdar = '/root/autodl-tmp/FOD/SOSDaR24' # SOSDaR (Sim)

# 类别定义 (OSDaR23 标准类别)
class_names = ['pedestrian', 'car', 'obstacle', 'signal', 'buffer_stop']

# 输入模态配置
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

# [SOTA 核心] 关键参数
frames_num = 4          # 加载当前帧 + 过去3帧 (Source 350)
point_cloud_range = [-50, -50, -5, 50, 50, 3]
voxel_size = [0.16, 0.16, 4] # 保持 PointPillars 默认 (Source 352)

# 训练数据流水线
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_poly_3d=True), # 开启 Poly3D 加载
    
    # [v2.0 核心] 时序点云加载与对齐
    dict(type='LoadMultiViewTemporalPoints', 
         frames_num=frames_num, 
         use_odom=True), # 使用 Odom 矩阵进行多帧对齐 (Source 324)

    # [v2.0 修复] 几何同步增强 (需配合修复后的 transforms.py)
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.785, 0.785],
         scale_ratio_range=[0.95, 1.05],
         translation_std=[0, 0, 0],
         update_poly3d=True), # 确保轨道标签同步旋转 (Source 194, 328)

    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_poly_3d'])
]

# 测试数据流水线
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadMultiViewTemporalPoints', frames_num=frames_num, use_odom=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='GlobalRotScaleTrans', rot_range=[0, 0], scale_ratio_range=[1.0, 1.0], translation_std=[0, 0, 0]),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    samples_per_gpu=4, # 48GB VGPU 允许较大 Batch Size
    workers_per_gpu=4,
    train=dict(
        type='ConcatDataset', # 混合 OSDaR 和 SOSDaR
        datasets=[
            # OSDaR23 (主战场: 检测)
            dict(
                type=dataset_type,
                data_root=data_root_osdar,
                ann_file=data_root_osdar + '/osdar23_infos_train.pkl',
                pipeline=train_pipeline,
                classes=class_names,
                modality=input_modality,
                test_mode=False,
                box_type_3d='LiDAR'),
            # SOSDaR (辅助: 几何增强)
            dict(
                type=dataset_type,
                data_root=data_root_sosdar,
                ann_file=data_root_sosdar + '/sosdar24_infos_train.pkl',
                pipeline=train_pipeline,
                classes=class_names,
                modality=input_modality,
                test_mode=False,
                box_type_3d='LiDAR')
        ]
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root_osdar,
        ann_file=data_root_osdar + '/osdar23_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root_osdar,
        ann_file=data_root_osdar + '/osdar23_infos_test.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR')
)