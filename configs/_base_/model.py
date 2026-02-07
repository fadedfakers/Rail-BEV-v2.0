# configs/_base_/model.py

model = dict(
    type='RailFusionNet', # v2.0 新定义的检测器类
    
    # 1. Backbone: 沿用 PointPillars (Source 332)
    backbone=dict(
        type='PillarEncoder', # 对应 models/backbones/pillar_net.py
        voxel_size=[0.16, 0.16, 4],
        point_cloud_range=[-50, -50, -5, 50, 50, 3],
        in_channels=4, # x, y, z, intensity (dt 在 adapter 中拼接，这里可能需要改为 5)
        feat_channels=[64, 64],
        with_distance=False,
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=[-50, -50, -5, 50, 50, 3],
            voxel_size=[0.16, 0.16, 4],
            max_voxels=(16000, 40000))
    ),

    # 2. Neck: [SOTA核心] 时序特征融合 (Source 334)
    neck=dict(
        type='TemporalFusion', # 对应 models/necks/temporal_fusion.py
        in_channels=64,
        out_channels=128,
        frames_num=4,   # 必须与 dataset.py 一致
        fusion_method='conv_gru' # 或 'concat', 推荐 ConvGRU 处理时序
    ),

    # 3. Heads: 多任务头
    bbox_head=dict(
        type='CenterHead', # Source 278: 无 Anchor 检测头
        in_channels=128,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=1, class_names=['pedestrian']),
            dict(num_class=1, class_names=['obstacle']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.16, 0.16]
        ),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True
    ),

    # 4. Rail Head: [SOTA核心] 轨道几何头 (Source 340)
    rail_head=dict(
        type='PolyHead', # 对应 models/heads/poly_head.py
        in_channels=128,
        num_polys=2,     # 预测左右两根轨道
        num_control_points=5, # 每根轨道5个控制点
        # Loss 变更: 使用 Chamfer Distance (Source 345)
        loss_poly=dict(type='ChamferDistanceLoss', loss_weight=1.0) 
    ),

    # 训练与测试配置
    train_cfg=dict(
        pts=dict(
            point_cloud_range=[-50, -50, -5, 50, 50, 3],
            grid_size=[640, 640, 40],
            voxel_size=[0.16, 0.16, 4],
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )
    ),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.16, 0.16],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2
        )
    )
)