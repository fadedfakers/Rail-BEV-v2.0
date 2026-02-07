# configs/_base_/schedule.py

# 优化器配置
optimizer = dict(
    type='AdamW', 
    lr=0.001, 
    betas=(0.95, 0.99), 
    weight_decay=0.01
)

optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2) # 防止梯度爆炸
)

# 学习率调度 (Cosine Annealing)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

# 运行参数
runner = dict(type='EpochBasedRunner', max_epochs=24) # 建议24轮，Source 350+

# Checkpoint 保存策略
checkpoint_config = dict(interval=1, max_keep_ckpts=5) # 每轮保存，最多留5个

# 日志配置 (Source 358)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook') # 必须开启，用于监控 loss_poly 和 loss_cls
    ]
)

# 运行设置
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/rail_bev_v2'
load_from = None
resume_from = None
workflow = [('train', 1)]

# [警告] 严禁在此处设置 find_unused_parameters=True，
# 必须在 train.py 中显式控制，避免隐藏 "僵尸网络" 问题 (Source 190)