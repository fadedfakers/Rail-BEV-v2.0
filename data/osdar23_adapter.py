import numpy as np
import raillabel
import os
from torch.utils.data import Dataset
from .transforms import LoadPointsFromFile, LoadAnnotations3D

class RailDataset(Dataset):
    def __init__(self, data_root, ann_file, pipeline=None, classes=None, 
                 test_mode=False, frames_num=4):
        self.data_root = data_root
        self.frames_num = frames_num
        self.test_mode = test_mode
        self.classes = classes
        
        # [v2.0] 使用 raillabel 读取场景数据
        print(f"Loading OSDaR23 annotations from {ann_file}...")
        self.scene = raillabel.load(ann_file)
        self.frame_ids = sorted(list(self.scene.frames.keys()))
        self.pipeline = pipeline

    def get_pose_matrix(self, sensor_data):
        """从标注数据中提取4x4位姿矩阵 (Odom)"""
        # 假设 sensor_data 包含位置(pos)和四元数(quat)
        from scipy.spatial.transform import Rotation
        pos = np.array(sensor_data['translation'])
        quat = np.array(sensor_data['rotation']) # [x, y, z, w]
        rot_mat = Rotation.from_quat(quat).as_matrix()
        
        pose = np.eye(4)
        pose[:3, :3] = rot_mat
        pose[:3, 3] = pos
        return pose

    def load_temporal_points(self, current_frame_id, current_pose):
        """[SOTA 核心] 加载并对齐历史帧点云"""
        points_list = []
        
        # 1. 寻找历史帧 (当前帧 + 过去 N-1 帧)
        # 逻辑：在 frame_ids 列表中向前查找，确保属于同一个 scene
        curr_idx = self.frame_ids.index(current_frame_id)
        current_scene_id = self.scene.frames[current_frame_id].attributes['scene_id']
        
        history_frames = []
        for i in range(self.frames_num):
            idx = curr_idx - i
            if idx >= 0:
                frame_id = self.frame_ids[idx]
                frame = self.scene.frames[frame_id]
                # 必须保证是同一个 Scene 才能拼接
                if frame.attributes['scene_id'] == current_scene_id:
                    history_frames.append(frame)
        
        # 2. 读取点云并变换
        points_all = []
        for i, frame in enumerate(history_frames):
            # 读取点云文件 (假设路径在 sensor['lidar_path'])
            lidar_path = os.path.join(self.data_root, frame.sensors['lidar'].uri)
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
            
            # 计算相对位姿变换: P_curr = T_curr_inv @ T_prev @ P_prev
            # 注意: 这里的 T 通常指 Pose (Vehicle -> World)
            # 变换逻辑: Point_prev -> World -> Point_curr
            prev_pose = self.get_pose_matrix(frame.sensors['lidar'].pose)
            
            # T_relative = T_current^(-1) @ T_prev
            relative_pose = np.linalg.inv(current_pose) @ prev_pose
            
            # 执行坐标变换 (旋转 + 平移)
            # points[:, :3] (N, 3) @ R.T + t
            points_xyz = points[:, :3]
            points_xyz_h = np.hstack([points_xyz, np.ones((len(points), 1))]) # 齐次坐标
            transformed_xyz = (relative_pose @ points_xyz_h.T).T[:, :3]
            
            # [关键] 增加 dt 特征 (Source 325)
            # dt = 0 for current frame, 0.1, 0.2... for past
            time_lag = np.full((len(points), 1), i * 0.1) # 假设 10Hz
            
            points_with_dt = np.hstack([transformed_xyz, points[:, 3:4], time_lag])
            points_all.append(points_with_dt)
            
        return np.concatenate(points_all, axis=0)

    def prepare_train_data(self, index):
        frame_id = self.frame_ids[index]
        frame = self.scene.frames[frame_id]
        
        # 1. 获取当前帧位姿
        current_pose = self.get_pose_matrix(frame.sensors['lidar'].pose)
        
        # 2. [v2.0] 加载时序融合点云
        points = self.load_temporal_points(frame_id, current_pose)
        
        # 3. 加载标注 (BBox + Poly3D)
        gt_bboxes = []
        gt_labels = []
        gt_polys = [] # 3D 轨道控制点
        
        for ann in frame.annotations.values():
            if ann.type == 'bbox_3d':
                # [x, y, z, dx, dy, dz, rot]
                bbox = [ann.pos.x, ann.pos.y, ann.pos.z, 
                        ann.size.x, ann.size.y, ann.size.z, ann.rot.z]
                gt_bboxes.append(bbox)
                gt_labels.append(self.classes.index(ann.object_type))
            elif ann.type == 'poly_3d' and 'rail' in ann.object_type:
                 # 将多项式系数转换为控制点，方便旋转 (Source 329)
                 # 假设 ann.data 存储的是系数，我们在 transforms 里处理转换
                 gt_polys.append(ann.data) 

        input_dict = dict(
            points=points,
            gt_bboxes_3d=np.array(gt_bboxes, dtype=np.float32),
            gt_labels_3d=np.array(gt_labels, dtype=np.int64),
            gt_poly_3d=gt_polys # list of coeffs or points
        )
        
        # 4. 进入 Augmentation Pipeline
        example = self.pipeline(input_dict)
        return example