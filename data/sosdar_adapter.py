import json
import numpy as np
import os
from torch.utils.data import Dataset

class SOSDaRDataset(Dataset):
    def __init__(self, data_root, ann_file, pipeline=None, classes=None):
        self.data_root = data_root
        self.classes = classes
        self.pipeline = pipeline
        
        print(f"Parsing SOSDaR OpenLABEL from {ann_file}...")
        with open(ann_file, 'r') as f:
            self.openlabel = json.load(f)['openlabel']
            
        self.frame_ids = sorted(list(self.openlabel['frames'].keys()), key=lambda x: int(x))

    def __getitem__(self, index):
        frame_id = self.frame_ids[index]
        frame_data = self.openlabel['frames'][frame_id]
        
        # 1. 获取对应的点云文件路径 (OpenLABEL streams)
        # 结构通常是 frames -> streams -> lidar -> uri
        # 如果 frame 里没有，可能在 objects 或者 external files 里，根据 Vicomtech 格式适配
        lidar_uri = frame_data['streams']['lidar']['uri'] 
        lidar_path = os.path.join(self.data_root, lidar_uri)
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        
        # SOSDaR 是仿真数据，通常不需要复杂的时序对齐，或者已经对齐好了
        # 为保持格式一致，增加 dt=0
        points = np.hstack([points, np.zeros((len(points), 1))])

        # 2. 解析对象 (Objects)
        gt_bboxes = []
        gt_labels = []
        
        # OpenLABEL 中 frame 下的 object_ref 指向 root objects
        for obj_id, obj_data in frame_data['objects'].items():
            # 获取对象静态属性 (Type)
            root_obj = self.openlabel['objects'][obj_id]
            obj_type = root_obj['type']
            
            if obj_type not in self.classes:
                continue
                
            # 获取动态属性 (BBox)
            # val: [center_x, center_y, center_z, ext_x, ext_y, ext_z, qx, qy, qz, qw]
            bbox_data = obj_data['object_data']['cuboid']['val']
            
            # 转换 Quaternion 到 Yaw (RotZ)
            from scipy.spatial.transform import Rotation
            r = Rotation.from_quat([bbox_data[6], bbox_data[7], bbox_data[8], bbox_data[9]])
            yaw = r.as_euler('xyz')[2]
            
            bbox = [*bbox_data[:6], yaw]
            gt_bboxes.append(bbox)
            gt_labels.append(self.classes.index(obj_type))

        # 3. 解析轨道 (Poly3D)
        # SOSDaR 可能直接提供 PolyLine 点集
        gt_polys = []
        # (解析逻辑省略，类似 BBox)

        input_dict = dict(
            points=points,
            gt_bboxes_3d=np.array(gt_bboxes, dtype=np.float32),
            gt_labels_3d=np.array(gt_labels, dtype=np.int64),
            gt_poly_3d=gt_polys
        )
        
        return self.pipeline(input_dict)