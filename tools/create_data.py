import pickle
import os
import glob
import numpy as np
import argparse
from pathlib import Path
import json

def create_osdar_infos(root_path, out_dir):
    """
    处理 OSDaR23 数据集结构
    root_path/
      - OSDaR23_Image_Semantic/ (images)
      - OSDaR23_LiDAR_Point_Clouds/ (lidar)
      - annotation/ (json)
    """
    print(f"Processing OSDaR23 at {root_path}")
    
    # 定义划分 (假设已有 split 文件，或简单的按比例划分)
    # 这里演示简单的全量读取并生成 info
    
    infos_train = []
    infos_val = []
    
    # 查找所有 annotation json
    ann_files = glob.glob(os.path.join(root_path, 'annotation', '*.json'))
    
    for ann_file in ann_files:
        with open(ann_file, 'r') as f:
            data = json.load(f)
            
        scene_id = os.path.basename(ann_file).replace('.json', '')
        frames = data['frames']
        
        # 按 8:2 划分 Train/Val
        frame_ids = sorted(list(frames.keys()))
        split_idx = int(len(frame_ids) * 0.8)
        
        for i, fid in enumerate(frame_ids):
            frame = frames[fid]
            info = dict()
            info['sample_idx'] = fid
            info['scene_id'] = scene_id
            
            # 路径处理 (需要根据实际解压路径调整)
            # OSDaR URI: "lidar/..." -> absolute path
            info['lidar_path'] = os.path.join(root_path, frame['sensors']['lidar']['uri'])
            info['image_path'] = os.path.join(root_path, frame['sensors']['camera']['uri'])
            
            # 读取标定 (Odom & Intrinsics)
            # 建议在此处预先存入 info，方便 dataset 读取
            info['pose'] = frame['sensors']['lidar']['pose'] # translation, rotation
            
            # 分配到 train 或 val
            if i < split_idx:
                infos_train.append(info)
            else:
                infos_val.append(info)

    # 保存
    with open(os.path.join(out_dir, 'osdar23_infos_train.pkl'), 'wb') as f:
        pickle.dump(infos_train, f)
    with open(os.path.join(out_dir, 'osdar23_infos_val.pkl'), 'wb') as f:
        pickle.dump(infos_val, f)
    print(f"Generated OSDaR23 infos: {len(infos_train)} train, {len(infos_val)} val")

def create_sosdar_infos(root_path, out_dir):
    """
    处理 SOSDaR (OpenLABEL)
    """
    print(f"Processing SOSDaR at {root_path}")
    infos_train = []
    
    # 假设 SOSDaR 是一个大的 openlabel json
    ann_files = glob.glob(os.path.join(root_path, '*.json'))
    
    for ann_file in ann_files:
        with open(ann_file, 'r') as f:
            data = json.load(f)
            
        frames = data['openlabel']['frames']
        for fid, frame in frames.items():
            info = dict()
            info['sample_idx'] = fid
            # SOSDaR URI 处理
            if 'lidar' in frame['streams']:
                info['lidar_path'] = os.path.join(root_path, frame['streams']['lidar']['uri'])
            
            # SOSDaR 通常用于预训练，全量放入 train
            infos_train.append(info)
            
    with open(os.path.join(out_dir, 'sosdar24_infos_train.pkl'), 'wb') as f:
        pickle.dump(infos_train, f)
    print(f"Generated SOSDaR infos: {len(infos_train)} train")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--osdar-root', default='/root/autodl-tmp/FOD/data')
    parser.add_argument('--sosdar-root', default='/root/autodl-tmp/FOD/SOSDaR24')
    args = parser.parse_args()
    
    create_osdar_infos(args.osdar_root, args.osdar_root)
    create_sosdar_infos(args.sosdar_root, args.sosdar_root)

if __name__ == '__main__':
    main()