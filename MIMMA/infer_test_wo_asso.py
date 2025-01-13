import argparse
import os
import pdb
import time
import yaml
import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
# from progress.bar import Bar
from torch.utils.data import DataLoader
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tools_msta_wo_asso import *


import matplotlib.pyplot as plt

def plot_obs_t_s(obs_array, plot_root, t):
    os.makedirs(plot_root, exist_ok=True)
    # 从数据中提取 lat (纬度), lon (经度), 和 mmsi
    # pdb.set_trace()
    lat = obs_array[:, 3]
    lon = obs_array[:, 4]
    mmsi = obs_array[:, 1]
    # 获取唯一的 mmsi 值
    unique_mmsi = np.unique(mmsi)
    # 生成不同颜色的 colormap
    colors = cm.rainbow(np.linspace(0, 1, len(unique_mmsi)))
    # 创建图形
    plt.figure(figsize=(8, 6))
    # 根据 mmsi 分组并绘制不同颜色的点
    for i, ummsi in enumerate(unique_mmsi):
        mask = mmsi == ummsi
        plt.scatter(lon[mask], lat[mask], color=colors[i], label=f'MMSI {int(ummsi)}')

    # 添加图例、标题和标签
    plt.title('Scatter Plot of Points by MMSI')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.savefig(os.path.join(plot_root, str(t) + '.png'), format='png', dpi=300)
    plt.close()

def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_root", type=str, default='tracking/experiments',  help="checkpoint root")
    # parser.add_argument("--tracking_ckpt", type=str, default='tracking/experiments/MSTAv2/checkpoints/best_val_checkpoint.pth.tar',  help="checkpoint path")
    parser.add_argument("--data_root", type=str, default='data',  help="data_root path")
    parser.add_argument("--dataset_name", type=str, default='MSTAv2',  help="data_root path")
    parser.add_argument("--ckpt_dir", type=str, default='MSTAv2',  help="data_root path")
    parser.add_argument("--exp_name", type=str, default='MSTAv2',  help="data_root path")
    # parser.add_argument("--file_name", type=str, default='area_0.txt',  help="data_root path")
    # parser.add_argument("--obs", type=str, default='MSTAv2_obs',  help="obs path")
    # parser.add_argument("--gt", type=str, default='MSTAv2_gt',  help="gt path")
    # parser.add_argument("--area_lon_lat", type=str, default='area_lon_lat.yaml',  help="area_lon_lat path")
    parser.add_argument("--pred_root", type=str, default='outputs',   help="result path")
    # parser.add_argument("--pred_result_dir", type=str, default='MSTAv2',   help="result path")
    parser.add_argument("--time_axio", type=int, default=30,  help="result path")
    parser.add_argument("--in_F", type=int, default=30,  help="the input window length")
    parser.add_argument("--out_F", type=int, default=5,  help="the output window length")
    # parser.add_argument("--cfg", type=str, default="", help="Config name. Otherwise will use default config")
    parser.add_argument("--split", type=str, default="test", help="Split to use. one of [train, test, valid]")
    parser.add_argument("--metric", type=str, default="ade_fde", help="Evaluation metric. One of (vim, mpjpe)")
    parser.add_argument("--modality", type=str, default="traj+all", help="available modality combination from['traj','traj+2dbox']")

    parser.add_argument("--plot_dir", type=str, default='plot_dir',  help="plot_dir path")

    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    return args

if __name__ == "__main__":
    args = init_config()
    data_root = args.data_root
    # file_name = args.file_name
    row_obs_dataset_name = args.dataset_name + '_obs'
    row_gt_dataset_name = args.dataset_name + '_gt'
    obs_path_split = os.path.join(data_root, row_obs_dataset_name, args.split)
    gt_path_split = os.path.join(data_root, row_gt_dataset_name, args.split)
    print(f'load obs data from {obs_path_split}')
    print(f'load gt data from {gt_path_split}')
    for file_name in os.listdir(obs_path_split):
        # file_name = 'area_237.txt'
        args.file_name = file_name
        print(file_name)
        obs_file_path, gt_file_path = os.path.join(obs_path_split, file_name), os.path.join(gt_path_split, file_name)
        if os.path.getsize(obs_file_path) == 0:
            continue
            # break
        obs_array, gt_array, max_F, min_F = read_row_files(args, obs_file_path, gt_file_path)
        # 跟踪系统对象
        multiSensorShipTrackingModel = MultiSensorShipTrackingModel(args=args)
        satellite_id = None
        end_F = max_F
        # end_F = min_F + 80
        progress_bar = tqdm(range(min_F, end_F + 1), desc="Processing")
        for t in progress_bar:
            # Process time
            time_step = t - min_F

            # Filter and concatenate observation data
            obs_array_t = obs_array[obs_array[:, 0] == t]
            obs_mmsi_t = obs_array_t[:, 1]
            obs_array_t = np.concatenate([obs_array_t[:, :1], obs_array_t[:, 5:]], axis=1)

            # if obs_array_t.shape[0] == 0:
            #     continue

            # Multi-source heterogeneous target continuous tracking and trajectory association algorithm function
            ship_traj_long_dict_active_all, ship_traj_short_dict_active_all, ship_traj_long_dict_finish_all, ship_traj_short_dict_finish_all, ship_traj_finish_dict, ship_traj_dict_pred = multiSensorShipTrackingModel.multi_object_tracking(obs_array_t, obs_mmsi_t)

            # Update progress bar description with MOT data
            ActiveLong = len(ship_traj_long_dict_active_all)
            ActiveShort = len(ship_traj_short_dict_active_all)
            FinishedLong = len(ship_traj_long_dict_finish_all)
            FinishedShort = len(ship_traj_short_dict_finish_all)
            Active = len(list(ship_traj_dict_pred.keys()))
            Finished = len(list(ship_traj_finish_dict.keys()))
            progress_bar.set_postfix(
            Time=time_step,
            Active_long=ActiveLong,
            Active_short=ActiveShort,
            Finished_long=FinishedLong,
            Finished_short=FinishedShort,
            Active=Active,
            Finished=Finished
        )
            # if t >= min_F + 10:
            #     pdb.set_trace()
        multiSensorShipTrackingModel.save_tracking_traj_dict(args)
