import pathlib
import logging
import os
import pdb
import random
import pandas as pd

from torch import nn
import numpy as np
import torch
import torch.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
# from torchvision.datasets.cityscapes import Cityscapes
import cv2
from torchvision.transforms import ToPILImage

from torch.utils.data import DataLoader
from torchvision.transforms import (
    RandomResizedCrop,
    RandomHorizontalFlip,
    Normalize,
    RandomErasing,
    Resize,
    ToTensor,
    RandomAffine,
    Compose,
    ColorJitter,
)



class DataGenerator(Dataset):
    def __init__(self, split, dataroot, seq_len, image_transforms=None):
        # dataroot: 所有视频的目录文件夹 i.e. data/VISO/mot/car
        self.dataroot = dataroot
        if split == 'train':
            self.video_name_list = sorted(os.listdir(dataroot))[:20]  # [001, 002, ...]
        else:
            self.video_name_list = sorted(os.listdir(dataroot))[20:27]  # [001, 002, ...]
        self.seq_len = seq_len
        self.split = split
        self.image_transforms = image_transforms

    def __len__(self) -> int:
        # return min(len(self.dataset), 10)
        return len(self.video_name_list)

    def init_scenes(self):
        scenes = {}
        video_keys = list(self.videos.keys())
        for video_name in video_keys:
            print(f'loading video {video_name}')
            video = self.videos[video_name]
            # images = video['images']
            labels = video['labels']
            obj_list = torch.unique(labels[:, 1])
            scenes_video = {}
            for i in tqdm(range(len(obj_list))):
                obj = obj_list[obj]
                labels_obj = labels[labels[:, 1] == obj]
                # 可能会有中间帧缺失，用0来填充
                obj_max_frame = torch.max(labels_obj[:, 0])
                obj_min_frame = torch.min(labels_obj[:, 0])
                # pdb.set_trace()
                for f in range(int(obj_min_frame), int(obj_max_frame) - self.seq_len - 1, int(1 / self.sample_rate)):
                    scene = {}
                    traj = []
                    if f + self.seq_len not in labels_obj[:, 0]:
                        continue
                    label_next = labels_obj[labels_obj[:, 0] == f + self.seq_len][0]
                    for f_ in range(f, f + self.seq_len):
                        if f_ not in labels_obj[:, 0]:
                            traj.append(torch.tensor([[0, 0, 0, 0]], dtype=torch.float32))
                        else:
                            traj.append(labels_obj[labels_obj[:, 0] == f_][:, 2:6])
                    scene['p'] = obj
                    scene['traj'] = torch.cat(traj)
                    scene['label_next'] = label_next
                    if f + self.seq_len not in scenes_video.keys():
                        scenes_video[f + self.seq_len] = []
                    scenes_video[f + self.seq_len].append(scene)
            scenes['video_name'] = video_name
            scenes['scenes_video'] = scenes_video
        pdb.set_trace()
        self.scenes = scenes

    def __getitem__(self, idx):
        # 每次产生batch个短视频序列，每个短视频序列长度固定为 self.seq_len
        # 随机读取短视频序列图像序列及文本序列
        # print(self.video_name_list[idx], idx)
        image_list = os.listdir(os.path.join(self.dataroot, self.video_name_list[idx], 'img'))
        text_array = pd.read_csv(os.path.join(self.dataroot, self.video_name_list[idx], 'gt/gt.txt')).values
        start_idy = random.randint(0, len(image_list) - self.seq_len - 1)
        end_idy = start_idy + self.seq_len
        # 获取当前帧图像
        image_path = os.path.join(self.dataroot, self.video_name_list[idx], 'img', image_list[end_idy])
        image = cv2.imread(image_path)
        tensor_image = torch.tensor(image, dtype=torch.float32)
        tensor_image = tensor_image.permute(2, 0, 1)
        # pdb.set_trace()
        for i in range(len(self.image_transforms)):
            tensor_image = self.image_transforms[i](tensor_image)
        # 获取历史帧轨迹文本
        # pdb.set_trace()
        text_array = text_array.astype(np.float32)
        array_label = text_array[text_array[:, 0] < end_idy]
        array_label = array_label[array_label[:, 0] >= start_idy]
        tensor_label = torch.tensor(array_label, dtype=torch.float32)
        # 转为 x_min,y_min,x_max,y_max
        tensor_label[:, 4] += tensor_label[:, 2]
        tensor_label[:, 5] += tensor_label[:, 3]
        # pdb.set_trace()
        tensor_label[:, -1] = torch.ones((tensor_label.shape[0]))
        # tensor_laebl_idx = tensor_label[tensor_label[:, 1] == 4]
        tensor_traj = tensor_label[tensor_label[:, 0] < end_idy - 1]
        tensor_label = tensor_label[tensor_label[:, 0] == end_idy - 1]
        # pdb.set_trace()
        return tensor_image, tensor_traj, tensor_label
