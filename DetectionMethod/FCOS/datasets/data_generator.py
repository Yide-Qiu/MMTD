import pathlib
import logging
import os
import pdb
import random
import pandas as pd
from tqdm import tqdm

from torch import nn
import numpy as np
import torch
import torch.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
# from torchvision.datasets.cityscapes import Cityscapes
import cv2
from torchvision.transforms import ToPILImage

import time

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
    def __init__(self, split, dataroot, seq_len):
        # dataroot: 所有视频的目录文件夹 i.e. data/VISO/mot/car
        self.dataroot = dataroot
        if split == 'train':
            self.video_name_list = sorted(os.listdir(dataroot))[:1]  # [001, 002, ...]
        else:
            self.video_name_list = sorted(os.listdir(dataroot))[20:21]  # [001, 002, ...]
        self.seq_len = seq_len
        self.split = split
        self.image_size = 1024
        self.image_resize = Resize(self.image_size)
        self.sample_rate = 0.1
        self.patch_size = 512
        self.stride = 480
        self.init_videos()
        self.init_scenes()

    def init_scenes(self):
        scenes = []
        video_keys = list(self.videos.keys())
        for video_name in video_keys:
            print(f'init scenes video {video_name}')
            video = self.videos[video_name]
            images = video['images']
            labels = video['labels']
            frame_list = torch.unique(labels[:, 0])
            for i in tqdm(range(self.seq_len, len(frame_list))):
                frame = frame_list[i]
                scene = {'image': images[i]}
                trajs = []
                labels_scene = labels[labels[:, 0] <= frame]
                labels_scene = labels_scene[labels_scene[:, 0] >= frame - self.seq_len]
                obj_list = torch.unique(labels_scene[:, 1])
                for obj in obj_list:
                    labels_obj = labels_scene[labels_scene[:, 1] == obj]
                    traj_obj = []
                    # pdb.set_trace()
                    for f_ in range(int(frame) - self.seq_len, int(frame) + 1):
                        if f_ not in labels_obj[:, 0]:
                            traj_obj.append([0, 0, 0, 0])
                        else:
                            traj_obj.append(labels_obj[labels_obj[:, 0] == f_][0, 2:6].tolist())
                    # traj_obj = torch.cat(traj_obj)
                    trajs.append(traj_obj)
                # pdb.set_trace()
                scene['trajs'] = torch.tensor(trajs, dtype=torch.float32)
                # scene['label_next'] = labels[labels[:, 0] == frame]
                # pdb.set_trace()
                scenes.append(scene)
                # if i > 10:
                #     break
        self.scenes = scenes


    def init_videos(self):
        videos = {}
        for video_name in self.video_name_list:
            print(f'init videos {video_name}')
            videos[video_name] = {}
            image_list = sorted(os.listdir(os.path.join(self.dataroot, video_name, 'img')))
            # pdb.set_trace()
            text_array = pd.read_csv(os.path.join(self.dataroot, video_name, 'gt/gt.txt')).values
            tensor_label = torch.tensor(text_array, dtype=torch.float32)
            # 转为 x_min,y_min,x_max,y_max
            tensor_label[:, 4] += tensor_label[:, 2]
            tensor_label[:, 5] += tensor_label[:, 3]
            # pdb.set_trace()
            tensor_label[:, -1] = torch.ones((tensor_label.shape[0]))
            images = []
            for i in tqdm(range(len(image_list))):
                image_name = image_list[i]
                image_path = os.path.join(self.dataroot, video_name, 'img', image_name)
                image = cv2.imread(image_path)
                tensor_image = torch.tensor(image, dtype=torch.float32)
                tensor_image = tensor_image.permute(2, 0, 1)
                images.append(tensor_image)
            videos[video_name]['images'] = images
            videos[video_name]['labels'] = tensor_label
        self.videos = videos

    def __len__(self) -> int:
        # return min(len(self.dataset), 10)
        return len(self.scenes)

    def splitImages(self, tensor_image, tensor_traj):
        # 裁剪成512*512的图像，中间间隔32个像素
        tensor_image_list = []
        tensor_traj_list = []
        channels, image_height, image_width = tensor_image.shape

        # Patching size and stride
        patch_size = self.patch_size
        stride = self.stride

        # Calculate how many patches fit along the height and width
        num_patches_x = int((image_width - patch_size - 0.5) // stride + 2)
        num_patches_y = int((image_height - patch_size - 0.5) // stride + 2)

        zero_back = torch.zeros(size=(3, stride * num_patches_x + patch_size, stride * num_patches_y + patch_size), dtype=torch.float32)
        # 然后将 tensor_image 嵌入到 zero_back 中，并对 zero_back 进行裁剪
        # Embed the tensor_image into the zero-padded tensor (top-left corner)
        zero_back[:, 0:image_height, 0:image_width] = tensor_image

        # Perform the patching
        for y in range(num_patches_y):
            for x in range(num_patches_x):
                # Calculate the top-left and bottom-right coordinates of each patch
                y_min = y * stride
                x_min = x * stride
                y_max = y_min + patch_size
                x_max = x_min + patch_size

                # Crop the patch from the zero-padded image
                image_patch = zero_back[:, y_min:y_max, x_min:x_max]
                tensor_image_list.append(image_patch.unsqueeze(0))
                # pdb.set_trace()
                tensor_traj_patch = tensor_traj[tensor_traj[:, -1, 0] >= x_min]
                tensor_traj_patch = tensor_traj_patch[tensor_traj_patch[:, -1, 1] >= y_min]
                tensor_traj_patch = tensor_traj_patch[tensor_traj_patch[:, -1, 2] < x_max]
                tensor_traj_patch = tensor_traj_patch[tensor_traj_patch[:, -1, 3] < y_max]
                # pdb.set_trace()
                tensor_traj_patch[:, :, 0] -= x_min
                tensor_traj_patch[:, :, 1] -= y_min
                tensor_traj_patch[:, :, 2] -= x_min
                tensor_traj_patch[:, :, 3] -= y_min
                tensor_traj_list.append(tensor_traj_patch.unsqueeze(0))
                # pdb.set_trace()
        # pdb.set_trace()
        # tensor_traj_list = torch.cat(tensor_traj_list)
        tensor_image_list = torch.cat(tensor_image_list, dim=0)
        return tensor_image_list, tensor_traj_list

    def transform_data(self, tensor_image, tensor_traj_list):
        image_height_ratio = self.image_size / tensor_image.shape[2]  # 2.0
        image_width_ratio = self.image_size / tensor_image.shape[3]  # 2.0
        tensor_image = self.image_resize(tensor_image)  # torch.Size([9, 3, 1024, 1024])
        for i in range(len(tensor_traj_list)):
            tensor_traj_list[i][:, :, :, 0] = tensor_traj_list[i][:, :, :, 0] * image_width_ratio
            tensor_traj_list[i][:, :, :, 1] = tensor_traj_list[i][:, :, :, 1] * image_height_ratio
            tensor_traj_list[i][:, :, :, 2] = tensor_traj_list[i][:, :, :, 2] * image_width_ratio
            tensor_traj_list[i][:, :, :, 3] = tensor_traj_list[i][:, :, :, 3] * image_height_ratio
        return tensor_image, tensor_traj_list

    def __getitem__(self, idx):
        # time1 = time.time()
        scene = self.scenes[idx]
        # pdb.set_trace()
        tensor_image = scene['image']

        # # TODO 可能要对label里面的坐标进行缩放
        # for i in range(len(self.image_transforms)):
        #     tensor_image = self.image_transforms[0](tensor_image)  # 1 * 3 * 1024 * 1024
        tensor_traj = scene['trajs']  # 1 * N_history * 10
        # tensor_label = scene['label_next']  # 1 * N_next * 10
        # tensor_image_list, tensor_traj, tensor_label = self.splitImages(tensor_image, tensor_traj, tensor_label)
        tensor_image, tensor_traj_list = self.splitImages(tensor_image, tensor_traj)
        tensor_image, tensor_traj_list = self.transform_data(tensor_image, tensor_traj_list)
        # time2 = time.time()
        # print('getitem time cost:', time2 - time1)  # 0.03秒
        return tensor_image, tensor_traj_list
