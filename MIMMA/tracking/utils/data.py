import os
import pdb
import numpy as np
from tracking.utils.utils import path_to_data
from tracking.utils.trajnetplusplustools import Reader_ship_xysc
import torch

# 用于MOT测试
def load_mot_traj(obs):
    joint_and_mask=[]
    #  train_scenes  --> 59 * 3 ==>[filename, scene_id, paths]
    #  paths --> 59 * 9 * 14
    ## Iterate over file names
    reader = Reader_ship_xysc(obs=obs, scene_type='paths')
    scene = [(s_id, s) for s_id, s in reader.scenes()]
    # import pdb
    # pdb.set_trace()
    for scene_i, (scene_id, paths) in enumerate(scene):
        # print(filename, scene_id)
        # scene_train --> array
        scene_train, mmsi, frames = Reader_ship_xysc.paths_to_xy(paths)
        # print(scene_train.shape)  # (21, 10, 8)  (9, 59, 8)
        # if scene_train.shape[0] != 30:
        #     pdb.set_trace()
        scene_train = drop_ped_with_missing_frame(scene_train)
        # print(scene_train.shape)  # (21, 10, 8)  (9, 59, 8)
        scene_train, _ = drop_distant_far(scene_train)
        # print(scene_train.shape)  # (21, 10, 8)  (9, 49, 8)
        scene_train_real = scene_train.reshape(scene_train.shape[0],scene_train.shape[1],-1,4) ##(21, n, 16, 4) #jjjjjjjjjjjjjjjjjjjjjj 2j
        # print(scene_train_real.shape)  # (21, 10, 2, 4)  (9, 49, 2, 4)
        scene_train_real_ped = np.transpose(scene_train_real,(1,0,2,3)) ## (n, 21, 16, 3)
        # print(scene_train_real_ped.shape)  # (10, 21, 2, 4)  (49, 9, 2, 4)

        scene_train_mask = np.ones(scene_train_real_ped.shape[:-1])
        # print(scene_train_mask.shape)  # (10, 21, 2)  (49, 9, 2)
        joint_and_mask.append((np.asarray(scene_train_real_ped), np.asarray(scene_train_mask), mmsi, frames[0]))
        # import pdb
        # pdb.set_trace()

    return joint_and_mask

def load_msta_traj(split, dataset_root, dataset_name):
    joint_and_mask=[]
    ############## change dataset path
    train_scenes, _, _ = prepare_data(os.path.join(dataset_root, dataset_name), subset=split, sample=1.0, goals=False, dataset_name=dataset_name)
    for scene_i, (filename, scene_id, paths) in enumerate(train_scenes):
        scene_train, mmsi, frames = Reader_ship_xysc.paths_to_xy(paths)
        scene_train = drop_ped_with_missing_frame(scene_train)
        scene_train, _ = drop_distant_far(scene_train)
        scene_train_real = scene_train.reshape(scene_train.shape[0],scene_train.shape[1],-1,4) ##(21, n, 16, 4) #jjjjjjjjjjjjjjjjjjjjjj 2j
        scene_train_real_ped = np.transpose(scene_train_real,(1,0,2,3)) ## (n, 21, 16, 3)
        scene_train_mask = np.ones(scene_train_real_ped.shape[:-1])
        joint_and_mask.append((np.asarray(scene_train_real_ped), np.asarray(scene_train_mask), mmsi, frames[0]))
    return joint_and_mask

def prepare_data(path, subset='/train/', sample=1.0, goals=True, dataset_name=''):
    all_scenes = []
    ## List file names  可能和自己想象的顺序不一样
    files = [f.split('.')[-2] for f in os.listdir(os.path.join(path, subset)) if f.endswith('.ndjson')]
    ## Iterate over file names
    # if dataset_name == 'VISO' or dataset_name == 'MSTAv2' or dataset_name == 'MSTA_lite':
    #     for file in files:
    #         reader = Reader_ship_xysc(os.path.join(path, subset, file + '.ndjson'), scene_type='paths')
    #         scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
    #         all_scenes += scene
    #     return all_scenes, None, True
    # else:
    #     print('not implement this dataset, error from utils/data.py')
    #     exit()
    for file in files:
        reader = Reader_ship_xysc(os.path.join(path, subset, file + '.ndjson'), scene_type='paths')
        scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
        all_scenes += scene
    return all_scenes, None, True

def drop_ped_with_missing_frame(xy):
    xy_n_t = np.transpose(xy, (1, 0, 2))
    mask = np.ones(xy_n_t.shape[0], dtype=bool)
    for n in range(xy_n_t.shape[0]-1):
        for t in range(9):
            if np.isnan(xy_n_t[n+1, t, 0]) == True:
                mask[n+1] = False
                break
    return np.transpose(xy_n_t[mask], (1, 0, 2))

def drop_distant_far(xy, r=1):
    distance_2 = np.sum(np.square(xy[:, :, 0:2] - xy[:, 0:1, 0:2]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r**2
    return xy[:, mask], mask

