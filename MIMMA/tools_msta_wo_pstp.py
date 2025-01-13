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
# from progress.bar import Bar
from torch.utils.data import DataLoader
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import Counter

from tracking.dataset_xysc import batch_process_coords, collate_batch_mot, read_row_files, MultiPersonTrajPoseDataset
from tracking.model_ship import create_model, hungarian_matching_mmsi, hungarian_matching_traj, inference

from tracking.dataset_xysc import BYTETracker, STrack, KalmanFilter

# from classify.model_ship_img import rgb_classify, init_rgb_model, init_rgb_dataset, sar_classify, init_sar_model, init_sar_dataset

class MultiSensorShipTrackingModel:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tracking_model, self.tracking_config = init_tracking_model(args=self.args)
        self.in_F, self.out_F = self.tracking_config['TRAIN']['input_track_size'], self.tracking_config['TRAIN']['output_track_size']
        # self.ship_traj_dict_satellite = {}

        self.ship_traj_dict = {}
        self.ship_traj_dict_pred = {}
        self.ship_traj_break_time_dict = {}
        self.ship_traj_finish_dict = {}

        self.finish_time_max = 30
        self.pred_mmsi = 0
        self.merge_t_segment = self.in_F
        self.merge_t_interval = 1
        self.merge_t_thresh = 0.08  # 8000m
        self.association_long_thresh = 0.08  # 8000m
        self.association_short_thresh = 0.08  # 8000m
        self.device = torch.device('cuda:0')
        self.frame_id = 0

    def add_new_traj(self, obs_array_t, matched_obs_idx_list):
        # 实在没有匹配上的观测点则作为新的轨迹
        for i in range(obs_array_t.shape[0]):
            if i not in matched_obs_idx_list:
                # resource --> point_id, behavior, label1, label2, source
                self.ship_traj_dict[self.pred_mmsi] = {'traj':[], 'resource':[]}
                self.ship_traj_dict[self.pred_mmsi]['traj'].append(obs_array_t[i])
                self.ship_traj_dict_pred[self.pred_mmsi] = {'traj':[], 'resource':[]}
                self.ship_traj_dict_pred[self.pred_mmsi]['traj'].append(obs_array_t[i])
                self.ship_traj_break_time_dict[self.pred_mmsi] = 0
                self.pred_mmsi += 1

    def remove_duplicate_stracks_v1(self, association_map):
        points = []
        k_list = []
        points_start = []
        for k in list(self.ship_traj_dict.keys()):
            ship_traj = self.ship_traj_dict[k]['traj']
            points.append(ship_traj[-1])
            points_start.append(ship_traj[0])
            k_list.append(k)
        points = np.array(points, dtype=object)
        if points.shape[0] == 0:
            return
        points_xy = points[:, 3:5].astype(np.float64)
        # pdb.set_trace()
        cost_matrix = np.linalg.norm(points_xy[:, np.newaxis] - points_xy, axis=2)
        remain_k_list = []
        remove_id_list = []
        # pdb.set_trace()
        for p in range(cost_matrix.shape[0]):
            if p in remove_id_list:
                continue
            low_cost_idx = np.argwhere(cost_matrix[p] < 0.01)
            timep = points[p][0] - points_start[p][0]
            longest_idx = [p, timep]
            # print(low_cost_idx)
            # print(' ')
            # pdb.set_trace()
            for q in low_cost_idx:
                q = q[0]
                # pdb.set_trace()
                if q in remove_id_list:
                    continue
                timeq = points[q][0] - points_start[q][0]
                # print(timeq)
                # pdb.set_trace()
                if timeq > longest_idx[1]:
                    longest_idx = [q, timeq]
                remove_id_list.append(q)
            remain_k_list.append(k_list[longest_idx[0]])
        # pdb.set_trace()
        for k in list(self.ship_traj_dict.keys()):
            if k not in remain_k_list:
                self.ship_traj_finish_dict[k] = self.ship_traj_dict_pred[k]
                del self.ship_traj_dict[k]
                del self.ship_traj_dict_pred[k]
                del self.ship_traj_break_time_dict[k]

    def remove_duplicate_stracks(self, association_map):
        if len(association_map) == 0:
            return
        points = []
        k_list = []
        points_start = []
        for k in list(self.ship_traj_dict.keys()):
            ship_traj = self.ship_traj_dict[k]['traj']
            points.append(ship_traj[-1])
            points_start.append(ship_traj[0])
            k_list.append(k)
        if len(points) == 0:
            return
        # remain_k_list = []
        # remove_id_list = []
        # # pdb.set_trace()
        # for thresh in range(1, 100):
        #     remain_k_list = []
        #     remove_id_list = []
        #     for p in range(association_map.shape[0]):
        #         if p in remove_id_list:
        #             continue
        #         low_cost_idx = np.argwhere(association_map[p] > thresh / 100)
        #         timep = points[p][0] - points_start[p][0]
        #         longest_idx = [p, timep]
        #         # print(low_cost_idx)
        #         # print(' ')
        #         # pdb.set_trace()
        #         for q in low_cost_idx:
        #             q = q[0]
        #             # pdb.set_trace()
        #             if q in remove_id_list:
        #                 continue
        #             timeq = points[q][0] - points_start[q][0]
        #             # print(timeq)
        #             # pdb.set_trace()
        #             if timeq > longest_idx[1]:
        #                 longest_idx = [q, timeq]
        #             remove_id_list.append(q)
        #         remain_k_list.append(k_list[longest_idx[0]])
        #     print(remain_k_list, thresh / 100)
        # pdb.set_trace()

        remain_k_list = []
        remove_id_list = []
        # if association_map.shape[0] != len(points):
        #     pdb.set_trace()
        if association_map.shape[0] > len(points):
            association_map = association_map[:len(points), :len(points)]
        range_len = min(association_map.shape[0], len(points))
        for p in range(range_len):
            if p in remove_id_list:
                continue
            low_cost_idx = np.argwhere(association_map[p] > 0.5)
            timep = points[p][0] - points_start[p][0]
            longest_idx = [p, timep]
            # print(low_cost_idx)
            # print(' ')
            # pdb.set_trace()
            for q in low_cost_idx:
                q = q[0]
                # pdb.set_trace()
                if q in remove_id_list:
                    continue
                timeq = points[q][0] - points_start[q][0]
                # print(timeq)
                # pdb.set_trace()
                if timeq > longest_idx[1]:
                    longest_idx = [q, timeq]
                remove_id_list.append(q)
            remain_k_list.append(k_list[longest_idx[0]])

        for k in list(self.ship_traj_dict.keys()):
            if k not in remain_k_list:
                self.ship_traj_finish_dict[k] = self.ship_traj_dict_pred[k]
                del self.ship_traj_dict[k]
                del self.ship_traj_dict_pred[k]
                del self.ship_traj_break_time_dict[k]

    def traj_preprocess(self):
        ship_traj_list = []
        ship_mmsi_list = []
        for k in list(self.ship_traj_dict.keys()):
            ship_traj = self.ship_traj_dict[k]['traj']
            if len(ship_traj) >= self.in_F:
                ship_traj = ship_traj[-self.in_F:]
            times, indices = np.unique(np.array(ship_traj)[:, 0], return_index=True)
            missing_time_range = np.setdiff1d(np.arange(max(times) - self.in_F + 1, max(times), 1), times)
            # pdb.set_trace()
            missing_bottom = np.array(list(ship_traj[-1][1] for i in range(missing_time_range.shape[0])), dtype=object)
            missing_right = np.array(list(ship_traj[-1][2] for i in range(missing_time_range.shape[0])), dtype=object)
            missing_top = np.array(list(ship_traj[-1][3] for i in range(missing_time_range.shape[0])), dtype=object)
            missing_left = np.array(list(ship_traj[-1][4] for i in range(missing_time_range.shape[0])), dtype=object)
            missing_sog = np.array(list(ship_traj[-1][5] for i in range(missing_time_range.shape[0])), dtype=object)
            missing_cog = np.array(list(ship_traj[-1][6] for i in range(missing_time_range.shape[0])), dtype=object)
            # missing_MMSI = np.array(list(k for i in range(missing_time_range.shape[0])), dtype=object)
            missing_source = np.array(list('pred' for i in range(missing_time_range.shape[0])), dtype=object)
            missing_satellite_id = np.array(list('0' for i in range(missing_time_range.shape[0])), dtype=object)
            missing_points = np.column_stack((missing_time_range,
                                      missing_bottom, missing_right, missing_top, missing_left,
                                      missing_sog, missing_cog, missing_source, missing_satellite_id))
            # if self.frame_id == 2:
            #     pdb.set_trace()
            ship_traj = np.vstack((np.array(ship_traj, dtype=object), missing_points))
            sorted_indices = np.argsort(ship_traj[:, 0])
            ship_traj = ship_traj[sorted_indices]
            ship_traj = ship_traj[-self.in_F:]
            ship_traj_list.append(ship_traj)
            ship_mmsi_list.append(k)
        # if self.frame_id == 38:
        #     pdb.set_trace()
        # pdb.set_trace()
        return ship_traj_list, ship_mmsi_list

    def traj_predict(self, ship_traj_list):
        # 设置一个临时的用于保存已被分配的观测点的list
        # traj_pred_result_points = []
        # pdb.set_trace()
        pdo = 0
        ef = 0
        # print(ship_traj_long.shape)
        start_time1 = time.time()
        obs = ship_traj_list
        # pdb.set_trace()
        dataset = MultiPersonTrajPoseDataset(obs, split=self.args.split, track_size=(self.in_F+self.out_F), track_cutoff=self.in_F)
        end_time1 = time.time()
        dataloader = DataLoader(dataset, batch_size=32, num_workers=self.tracking_config['TRAIN']['num_workers'],
                                shuffle=False, collate_fn=collate_batch_mot)

        pdo += (end_time1 - start_time1)
        # 轨迹预测
        start_time2 = time.time()
        pred_joints_list, refine_joints_list, asso_feature_list = evaluate_frame(self.tracking_model, self.args.modality, dataloader, self.tracking_config)
        end_time2 = time.time()
        ef += (end_time2 - start_time2)
        # todo 使用算法对所有轨迹进行预测
        if pred_joints_list == []:
            return pred_joints_list, asso_feature_list
        pred_joints_list = np.array(pred_joints_list)[:, 0, 0, :]
        pred_joints_list[:, [0, 1]] = pred_joints_list[:, [1, 0]]
        # norms = torch.norm(asso_feature_list, p=2, dim=1, keepdim=True)  # 计算每行的 L2 范数
        # normalized_matrix = asso_feature_list / norms  # 归一化
        # association_map = torch.mm(normalized_matrix, normalized_matrix.t())
        # association_map[association_map >= 0.5] = 1
        # association_map[association_map < 0.5] = 0

        matrix = cdist(asso_feature_list.cpu().detach().numpy(), asso_feature_list.cpu().detach().numpy(), metric='euclidean')
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        if min_val == max_val:
            association_map = matrix
        else:
            normalized_matrix = (matrix - min_val) / (max_val - min_val)
            association_map = 1 - normalized_matrix
        # association_map[association_map >= 0.5] = 1
        # association_map[association_map < 0.5] = 0

        # print(association_map)
        # if self.frame_id == 30:
        #     pdb.set_trace()
        # print('pdo:', pdo, end=' ')
        # print('ef:', ef, end=' ')
        # pdb.set_trace()
        return pred_joints_list, association_map

    def traj_update(self, obs_array_t, traj_pred_result_points, ship_mmsi, ship_traj):
        matched_obs_idx_list = []
        matched_pred_idx_list = []
        if obs_array_t.shape[0] == 0 or traj_pred_result_points == []:
            matches = []
            cost_matrix = []
        else:
            # 暂时就用下一帧，以后可以考虑中断多长时间就用第几帧
            traj_pred_result_points_use = []

            # matches, cost_matrix = hungarian_matching_traj(obs_array_t[:, 3:5].astype(np.float32), traj_pred_result_points[:, :2].astype(np.float32))  # 选用第t+1帧的预测结果
            matches, cost_matrix = hungarian_matching_traj(traj_pred_result_points[:, :2].astype(np.float32), obs_array_t[:, 3:5].astype(np.float32))  # 选用第t+1帧的预测结果

        # pdb.set_trace()
        for pred_idx, obs_idx in matches:
            # 对于匹配上的{观测点,预测点}，将预测点加入到ship_traj_list中
            if cost_matrix[pred_idx, obs_idx] >= self.association_long_thresh:
                continue
            # pdb.set_trace()
            # if obs_idx in matched_obs_idx_list:
            #     continue
            # obs_array_t.append([str(poing_id), int(mmsi), int(f), float(lat), float(lon), float(s), float(c), str(source)])
            # pdb.set_trace()
            # 计算长宽，直接使用观测的长宽
            h, w = abs(obs_array_t[obs_idx, 1] - obs_array_t[obs_idx, 3]), abs(obs_array_t[obs_idx, 2] - obs_array_t[obs_idx, 4])

            self.ship_traj_dict_pred[ship_mmsi[pred_idx]]['traj'].append(np.array([obs_array_t[0, 0], traj_pred_result_points[pred_idx][0] - h,
                                                                                            traj_pred_result_points[pred_idx][1] - w,
                                                                                            *traj_pred_result_points[pred_idx],
                                                                                            *obs_array_t[obs_idx][5:]], dtype=object))
                # self.ship_traj_dict_pred[ship_mmsi[pred_idx]]['traj'][-1][-2] = 'pred'
            self.ship_traj_dict[ship_mmsi[pred_idx]]['traj'].append(obs_array_t[obs_idx])
            self.ship_traj_break_time_dict[ship_mmsi[pred_idx]] = 0
            matched_obs_idx_list.append(obs_idx)
            matched_pred_idx_list.append(pred_idx)
        # 没有匹配上的轨迹点则直接将预测结果加入到ship_traj_list中 或者 直接剔除掉？ ByteTrack里面是不保存这部分结果的
        for i in range(len(ship_mmsi)):
            if i not in matched_pred_idx_list:
                # 更新丢失次数，当丢失次数超过阈值就判定为结束的轨迹
                # last_ship_point = self.ship_traj_dict[ship_mmsi[i]]['traj'][-1]
                # pdb.set_trace()
                # crt_t = last_ship_point[0] + 1
                # h, w = abs(last_ship_point[1] - last_ship_point[3]), abs(last_ship_point[2] - last_ship_point[4])
                # self.ship_traj_dict[ship_mmsi[i]]['traj'].append(np.array([crt_t, 0, 0, 0, 0, 0, 0, 'pred', 'satellite_zero'], dtype=object))
                # # last_ship_point_pred = self.ship_traj_dict_pred[ship_mmsi[i]]['traj'][-1]
                # self.ship_traj_dict_pred[ship_mmsi[i]]['traj'].append(np.array([crt_t, 0, 0, 0, 0, 0, 0, 'pred', 'satellite_zero'], dtype=object))
                # pdb.set_trace()
                self.ship_traj_break_time_dict[ship_mmsi[i]] += 1
                if self.ship_traj_break_time_dict[ship_mmsi[i]] >= self.finish_time_max:
                    # pdb.set_trace()
                    self.ship_traj_finish_dict[ship_mmsi[i]] = self.ship_traj_dict_pred[ship_mmsi[i]]
                    del self.ship_traj_dict[ship_mmsi[i]]
                    del self.ship_traj_dict_pred[ship_mmsi[i]]
                    del self.ship_traj_break_time_dict[ship_mmsi[i]]
        return matched_obs_idx_list


    def multi_object_tracking(self, obs_array_t, obs_mmsi_t):
        self.frame_id += 1
        # crt_t = obs_array_t[0, 0]
        # if crt_t == 106:
        #     pdb.set_trace()
        self.obs_mmsi_t = obs_mmsi_t
        start_time = time.time()
        ship_traj_list, ship_mmsi_list = self.traj_preprocess()
        traj_pred_result_points, association_map = self.traj_predict(np.array(ship_traj_list, dtype=object))
        matched_obs_idx_list = self.traj_update(obs_array_t, traj_pred_result_points, ship_mmsi_list, ship_traj_list)
        self.remove_duplicate_stracks(association_map)
        # pdb.set_trace()
        self.add_new_traj(obs_array_t, matched_obs_idx_list)
        # end_time = time.time()
        # print(f'MOT TC: {end_time - start_time:.2f} s', end=' ')
        return self.get_ship_traj_dict()


    def get_ship_traj_dict(self):
        # ship_traj_long_dict_all = {}
        ship_traj_short_dict_active_all = {}
        ship_traj_short_dict_finish_all = {}
        ship_traj_long_dict_active_all = {}
        ship_traj_long_dict_finish_all = {}
        # todo 将resource中的信息也保存下来
        for traj_id in list(self.ship_traj_dict_pred.keys()):
            traj = self.ship_traj_dict_pred[traj_id]['traj']
            # resource = self.ship_traj_dict_pred[traj_id]['resource']
            if len(traj) < 3:
                ship_traj_short_dict_active_all[traj_id] = {}
                ship_traj_short_dict_active_all[traj_id]['traj'] = traj
                # ship_traj_short_dict_all[traj_id]['resource'] = resource
            else:
                ship_traj_long_dict_active_all[traj_id] = {}
                ship_traj_long_dict_active_all[traj_id]['traj'] = traj
                # ship_traj_long_dict_all[traj_id]['resource'] = resource
        for traj_id in list(self.ship_traj_finish_dict.keys()):
            traj = self.ship_traj_finish_dict[traj_id]['traj']
            # resource = self.ship_traj_dict_pred[traj_id]['resource']
            if len(traj) < 3:
                ship_traj_short_dict_finish_all[traj_id] = {}
                ship_traj_short_dict_finish_all[traj_id]['traj'] = traj
                # ship_traj_short_dict_all[traj_id]['resource'] = resource
            else:
                ship_traj_long_dict_finish_all[traj_id] = {}
                ship_traj_long_dict_finish_all[traj_id]['traj'] = traj
                # ship_traj_long_dict_all[traj_id]['resource'] = resource
        return ship_traj_long_dict_active_all, ship_traj_short_dict_active_all, ship_traj_long_dict_finish_all, ship_traj_short_dict_finish_all, self.ship_traj_finish_dict, self.ship_traj_dict_pred


    def save_tracking_traj_dict(self, args):
        # 保存所有预测结果
        ship_traj_long_dict_active_all, ship_traj_short_dict_active_all, ship_traj_long_dict_finish_all, ship_traj_short_dict_finish_all, ship_traj_finish_dict, ship_traj_dict_pred = self.get_ship_traj_dict()
        # ship_traj_dict_all, _, ship_traj_finish_dict_all, _ = self.get_ship_traj_dict()
        # pdb.set_trace()
        # ship_traj_long_dict_active_all = ship_traj_dict_pred
        # ship_traj_long_dict_finish_all = ship_traj_finish_dict
        os.makedirs(os.path.join(args.pred_root, args.dataset_name, args.exp_name), exist_ok=True)

        ship_traj_list_active = []
        long_active_len = []
        for k in list(ship_traj_long_dict_active_all.keys()):
            ship_traj = ship_traj_long_dict_active_all[k]['traj']
            # if len(ship_traj) == 1:
            #     continue
            long_active_len.append(len(ship_traj))
            for i in range(len(ship_traj)):
                if type(ship_traj[i]) != type([]):
                    ship_traj[i] = ship_traj[i].tolist()
                ship_traj[i].append(k)
            ship_traj_list_active += ship_traj
        print(long_active_len, np.mean(np.array(long_active_len)))
        ship_traj_list_active = np.array(ship_traj_list_active, dtype=object)
        ship_traj_list_active_match = np.empty((0, 10))
        ship_traj_list_active_unmatch = np.empty((0, 10))
        if ship_traj_list_active.shape[0] != 0:
            ship_traj_list_active_match = ship_traj_list_active[ship_traj_list_active[:, -3] != 'pred']
            ship_traj_list_active_unmatch = ship_traj_list_active[ship_traj_list_active[:, -3] == 'pred']
        ship_traj_list_finish = []
        for k in list(ship_traj_long_dict_finish_all.keys()):
            ship_traj = ship_traj_long_dict_finish_all[k]['traj']
            # if len(ship_traj) == 1:
            #     continue
            for i in range(len(ship_traj)):
                if type(ship_traj[i]) != type([]):
                    ship_traj[i] = ship_traj[i].tolist()
                ship_traj[i].append(k)
            ship_traj_list_finish += ship_traj
        ship_traj_list_finish = np.array(ship_traj_list_finish, dtype=object)
        # pdb.set_trace()
        ship_traj_list_finish_match = np.empty((0, 10))
        ship_traj_list_finish_unmatch = np.empty((0, 10))
        if ship_traj_list_finish.shape[0] != 0:
            ship_traj_list_finish_match = ship_traj_list_finish[ship_traj_list_finish[:, -3] != 'pred']
            ship_traj_list_finish_unmatch = ship_traj_list_finish[ship_traj_list_finish[:, -3] == 'pred']
        # # 第一种，仅 active_match
        # self.save_traj(args, ship_traj_list_active_match, '_active_match.txt')
        # # 第二种， active_match + active_unmatch
        # # pdb.set_trace()
        # save_trajs = np.concatenate([ship_traj_list_active_match, ship_traj_list_active_unmatch])
        # self.save_traj(args, save_trajs, '_active.txt')
        # # 第3种， active_match + finish_match
        # save_trajs = np.concatenate([ship_traj_list_active_match, ship_traj_list_finish_match])
        # self.save_traj(args, save_trajs, '_match.txt')
        # # 第4种， active_match + finish_unmatch
        # save_trajs = np.concatenate([ship_traj_list_active_match, ship_traj_list_finish_unmatch])
        # self.save_traj(args, save_trajs, '_amfu.txt')
        # # 第5种， active_match + active_unmatch + finish_match
        # save_trajs = np.concatenate([ship_traj_list_active_match, ship_traj_list_active_unmatch, ship_traj_list_finish_match])
        # self.save_traj(args, save_trajs, '_active_fm.txt')
        # # 第6种， active_match + finish_match + finish_unmatch
        # save_trajs = np.concatenate([ship_traj_list_active_match, ship_traj_list_finish_match, ship_traj_list_finish_unmatch])
        # self.save_traj(args, save_trajs, '_match_fu.txt')
        # # 第7种， active_match + active_unmatch + finish_match
        # save_trajs = np.concatenate([ship_traj_list_active_match, ship_traj_list_finish_unmatch, ship_traj_list_active_unmatch])
        # self.save_traj(args, save_trajs, '_active_fu.txt')

        # 第8种， active_match + active_unmatch + finish_match + finish_unmatch
        save_trajs = np.concatenate([ship_traj_list_active_match, ship_traj_list_finish_match, ship_traj_list_active_unmatch, ship_traj_list_finish_unmatch])
        self.save_traj(args, save_trajs, '.txt')

    def save_traj(self, args, save_trajs, last_name):
        sorted_indices = np.lexsort((save_trajs[:, -1], save_trajs[:, 0]))
        save_trajs = save_trajs[sorted_indices]
        wf = open(os.path.join(args.pred_root, args.dataset_name, args.exp_name, args.file_name.split('.')[0] + last_name), 'w')
        for idx in range(len(save_trajs)):
            t, y_b, x_r, y_t, x_l, s, c, source, satellite_id, k = list(save_trajs[idx])
            wf.write(str((t+1)) + ',' + str(k) + ',' + str(x_l) + ',' + str(y_t) + ',' + str(abs(y_t - y_b)) + ',' + str(abs(x_l - x_r)) + ',-1,-1,-1,-1' + '\n')
        wf.close()

    def save_gt_traj_dict(self, save_path, gt_array):
        # 保存所有真实轨迹
        os.makedirs(os.path.join(save_path, 'ship_traj_gt'), exist_ok=True)
        ped_list = np.unique(gt_array[:, 1])
        for ped in ped_list:
            ship_traj = gt_array[gt_array[:, 1] == ped]
            # import pdb
            # pdb.set_trace()
            txt_name = str((ped)) + '.txt'
            wf = open(os.path.join(save_path, 'ship_traj_gt', txt_name), 'w')
            for idx in range(len(ship_traj)):
                pid, p, f, x, y, s, c, b, label1, label2 = list(ship_traj[idx])
                wf.write(str((pid)) + ',' + str(int(p)) + ',' + str(int(float(f))) + ',' + str(x) + ',' + str(y) + ',' +
                         str(s) + ',' + str(c) + ',' + str(b) + ',' + str(label1) + ',' + str(label2) + '\n')
            wf.close()


def evaluate_frame(model, modality_selection, dataloader, config):
    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    # bar = Bar(f"EVAL ADE_FDE", fill="#", max=len(dataloader))

    pred_joints_list = []
    refine_joints_list = []
    asso_feature_list = []
    # import pdb
    # pdb.set_trace()
    for i, batch in enumerate(dataloader):
        # torch.Size([64, 5, 21, 2, 4]) --> joints.shape  -> N=5
        # torch.Size([2, 4, 21, 2, 4]) --> joints.shape  -> N=5
        joints, masks, mmsi_list, padding_mask = batch
        # if joints.shape[2] != 30:
        #     # pdb.set_trace()
        #     tmp_a = joints.shape[2]
        #     joints = torch.cat([joints, joints[:, :, -(in_F - tmp_a):, :, :]], dim=2)
        #     masks = torch.cat([masks, masks[:, :, -(in_F - tmp_a):, :]], dim=2)
        # import pdb
        # pdb.set_trace()
        # print('i, joints.shape', i, joints.shape)
        padding_mask = padding_mask.to(config["DEVICE"])
        # import pdb
        # pdb.set_trace()
        # torch.Size([64, 9, 10, 4]) --> in_joints.shape
        in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(joints, masks, padding_mask, config, modality_selection)
        # import pdb
        # pdb.set_trace()
        pred_output_joints, refine_output_joints, asso_feature = inference(model, config, in_joints, padding_mask, out_len=out_F)
        # import pdb
        # pdb.set_trace()
        for j in range(pred_output_joints.shape[0]):
            pred_output_joints[j, :, 0, 0] += joints[j, 0, -1, 0, 0]
            pred_output_joints[j, :, 0, 1] += joints[j, 0, -1, 0, 1]
        for j in range(refine_output_joints.shape[0]):
            refine_output_joints[j, :, 0, 0] += joints[j, 0, -1, 0, 0]
            refine_output_joints[j, :, 0, 1] += joints[j, 0, -1, 0, 1]
        # import pdb
        # pdb.set_trace()

        # out_joints = out_joints.cpu()  # torch.Size([64, 12, 10, 4])  10 == 5 * 2
        # pdb.set_trace()
        pred_output_joints = pred_output_joints.cpu().reshape(in_joints.size(0), out_F, 1, 2)  # torch.Size([64, 12, 1, 2])
        pred_joints_list.append(pred_output_joints)
        refine_output_joints = refine_output_joints.cpu().reshape(in_joints.size(0), in_F, 1, 2)  # torch.Size([64, 12, 1, 2])
        refine_joints_list.append(refine_output_joints)
        asso_feature_list.append(asso_feature.view(-1, 128).cpu())
        # pdb.set_trace()
    # import pdb
    # pdb.set_trace()
    if pred_joints_list == []:
        # pdb.set_trace()
        return pred_joints_list, refine_joints_list, asso_feature_list
    # pdb.set_trace()
    pred_joints_list = np.concatenate(pred_joints_list, axis=0)
    refine_joints_list = np.concatenate(refine_joints_list, axis=0)
    asso_feature_list = torch.cat(asso_feature_list, dim=0)
    # pdb.set_trace()
    return pred_joints_list, refine_joints_list, asso_feature_list




def init_tracking_model(args):
    ################################
    # Initialize tracking model
    ################################
    print('load tracking model...')
    tracking_ckpt = torch.load(os.path.join(args.ckpt_root, args.dataset_name, 'checkpoints/best_val_checkpoint.pth.tar'), map_location = torch.device('cuda:0'))
    tracking_config = tracking_ckpt['config']
    if torch.cuda.is_available():
        tracking_config["DEVICE"] = f"cuda:1"
        torch.cuda.manual_seed(0)
    else:
        tracking_config["DEVICE"] = "cpu"
    tracking_model = create_model(tracking_config)
    tracking_model.load_state_dict(tracking_ckpt['model'])
    tracking_config["mode"] = 'mot'
    return tracking_model, tracking_config


def plot_points(args, obs_array, id_f, t):
    os.makedirs(os.path.join(args.plot_dir, str(id_f)), exist_ok=True)
    # 从数据中提取 lat (纬度), lon (经度), 和 mmsi
    pdb.set_trace()
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
    plt.savefig(os.path.join(args.plot_dir, str(id_f), str(t) + '.png'), format='png', dpi=300)
    plt.close()







