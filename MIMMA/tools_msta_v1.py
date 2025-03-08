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
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import Counter

from tracking.dataset_xysc import batch_process_coords, collate_batch_mot, read_row_files, MultiPersonTrajPoseDataset
from tracking.model_ship import create_model, hungarian_matching_mmsi, hungarian_matching_traj, inference

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

        self.finish_time_max = self.in_F * 2
        self.pred_mmsi = 0
        self.merge_t_segment = self.in_F
        self.merge_t_interval = 1
        self.merge_t_thresh = 0.08  # 8000m
        self.association_long_thresh = 0.08  # 8000m
        self.association_short_thresh = 0.08  # 8000m
        self.device = torch.device('cuda:0')
        self.crt_t = 0

    def merge_trajs(self, traj_list):
        # Convert each trajectory array into a dictionary for easier manipulation
        traj_dicts = []
        for arr in traj_list:
            traj_dict = {}
            # pdb.set_trace()
            for t, y_b, x_r, y_t, x_l, s, c, source, satellite_id in arr:
                # traj_dict[source][int(t)] = (x, y, s, c, source, b)  # Store (x, y, s, c) tuples indexed by time
                traj_dict[int(t)] = (y_b, x_r, y_t, x_l, s, c, source, satellite_id)  # Store (x, y, s, c) tuples indexed by time
            traj_dicts.append(traj_dict)
        # Find all unique time points
        all_times = sorted(set(t for traj in traj_dicts for t in traj.keys()))

        # Initialize a list to store the results
        results = []

        # Process each time point to align the data
        for t in all_times:
            # Initialize storage for values
            y_b_vals = []
            x_r_vals = []
            y_t_vals = []
            x_l_vals = []
            s_vals = []
            c_vals = []
            b_list = []
            so_list = []
            satellite_id_list = []

            for traj_dict in traj_dicts:
                if t in traj_dict:
                    y_b, x_r, y_t, x_l, s, c, source, satellite_id = traj_dict[t]
                    y_b_vals.append(float(y_b))
                    x_r_vals.append(float(x_r))
                    y_t_vals.append(float(y_t))
                    x_l_vals.append(float(x_l))
                    s_vals.append(float(s))
                    c_vals.append(float(c))
                    so_list.append(source)
                    satellite_id_list.append(satellite_id)
                else:
                    # If the time point is missing, append NaN
                    y_b_vals.append(float('nan'))
                    x_r_vals.append(float('nan'))
                    y_t_vals.append(float('nan'))
                    x_l_vals.append(float('nan'))
                    s_vals.append(float('nan'))
                    c_vals.append(float('nan'))
                    b_list.append('None')
                    so_list.append('pred')
                    satellite_id_list.append('None')

            # Calculate means, ignoring NaN values
            # pdb.set_trace()
            y_b_vals = np.nanmean(y_b_vals)
            x_r_vals = np.nanmean(x_r_vals)
            y_t_vals = np.nanmean(y_t_vals)
            x_l_vals = np.nanmean(x_l_vals)
            s_mean = np.nanmean(s_vals)
            c_mean = np.nanmean(c_vals)
            satellite_id_final = Counter(satellite_id_list).most_common(1)[0][0]
            so_final = 'radar'
            if 'image' in so_list:
                so_final = 'image'
            elif 'radar' in so_list:
                so_final = 'radar'
            elif 'pred' in so_list:
                so_final = 'pred'
            elif 'elec' in so_list:
                so_final = 'elec'
            # pdb.set_trace()
            # Append the results for the current time point
            results.append([t, y_b_vals, x_r_vals, y_t_vals, x_l_vals, s_mean, c_mean, so_final, satellite_id_final])

        # pdb.set_trace()
        return np.array(results, dtype=object)

    def multi_source_merging_working(self):
        start_time = time.time()
        merging_segment_list = []
        traj_id_list = []
        # mmsi_list = []
        # resource_list = []
        for traj_id in self.ship_traj_dict.keys():
            if len(self.ship_traj_dict[traj_id]['traj']) < self.merge_t_segment:
                continue
            # pdb.set_trace()
            merging_segment_list.append(np.array(self.ship_traj_dict[traj_id]['traj'], dtype=object)[-self.merge_t_segment:, 3:5])
            traj_id_list.append(traj_id)
            # pdb.set_trace()
            # mmsi_list.append(self.ship_traj_dict[traj_id]['resource'][0][-1])
            # resource_list.append(self.ship_traj_dict[traj_id]['resource'])
        # print('len(merging_segment_list)', len(merging_segment_list))
        if len(merging_segment_list) <= 1:
            return None
        # pdb.set_trace()
        merging_segment_list = np.concatenate(merging_segment_list).reshape(len(merging_segment_list), -1)
        merging_segment_list = merging_segment_list.astype(np.float32)
        distance_matrix = squareform(pdist(merging_segment_list, metric='euclidean')) / self.merge_t_segment
        merging_segment_finish_id_list = []
        # pdb.set_trace()
        # merge_pair_list = []
        # merge_mmsi_list = []
        merge_time = 0
        traj_list_len_list = []
        # traj_list_sum = 0
        for i in range(distance_matrix.shape[0]):
            if i in merging_segment_finish_id_list:
                continue
            # merge_pair = [[i, mmsi_list[i]]]
            merging_segment_finish_id_list.append(i)
            traj_list = [np.array(self.ship_traj_dict_pred[traj_id_list[i]]['traj']).astype(object)]
            # resource_i_list = resource_list[i]
            # pdb.set_trace()
            # print('delect traj:', self.ship_traj_dict_satellite[satellite_id_list[i]]['ship_traj_dict'][traj_id_list[i]])
            del self.ship_traj_dict[traj_id_list[i]]
            del self.ship_traj_dict_pred[traj_id_list[i]]
            for j in range(1, distance_matrix.shape[1]):
                if i == j:
                    continue
                if j in merging_segment_finish_id_list:
                    continue
                if distance_matrix[i, j] <= self.merge_t_thresh:
                    # if self.crt_t == 10:
                    #     print(self.ship_traj_dict_pred[traj_id_list[j]]['traj'])
                    traj_list.append(np.array(self.ship_traj_dict_pred[traj_id_list[j]]['traj']).astype(object))
                    # resource_i_list+=resource_list[j]
                    merging_segment_finish_id_list.append(j)
                    # print('delect traj:', self.ship_traj_dict_satellite[satellite_id_list[j]]['ship_traj_dict'][traj_id_list[j]])
                    del self.ship_traj_dict[traj_id_list[j]]
                    del self.ship_traj_dict_pred[traj_id_list[j]]
                    # merge_pair.append([j, mmsi_list[j]])
            # pdb.set_trace()
            # merge_pair_list.append(merge_pair)
            # merge_mmsi_list.append([mmsi_list[i], len(merge_pair)])
            # pdb.set_trace()
            # traj_list = traj_list.astype(np.float32)
            # traj_list_sum += len(traj_list)
            if len(traj_list) == 1:
                traj_merge = np.array(traj_list, dtype=object)[0]
            else:
                start_time1 = time.time()
                traj_merge = self.merge_trajs(traj_list)
                # pdb.set_trace()
                start_time2 = time.time()
                merge_time += (start_time2 - start_time1)
            traj_list_len_list.append(len(traj_list))
            # pdb.set_trace()
            # p_id_array = np.array([['merge'] for k in range(traj_merge.shape[0])])
            # mmsi_array = np.array([[traj_id_list[i]] for k in range(traj_merge.shape[0])])
            # # source_array = np.array([['merge_source'] for k in range(traj_merge.shape[0])])
            # # pdb.set_trace()
            # if p_id_array.ndim == 1 or mmsi_array.ndim == 1 or traj_merge.ndim == 1:
            #     pdb.set_trace()
            # traj_merge = np.concatenate([p_id_array, mmsi_array, traj_merge], axis=1, dtype=object)
            # pdb.set_trace()
            self.ship_traj_dict[traj_id_list[i]] = {}
            self.ship_traj_dict[traj_id_list[i]]['traj'] = traj_merge.tolist()
            # self.ship_traj_dict[traj_id_list[i]]['resource'] = resource_i_list
            self.ship_traj_dict_pred[traj_id_list[i]] = {}
            self.ship_traj_dict_pred[traj_id_list[i]]['traj'] = traj_merge.tolist()
            # self.ship_traj_dict_pred[traj_id_list[i]]['resource'] = resource_i_list
            # pdb.set_trace()
        # traj_num_list_2 = [len(self.ship_traj_dict_satellite[satellite_id_list[i]]['ship_traj_dict'].keys()) for i in range(np.unique(np.array(satellite_id_list)).shape[0])]
        # len(self.ship_traj_dict_satellite['satellite_3004772805']['ship_traj_dict'].keys())
        # merge_mmsi_list = np.array(merge_mmsi_list)
        # # pdb.set_trace()
        # merge_acc_num = len(merge_pair_list)
        # error_instance = []
        # for merge_pair in merge_pair_list:
        #     mmsi = merge_pair[0]
        #     for mmsi_j in merge_pair:
        #         if mmsi_j != mmsi:
        #             merge_acc_num-=1
        #             error_instance.append(merge_pair)
        #             break
        # if len(merge_pair_list) != 0:
        #     merge_acc = merge_acc_num / len(merge_pair_list)
        # pdb.set_trace()

        end_time = time.time()
        print(f'MSM TC: {end_time - start_time:.2f}, merge time cost: {merge_time:.2f} s', end=' ')

    def split_long_and_short_traj(self):
        ship_traj_long = []
        ship_mmsi_long = []
        ship_traj_short = []
        ship_mmsi_short = []
        # pdb.set_trace()
        for k in list(self.ship_traj_dict.keys()):
            ship_traj = self.ship_traj_dict[k]['traj']
            if len(ship_traj) >= self.in_F:
                ship_traj_long.append(ship_traj[-self.in_F:])
                ship_mmsi_long.append(k)
            else:
                ship_traj_short.append(ship_traj[-1])
                ship_mmsi_short.append(k)
        return ship_traj_long, ship_mmsi_long, ship_traj_short, ship_mmsi_short

    def long_traj_predict(self, ship_traj_long):
        traj_pred_result_points = []
        # pdb.set_trace()
        pdo = 0
        ef = 0
        start_time1 = time.time()
        # pdb.set_trace()
        obs = ship_traj_long
        dataset = MultiPersonTrajPoseDataset(obs, split=self.args.split, track_size=(self.in_F+self.out_F), track_cutoff=self.in_F)
        dataloader = DataLoader(dataset, batch_size=256, num_workers=self.tracking_config['TRAIN']['num_workers'],
                                shuffle=False, collate_fn=collate_batch_mot)
        end_time1 = time.time()
        pdo += (end_time1 - start_time1)
        start_time2 = time.time()
        pred_joints_list, refine_joints_list = evaluate_frame(self.tracking_model, self.args.modality, dataloader, self.tracking_config)
        end_time2 = time.time()
        ef += (end_time2 - start_time2)
        traj_pred_result_points.append(np.array(pred_joints_list[:, 0, 0, :]))  
        # pdb.set_trace()
        print('pdo:', pdo, end=' ')
        print('ef:', ef, end=' ')
        # pdb.set_trace()
        traj_pred_result_points = np.concatenate(traj_pred_result_points)
        traj_pred_result_points[:, [0, 1]] = traj_pred_result_points[:, [1, 0]]
        # pdb.set_trace()
        return traj_pred_result_points

    def long_traj_update(self, obs_array_t, traj_pred_result_points, ship_mmsi_long, ship_traj_long):
        crt_t = obs_array_t[0, 0]
        matched_obs_idx_list = []
        matched_pred_idx_list = []
        # matches, cost_matrix = hungarian_matching_traj(obs_array_t[:, 3:5].astype(np.float32), traj_pred_result_points[:, :2].astype(np.float32))  # 选用第t+1帧的预测结果
        matches, cost_matrix = hungarian_matching_traj(traj_pred_result_points[:, :2].astype(np.float32), obs_array_t[:, 3:5].astype(np.float32))  # 选用第t+1帧的预测结果
        # pdb.set_trace()
        for pred_idx, obs_idx in matches:
            if cost_matrix[pred_idx, obs_idx] >= self.association_long_thresh:
                continue
            # pdb.set_trace()
            if obs_idx in matched_obs_idx_list:
                continue
            # obs_array_t.append([str(poing_id), int(mmsi), int(f), float(lat), float(lon), float(s), float(c), str(source)])
            # pdb.set_trace()
            h, w = abs(obs_array_t[obs_idx, 1] - obs_array_t[obs_idx, 3]), abs(obs_array_t[obs_idx, 2] - obs_array_t[obs_idx, 4])
            if obs_array_t[obs_idx][-1] in ['radar']:
                self.ship_traj_dict_pred[ship_mmsi_long[pred_idx]]['traj'].append(np.array([*obs_array_t[obs_idx]], dtype=object))
            else:
                self.ship_traj_dict_pred[ship_mmsi_long[pred_idx]]['traj'].append(np.array([crt_t, traj_pred_result_points[pred_idx][0] - h,
                                                                                                traj_pred_result_points[pred_idx][1] - w,
                                                                                                *traj_pred_result_points[pred_idx],
                                                                                                *obs_array_t[obs_idx][5:]], dtype=object))
                self.ship_traj_dict_pred[ship_mmsi_long[pred_idx]]['traj'][-1][-2] = 'pred'
            self.ship_traj_dict[ship_mmsi_long[pred_idx]]['traj'].append(obs_array_t[obs_idx])
            self.ship_traj_break_time_dict[ship_mmsi_long[pred_idx]] = 0
            matched_obs_idx_list.append(obs_idx)
            matched_pred_idx_list.append(pred_idx)
        for i in range(len(ship_mmsi_long)):
            if i not in matched_pred_idx_list:
                h, w = abs(obs_array_t[obs_idx, 1] - obs_array_t[obs_idx, 3]), abs(obs_array_t[obs_idx, 2] - obs_array_t[obs_idx, 4])
                last_ship_point = self.ship_traj_dict[ship_mmsi_long[i]]['traj'][-1]
                # pdb.set_trace()
                self.ship_traj_dict[ship_mmsi_long[i]]['traj'].append(np.array([crt_t, traj_pred_result_points[i][0] - h,
                                                                                        traj_pred_result_points[i][1] - w,
                                                                                        *traj_pred_result_points[i], *last_ship_point[5:]], dtype=object))
                # last_ship_point_pred = self.ship_traj_dict_pred[ship_mmsi_long[i]]['traj'][-1]
                self.ship_traj_dict_pred[ship_mmsi_long[i]]['traj'].append(np.array([crt_t, traj_pred_result_points[i][0] - h,
                                                                                            traj_pred_result_points[i][1] - w,
                                                                                            *traj_pred_result_points[i], *last_ship_point[5:]], dtype=object))
                self.ship_traj_dict_pred[ship_mmsi_long[i]]['traj'][-1][-2] = 'pred'
                # pdb.set_trace()
                self.ship_traj_break_time_dict[ship_mmsi_long[i]] += 1
                if self.ship_traj_break_time_dict[ship_mmsi_long[i]] >= self.finish_time_max:
                    # pdb.set_trace()
                    self.ship_traj_finish_dict[ship_mmsi_long[i]] = self.ship_traj_dict_pred[ship_mmsi_long[i]]
                    del self.ship_traj_dict[ship_mmsi_long[i]]
                    del self.ship_traj_dict_pred[ship_mmsi_long[i]]
                    del self.ship_traj_break_time_dict[ship_mmsi_long[i]]
        return matched_obs_idx_list

    def short_traj_predict(self, ship_traj_short):
        traj_pred_points_short = []
        return ship_traj_short[:, 3:5].astype(np.float32)

    def short_traj_update(self, obs_array_t, traj_pred_points_short, ship_traj_short, ship_mmsi_short, matched_obs_idx_list):
        matched_pred_idx_list = []
        crt_t = obs_array_t[0, 0]
        matches, cost_matrix = hungarian_matching_traj(obs_array_t[:, 3:5].astype(np.float32), traj_pred_points_short)  # 选用第t+1帧的预测结果
        # pdb.set_trace()
        for obs_idx, pred_idx in matches:
            # 对于匹配上的{观测点,预测点}，将预测点加入到ship_traj_list中
            if cost_matrix[obs_idx, pred_idx] >= self.association_short_thresh:
                continue
            if obs_idx in matched_obs_idx_list:
                continue
            # import pdb
            # pdb.set_trace()
            for t1 in range(int(ship_traj_short[pred_idx, 0]) + 1, crt_t):
                # if type(self.ship_traj_dict[ship_mmsi_short[pred_idx]]['traj'][-1][3:7]) == type(['0']):
                #     pdb.set_trace()
                linear_data = obs_array_t[obs_idx][1:7].astype(np.float32) - np.array(self.ship_traj_dict[ship_mmsi_short[pred_idx]]['traj'][-1][1:7], dtype=np.float32)
                linear_data = linear_data * (t1 - int(ship_traj_short[pred_idx, 0]))
                linear_data = linear_data / (crt_t - int(ship_traj_short[pred_idx, 0]))
                linear_data = linear_data + np.array(self.ship_traj_dict[ship_mmsi_short[pred_idx]]['traj'][-1][1:7], dtype=np.float32)
                self.ship_traj_dict[ship_mmsi_short[pred_idx]]['traj'].append(np.array([t1, *linear_data, *obs_array_t[obs_idx][-2:]], dtype=object))
                self.ship_traj_dict_pred[ship_mmsi_short[pred_idx]]['traj'].append(np.array([t1, *linear_data, *obs_array_t[obs_idx][-2:]], dtype=object))
            self.ship_traj_dict[ship_mmsi_short[pred_idx]]['traj'].append(obs_array_t[obs_idx])
            self.ship_traj_dict_pred[ship_mmsi_short[pred_idx]]['traj'].append(np.array(obs_array_t[obs_idx], dtype=object))
            self.ship_traj_break_time_dict[ship_mmsi_short[pred_idx]] = 0
            matched_obs_idx_list.append(obs_idx)
            matched_pred_idx_list.append(pred_idx)
        for i in range(len(ship_mmsi_short)):
            if i not in matched_pred_idx_list:
                self.ship_traj_break_time_dict[ship_mmsi_short[i]] += 1
                if self.ship_traj_break_time_dict[ship_mmsi_short[i]] >= self.finish_time_max:
                    # ship_traj_finish_dict[ship_mmsi_short[i]] = ship_traj_dict[ship_mmsi_short[i]]
                    del self.ship_traj_dict[ship_mmsi_short[i]]
                    del self.ship_traj_dict_pred[ship_mmsi_short[i]]
                    del self.ship_traj_break_time_dict[ship_mmsi_short[i]]
        return matched_obs_idx_list


    def add_new_traj(self, obs_array_t, matched_obs_idx_list):
        for i in range(obs_array_t.shape[0]):
            if i not in matched_obs_idx_list:
                # resource --> point_id, behavior, label1, label2, source
                self.ship_traj_dict[self.pred_mmsi] = {'traj':[], 'resource':[]}
                self.ship_traj_dict[self.pred_mmsi]['traj'].append(obs_array_t[i])
                self.ship_traj_dict_pred[self.pred_mmsi] = {'traj':[], 'resource':[]}
                self.ship_traj_dict_pred[self.pred_mmsi]['traj'].append(obs_array_t[i])
                self.ship_traj_break_time_dict[self.pred_mmsi] = 0
                self.pred_mmsi += 1
                # print(self.pred_mmsi)
                # pdb.set_trace()

    def multi_object_tracking(self, obs_array_t):
        start_time = time.time()
        ship_traj_long, ship_mmsi_long, ship_traj_short, ship_mmsi_short = self.split_long_and_short_traj()
        matched_obs_idx_list = []
        if len(ship_traj_long) != 0:
            traj_pred_result_points = self.long_traj_predict(np.array(ship_traj_long, dtype=object))
            matched_obs_idx_list = self.long_traj_update(obs_array_t, traj_pred_result_points, ship_mmsi_long, ship_traj_long)
        if len(ship_traj_short) != 0:
            traj_pred_points_short = self.short_traj_predict(np.array(ship_traj_short, dtype=object))
            matched_obs_idx_list = self.short_traj_update(obs_array_t, traj_pred_points_short, np.array(ship_traj_short, dtype=object), ship_mmsi_short, matched_obs_idx_list)
        self.add_new_traj(obs_array_t, matched_obs_idx_list)
        end_time = time.time()
        print(f'MOT TC: {end_time - start_time:.2f} s', end=' ')
        self.crt_t += 1
        # if self.crt_t % self.merge_t_interval == 0:
        #     # 执行融合
        #     # pdb.set_trace()
        #     self.multi_source_merging_working()
        #     # pdb.set_trace()
        # # return self.ship_traj_dict_all
        return self.get_ship_traj_dict()

    def get_ship_traj_dict(self):
        ship_traj_long_dict_all = {}
        ship_traj_short_dict_all = {}
        for traj_id in list(self.ship_traj_dict_pred.keys()):
            traj = self.ship_traj_dict_pred[traj_id]['traj']
            # resource = self.ship_traj_dict_pred[traj_id]['resource']
            if len(traj) < self.in_F:
                ship_traj_short_dict_all[traj_id] = {}
                ship_traj_short_dict_all[traj_id]['traj'] = traj
                # ship_traj_short_dict_all[traj_id]['resource'] = resource
            else:
                ship_traj_long_dict_all[traj_id] = {}
                ship_traj_long_dict_all[traj_id]['traj'] = traj
                # ship_traj_long_dict_all[traj_id]['resource'] = resource
        return ship_traj_long_dict_all, self.ship_traj_finish_dict, ship_traj_short_dict_all, self.ship_traj_dict_pred

    def save_tracking_traj_dict(self, args):
        ship_traj_long_dict_all, ship_traj_finish_dict_all, ship_traj_short_dict_all, ship_traj_dict_all = self.get_ship_traj_dict()
        save_path = args.pred_result_dir
        os.makedirs(os.path.join(save_path, args.exp_name), exist_ok=True)

        ship_traj_list = []
        for k in list(ship_traj_dict_all.keys()):
            ship_traj = ship_traj_dict_all[k]['traj']
            # ship_resource = np.array(ship_traj_dict_all[k]['resource'])
            if len(ship_traj) < self.merge_t_segment:
                continue
            for i in range(len(ship_traj)):
                if type(ship_traj[i]) != type([]):
                    ship_traj[i] = ship_traj[i].tolist()
                ship_traj[i].append(k)
            ship_traj_list += ship_traj
        # for k in list(ship_traj_finish_dict_all.keys()):
        #     ship_traj = ship_traj_finish_dict_all[k]['traj']
        #     if len(ship_traj) < self.merge_t_segment:
        #         continue
        #     # pdb.set_trace()
        #     for i in range(len(ship_traj)):
        #         if type(ship_traj[i]) != type([]):
        #             ship_traj[i] = ship_traj[i].tolist()
        #         ship_traj[i].append(k)
        #     ship_traj_list += ship_traj
        ship_traj_list = np.array(ship_traj_list, dtype=object)
        # ship_traj_list = ship_traj_list[ship_traj_list[:, 0].argsort()]
        sorted_indices = np.lexsort((ship_traj_list[:, 1], ship_traj_list[:, 0]))
        ship_traj_list = ship_traj_list[sorted_indices]
        # pdb.set_trace()
        wf = open(os.path.join(save_path, args.exp_name, args.file_name), 'w')
        for idx in range(len(ship_traj_list)):
            # pdb.set_trace()
            # 1,2475,-134.27806091308594,11.594024658203125,0.003662109375,0.0005388259887695312,-1,-1,-1,-1
            t, y_b, x_r, y_t, x_l, s, c, source, satellite_id, k = list(ship_traj_list[idx])
            wf.write(str((t+1)) + ',' + str(k) + ',' + str(x_l) + ',' + str(y_t) + ',' + str(abs(y_t - y_b)) + ',' + str(abs(x_l - x_r)) + ',-1,-1,-1,-1' + '\n')
        wf.close()

    def save_gt_traj_dict(self, save_path, gt_array):
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
    # import pdb
    # pdb.set_trace()
    for i, batch in enumerate(dataloader):
        # torch.Size([64, 5, 21, 2, 4]) --> joints.shape  -> N=5
        # torch.Size([2, 4, 21, 2, 4]) --> joints.shape  -> N=5
        joints, masks, mmsi_list, padding_mask = batch
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
        pred_output_joints, refine_output_joints, association_map = inference(model, config, in_joints, padding_mask, out_len=out_F)
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
        # pdb.set_trace()
    # import pdb
    # pdb.set_trace()
    if pred_joints_list == []:
        # pdb.set_trace()
        return pred_joints_list, refine_joints_list
    pred_joints_list = np.concatenate(pred_joints_list, axis=0)
    refine_joints_list = np.concatenate(refine_joints_list, axis=0)
    # pdb.set_trace()
    return pred_joints_list, refine_joints_list


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking_ckpt", type=str, default='tracking/experiments/MSTAv2/checkpoints/best_val_checkpoint.pth.tar',  help="checkpoint path")
    parser.add_argument("--data_root", type=str, default='data',  help="data_root path")
    parser.add_argument("--exp_name", type=str, default='MSTAv2_ours_wo_asso',  help="data_root path")
    parser.add_argument("--file_name", type=str, default='area_0.txt',  help="data_root path")
    parser.add_argument("--obs", type=str, default='MSTAv2_obs',  help="obs path")
    parser.add_argument("--gt", type=str, default='MSTAv2_gt',  help="gt path")
    # parser.add_argument("--area_lon_lat", type=str, default='area_lon_lat.yaml',  help="area_lon_lat path")
    parser.add_argument("--pred_result_dir", type=str, default='outputs/MSTAv2',   help="result path")
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

def init_tracking_model(args):
    ################################
    # Initialize tracking model
    ################################
    print('load tracking model...')
    tracking_ckpt = torch.load(args.tracking_ckpt, map_location = torch.device('cuda:0'))
    tracking_config = tracking_ckpt['config']
    if torch.cuda.is_available():
        tracking_config["DEVICE"] = f"cuda:0"
        torch.cuda.manual_seed(0)
    else:
        tracking_config["DEVICE"] = "cpu"
    tracking_model = create_model(tracking_config)
    tracking_model.load_state_dict(tracking_ckpt['model'])
    return tracking_model, tracking_config


def plot_points(args, obs_array, id_f, t):
    os.makedirs(os.path.join(args.plot_dir, str(id_f)), exist_ok=True)
    pdb.set_trace()
    lat = obs_array[:, 3]
    lon = obs_array[:, 4]
    mmsi = obs_array[:, 1]
    unique_mmsi = np.unique(mmsi)
    colors = cm.rainbow(np.linspace(0, 1, len(unique_mmsi)))
    plt.figure(figsize=(8, 6))
    for i, ummsi in enumerate(unique_mmsi):
        mask = mmsi == ummsi
        plt.scatter(lon[mask], lat[mask], color=colors[i], label=f'MMSI {int(ummsi)}')

    plt.title('Scatter Plot of Points by MMSI')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.savefig(os.path.join(args.plot_dir, str(id_f), str(t) + '.png'), format='png', dpi=300)
    plt.close()







