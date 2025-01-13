import os
import pdb
import json
import random
import numpy as np
import argparse
from scipy.interpolate import interp1d


def mask_and_interp_points(data_array_tmp, min_frame_id, max_frame_id, mode=0):
    # data_list.append([int(t), str(mmsi), str(label2), float(length), float(width),
    #                               float(lat_bottle), float(lon_right), float(lat_top), float(lon_left),
    #                               float(s), float(c), str(source), str(satellite_id)])
    # 拆分time, x, y, sog, cog
    # pdb.set_trace()
    times, indices = np.unique(data_array_tmp[:, 0], return_index=True)
    lat_bottle_coords = data_array_tmp[:, 5][indices]
    lon_right_coords = data_array_tmp[:, 6][indices]
    lat_top_coords = data_array_tmp[:, 7][indices]
    lon_left_coords = data_array_tmp[:, 8][indices]
    sog = data_array_tmp[:, 9][indices]
    cog = data_array_tmp[:, 10][indices]
    # 创建插值函数
    lat_bottle_interp_func = interp1d(times, lat_bottle_coords, kind='linear', fill_value="extrapolate")
    lon_right_interp_func = interp1d(times, lon_right_coords, kind='linear', fill_value="extrapolate")
    lat_top_interp_func = interp1d(times, lat_top_coords, kind='linear', fill_value="extrapolate")
    lon_left_interp_func = interp1d(times, lon_left_coords, kind='linear', fill_value="extrapolate")
    sog_interp_func = interp1d(times, sog, kind='linear', fill_value="extrapolate")
    cog_interp_func = interp1d(times, cog, kind='linear', fill_value="extrapolate")
    # import pdb
    # pdb.set_trace()

    missing_time_range = np.setdiff1d(np.arange(min_frame_id, max_frame_id + 1, 1), times)
    # pdb.set_trace()
    if mode == 0:
        # 用全0来填充
        missing_lat_bottle = np.array(list(0 for i in range(missing_time_range.shape[0])))
        missing_lon_right = np.array(list(0 for i in range(missing_time_range.shape[0])))
        missing_lat_top = np.array(list(0 for i in range(missing_time_range.shape[0])))
        missing_lon_left = np.array(list(0 for i in range(missing_time_range.shape[0])))
        missing_sog = np.array(list(0 for i in range(missing_time_range.shape[0])))
        missing_cog = np.array(list(0 for i in range(missing_time_range.shape[0])))
    elif mode == 1:
        # 对缺失时间点进行插值
        # pdb.set_trace()
        missing_lat_bottle = lat_bottle_interp_func(missing_time_range)
        missing_lon_right = lon_right_interp_func(missing_time_range)
        missing_lat_top = lat_top_interp_func(missing_time_range)
        missing_lon_left = lon_left_interp_func(missing_time_range)
        missing_sog = sog_interp_func(missing_time_range)
        missing_cog = cog_interp_func(missing_time_range)
    # data_list.append([int(f), str(mmsi), str(label1), str(label2), float(lat), float(lon), float(s), float(c), str(behavior), str(source), str(satellite_id)])
    # pdb.set_trace()
    missing_MMSI = np.array(list(data_array_tmp[0, 1] for i in range(missing_time_range.shape[0])))
    missing_ship_type = np.array(list(data_array_tmp[0, 2] for i in range(missing_time_range.shape[0])))
    missing_length = np.array(list(data_array_tmp[0, 3] for i in range(missing_time_range.shape[0])))
    missing_width = np.array(list(data_array_tmp[0, 4] for i in range(missing_time_range.shape[0])))
    missing_source = np.array(list('elec' for i in range(missing_time_range.shape[0])))
    missing_satellite_id = np.array(list(data_array_tmp[0, -1] for i in range(missing_time_range.shape[0])))
    # 将插值结果合并到缺失点
    # data_list.append([int(t), str(mmsi), str(label2), float(length), float(width),
    #                               float(lat_bottle), float(lon_right), float(lat_top), float(lon_left),
    #                               float(s), float(c), str(source), str(satellite_id)])
    missing_points = np.column_stack((missing_time_range, missing_MMSI, missing_ship_type, missing_length, missing_width,
                                      missing_lat_bottle, missing_lon_right, missing_lat_top, missing_lon_left,
                                      missing_sog, missing_cog, missing_source, missing_satellite_id))
    # import pdb
    # pdb.set_trace()

    # 将插值后的点与原始点结合
    all_points = np.vstack((data_array_tmp, missing_points))

    # 按时间排序
    all_points = all_points[np.argsort(all_points[:, 1])]
    return all_points


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='data',  help="data_root path")
    parser.add_argument("--dataset_name", type=str, default='MSTAv2',  help="data_root path")
    parser.add_argument("--time_axio", type=int, default=30,  help="result path")
    parser.add_argument("--in_F", type=int, default=30,  help="the input window length")
    parser.add_argument("--out_F", type=int, default=5,  help="the output window length")

    args = parser.parse_args()
    return args


args = init_config()
in_F = args.in_F
out_F = args.out_F
time_axio = args.time_axio
txt_data_root = f'{args.data_root}/{args.dataset_name}_obs'
gt_txt_data_root = f'{args.data_root}/{args.dataset_name}_gt'
split_list = ['train', 'test', 'val']
json_data_root = f'{args.data_root}/{args.dataset_name}'
print(args.dataset_name)
for split in split_list:
    print(split)
    path_split = os.path.join(txt_data_root, split)
    gt_path_split = os.path.join(gt_txt_data_root, split)
    write_path_split = os.path.join(json_data_root, split)
    os.makedirs(write_path_split, exist_ok=True)
    area_num = 0
    source_num_list = [0, 0]
    for file_name in os.listdir(path_split):
        area_num += 1
        if source_num_list[0] >= 100000 and split != 'test':
            break
        # file_name = 'area_1272.txt'
        print(file_name)
        if file_name.split('.')[-1] != 'txt':
            continue
        # 读取观测txt文件
        file_path = os.path.join(path_split, file_name)
        if os.path.getsize(file_path) / (1024 * 1024) > 10:
            continue
        data_list = []
        with open(file_path, 'r') as rf:
            for line_data in rf.readlines():
                # 1 108954000 100000018 400 60 11.593485512055697 -134.27439260140977 11.59402449876069 -134.27806068190026 1.505472196655527 47.77819766198284 elec satellite_3092764353
                t, mmsi, label2, length, width, lat_bottle, lon_right, lat_top, lon_left, s, c, source, satellite_id = line_data.strip('\n').split(' ')
                # time_list = t.split('T')[1].split(':')
                # f = int(int(time_list[0]) * 3600 + int(time_list[1]) * 60 + int(time_list[2]))
                # f = int(f / time_axio)
                if args.dataset_name == 'MTAD':
                    mmsi = mmsi[:-2]
                data_list.append([int(t), int(mmsi), str(label2), float(length), float(width),
                                  float(lat_bottle), float(lon_right), float(lat_top), float(lon_left),
                                  float(s), float(c), str(source), str(satellite_id)])
        data_array = np.array(data_list, dtype=object)
        if data_array.shape[0] == 0:
            continue
        data_array[:, 0] -= np.min(data_array[:, 0])

        # data_array = data_array[data_array[:, -2] == 'radar']
        # 读取GT.txt文件
        gt_file_path = os.path.join(gt_path_split, file_name)
        gt_data_list = []
        with open(gt_file_path, 'r') as rf:
            for line_data in rf.readlines():
                # TODO 去掉behavior
                # p000000001 108954000 100000004 100000018 2024-10-10T06:20:15 11.585119320000002 -134.3025129 0.0 44.91186941723307 Remain
                poing_id, mmsi, label1, label2, t, lat, lon, s, c, behavior = line_data.strip('\n').split(' ')
                if args.dataset_name == 'MSTAv2':
                    time_list = t.split('T')[1].split(':')
                    f = int(int(time_list[0]) * 3600 + int(time_list[1]) * 60 + int(time_list[2]))
                    t = int(f / time_axio)
                if args.dataset_name == 'MTAD':
                    mmsi = mmsi[:-2]
                gt_data_list.append([int(t), int(mmsi), str(label1), str(label2), float(lat), float(lon), float(s), float(c), str(behavior)])
        gt_data_array = np.array(gt_data_list, dtype=object)
        if gt_data_array.shape[0] == 0:
            continue
        gt_data_array[:, 0] -= np.min(gt_data_array[:, 0])
        # pdb.set_trace()
        # 将每个智能体的轨迹切分scene
        write_file_path = os.path.join(write_path_split, file_name.split('.')[0] + '.ndjson')
        # # 每个场景的片段数量随机化
        # segment_num = random.randint(10, 30)
        # 每个场景的片段数量固定为30
        # segment_num = 30
        with open(write_file_path, 'w') as wf:
            idx = 0
            ped_list = np.unique(gt_data_array[:, 1])
            for ped in ped_list:
                # 假设gt每一个目标都有960个点
                data_array_ped_gt = gt_data_array[gt_data_array[:, 1] == ped]
                # pdb.set_trace()
                min_frame_id = np.min(data_array_ped_gt[:, 0].astype(np.int32)).item()
                max_frame_id = np.max(data_array_ped_gt[:, 0].astype(np.int32)).item()
                segment_split = 100

                data_array_ped = data_array[data_array[:, 1] == ped].astype(object)
                # 观测点
                if data_array_ped.shape[0] == 0 or data_array_ped_gt.shape[0] == 0:
                    continue
                data_array_ped[:, 0] = data_array_ped[:, 0].astype(np.int32)
                if data_array_ped.shape[0] <= 35 or max_frame_id - min_frame_id <= 35:
                    continue
                # pdb.set_trace()
                data_array_ped = mask_and_interp_points(data_array_ped, min_frame_id, max_frame_id, mode=1)
                sorted_indices = np.argsort(data_array_ped[:, 0])
                data_array_ped = data_array_ped[sorted_indices]
                # 依次添加场景，及其对应的GT和OBS
                for frame_id in range(int(min_frame_id), int(max_frame_id) - out_F - in_F, segment_split):
                    # 统计[frame_id, frame_id + out_F + in_F]之间有多少观测
                    # min_obs_len = 1e9
                    # for t in range(frame_id, frame_id + out_F + in_F):
                    #     data_array_ped_t = data_array_ped[data_array_ped[:, 0] == t]
                    #     if min_obs_len > data_array_ped_t.shape[0]:
                    #         min_obs_len = data_array_ped_t.shape[0]
                    max_obs_len = 0
                    for t in range(frame_id, frame_id + out_F + in_F):
                        data_array_ped_t = data_array_ped[data_array_ped[:, 0] == t]
                        if max_obs_len < data_array_ped_t.shape[0]:
                            max_obs_len = data_array_ped_t.shape[0]
                    # pdb.set_trace()
                    # 添加 min_obs_len 个场景
                    for obs_id in range(max_obs_len):

                        # data_list.append([int(t), str(mmsi), str(label2), float(length), float(width),
                        #           float(lat_bottle), float(lon_right), float(lat_top), float(lon_left),
                        #           float(s), float(c), str(source), str(satellite_id)])

                        scene_flag = True
                        for t in range(frame_id, frame_id + out_F + in_F):
                            data_array_ped_t = data_array_ped[data_array_ped[:, 0] == t]
                            if data_array_ped_t.shape[0] == 0:
                                scene_flag = False
                                break
                            if data_array_ped_gt[data_array_ped_gt[:, 0] == int(t)].shape[0] == 0:
                                scene_flag = False
                                break
                        if scene_flag:
                            for t in range(frame_id, frame_id + out_F + in_F):
                                data_array_ped_t = data_array_ped[data_array_ped[:, 0] == t]
                                # pdb.set_trace()
                                if data_array_ped_t.shape[0] <= obs_id:
                                    data_array_ped_t_obs = data_array_ped_t[0]
                                else:
                                    data_array_ped_t_obs = data_array_ped_t[obs_id]
                                # pdb.set_trace()
                                if data_array_ped_t_obs[-2] == 'elec':
                                    source_num_list[0] += 1
                                else:
                                    source_num_list[1] += 1
                                # pdb.set_trace()
                                # print(data_array_ped_gt[data_array_ped_gt[:, 0] == int(t)].shape)
                                if data_array_ped_gt[data_array_ped_gt[:, 0] == int(t)].shape[0] == 0:
                                    continue
                                data_array_gt_t = data_array_ped_gt[data_array_ped_gt[:, 0] == int(t)][0]
                                track = {'track':{
                                    'f': int(t) - min_frame_id,
                                    'o': obs_id,
                                    'p': str(ped),
                                    'ship_type': str(data_array_ped_t_obs[2]),
                                    'length': float(data_array_ped_t_obs[3]),
                                    'width': float(data_array_ped_t_obs[4]),
                                    'lat_bottle': float(data_array_ped_t_obs[5]),
                                    'lon_right': float(data_array_ped_t_obs[6]),
                                    'lat_top': float(data_array_ped_t_obs[7]),
                                    'lon_left': float(data_array_ped_t_obs[8]),
                                    'sog': float(data_array_ped_t_obs[9]),
                                    'cog': float(data_array_ped_t_obs[10]),
                                    'source': str(data_array_ped_t_obs[11]),
                                }}
                                json.dump(track, wf)
                                wf.write('\n')

                                # gt_data_list.append([int(f), str(mmsi), str(label1), str(label2), float(lat), float(lon), float(s), float(c), str(behavior)])
                                track = {'gt_track':{
                                    'f': int(t) - min_frame_id,
                                    'o': obs_id,
                                    'p': str(ped),
                                    'ship_type': str(data_array_gt_t[3]),
                                    'lat_top': float(data_array_gt_t[4]),
                                    'lon_left': float(data_array_gt_t[5]),
                                    'sog': float(data_array_gt_t[6]),
                                    'cog': float(data_array_gt_t[7]),
                                }}
                                json.dump(track, wf)
                                wf.write('\n')
                            scene = {'scene':{
                                'id': idx,
                                'o': obs_id,
                                'p': str(ped),
                                's': frame_id - min_frame_id,
                                'e': frame_id + in_F + out_F - 1 - min_frame_id,
                            }}
                            idx += 1
                            json.dump(scene, wf)
                            wf.write('\n')

                # break
        print(source_num_list)
        # break





