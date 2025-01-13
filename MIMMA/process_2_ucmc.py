#  源文件在1013:/workspace/MOT/st-ship/social-transmotion-multi-source/data
import os
import pdb
import json
import random
import numpy as np
import argparse
from scipy.interpolate import interp1d


def mask_and_interp_points(data_array_tmp, min_frame_id, max_frame_id, mode=0):
    # 拆分time, x, y, sog, cog
    times = data_array_tmp[:, 0]
    x_coords = data_array_tmp[:, 4]
    y_coords = data_array_tmp[:, 5]
    sog = data_array_tmp[:, 6]
    cog = data_array_tmp[:, 7]
    # 创建插值函数
    x_interp_func = interp1d(times, x_coords, kind='linear', fill_value="extrapolate")
    y_interp_func = interp1d(times, y_coords, kind='linear', fill_value="extrapolate")
    sog_interp_func = interp1d(times, sog, kind='linear', fill_value="extrapolate")
    cog_interp_func = interp1d(times, cog, kind='linear', fill_value="extrapolate")
    # import pdb
    # pdb.set_trace()

    missing_time_range = np.setdiff1d(np.arange(min_frame_id, max_frame_id + 1, 1), times)
    # pdb.set_trace()
    if mode == 0:
        # 用全0来填充
        missing_x = np.array(list(0 for i in range(missing_time_range.shape[0])))
        missing_y = np.array(list(0 for i in range(missing_time_range.shape[0])))
        missing_sog = np.array(list(0 for i in range(missing_time_range.shape[0])))
        missing_cog = np.array(list(0 for i in range(missing_time_range.shape[0])))
    elif mode == 1:
        # 对缺失时间点进行插值
        missing_x = x_interp_func(missing_time_range)
        missing_y = y_interp_func(missing_time_range)
        missing_sog = sog_interp_func(missing_time_range)
        missing_cog = cog_interp_func(missing_time_range)
    # data_list.append([int(f), int(mmsi), str(label1), str(label2), float(lat), float(lon), float(s), float(c), str(behavior), str(source), str(satellite_id)])
    missing_MMSI = np.array(list(data_array_tmp[0, 1] for i in range(missing_time_range.shape[0])))
    missing_label1 = np.array(list(data_array_tmp[0, 2] for i in range(missing_time_range.shape[0])))
    missing_label2 = np.array(list(data_array_tmp[0, 3] for i in range(missing_time_range.shape[0])))
    missing_behavior = np.array(list(data_array_tmp[0, 8] for i in range(missing_time_range.shape[0])))
    missing_source = np.array(list(data_array_tmp[0, 9] for i in range(missing_time_range.shape[0])))
    missing_satellite_id = np.array(list(data_array_tmp[0, 10] for i in range(missing_time_range.shape[0])))
    # 将插值结果合并到缺失点
    missing_points = np.column_stack((missing_time_range, missing_MMSI, missing_label1, missing_label2, missing_x,
                                      missing_y, missing_sog, missing_cog, missing_behavior, missing_source, missing_satellite_id))
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

    args = parser.parse_args()
    return args


args = init_config()
time_axio = args.time_axio
txt_data_root = f'{args.data_root}/{args.dataset_name}_obs'
gt_txt_data_root = f'{args.data_root}/{args.dataset_name}_gt'
split_list = ['train', 'test']
save_root = f'{args.data_root}/{args.dataset_name}_UCMC/det_results/{args.dataset_name}_UCMC'
cam_para_save_root = f'{args.data_root}/{args.dataset_name}_UCMC/cam_para/{args.dataset_name}_UCMC'
for split in split_list:
    print(split)
    path_split = os.path.join(txt_data_root, split)
    gt_path_split = os.path.join(gt_txt_data_root, split)
    # write_path_split = os.path.join(save_data_root, split)
#     os.makedirs(write_path_split, exist_ok=True)
    for file_name in os.listdir(path_split):

        if file_name.split('.')[-1] != 'txt':
            continue
        # 读取txt文件
        file_path = os.path.join(path_split, file_name)
        if os.path.getsize(file_path) >= 5 * 1024 * 1024:
            continue
        print(file_name)
        data_list = []
        with open(file_path, 'r') as rf:
            for line_data in rf.readlines():
                # poing_id, mmsi, label1, label2, t, lat, lon, s, c, behavior, source, satellite_id = line_data.strip('\n').split(' ')
                # time_list = t.split('T')[1].split(':')
                # f = int(int(time_list[0]) * 3600 + int(time_list[1]) * 60 + int(time_list[2]))
                # f = int(f / time_axio)
                # data_list.append([int(mmsi), str(source), int(f), float(lat), float(lon), float(s), float(c)])
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
        measure = np.array(data_list, dtype=object)
        if measure.shape[0] == 0:
            continue
        measure[:, 0] -= np.min(measure[:, 0])
        # measure = data_array[data_array[:, -2] == 'radar']
        # 读取GT.txt文件
        gt_file_path = os.path.join(gt_path_split, file_name)
        gt_data_list = []
        with open(gt_file_path, 'r') as rf:
            for line_data in rf.readlines():
                # poing_id, mmsi, label1, label2, t, lat, lon, s, c, behavior = line_data.strip('\n').split(' ')
                # time_list = t.split('T')[1].split(':')
                # f = int(int(time_list[0]) * 3600 + int(time_list[1]) * 60 + int(time_list[2]))
                # f = int(f / time_axio)
                # gt_data_list.append([int(mmsi), int(f), float(lat), float(lon), float(s), float(c)])
                # TODO 去掉behavior
                # p000000001 108954000 100000004 100000018 2024-10-10T06:20:15 11.585119320000002 -134.3025129 0.0 44.91186941723307 Remain
                poing_id, mmsi, label1, label2, t, lat, lon, s, c, behavior = line_data.strip('\n').split(' ')
                # time_list = t.split('T')[1].split(':')
                # f = int(int(time_list[0]) * 3600 + int(time_list[1]) * 60 + int(time_list[2]))
                # f = int(f / time_axio)
                if args.dataset_name == 'MTAD':
                    mmsi = mmsi[:-2]
                gt_data_list.append([int(t), int(mmsi), str(label1), str(label2), float(lat), float(lon), float(s), float(c), str(behavior)])
        gt = np.array(gt_data_list, dtype=object)
        gt[:, 0] -= np.min(gt[:, 0])

        gt_save_dir = os.path.join(save_root, split, 'gt', file_name.split('.')[0])
        os.makedirs(os.path.join(gt_save_dir, 'gt'), exist_ok=True)
        seqinfo_ini_wf = open(os.path.join(gt_save_dir, 'seqinfo.ini'), 'w')
        gt_txt_wf = open(os.path.join(gt_save_dir, 'gt', 'gt.txt'), 'w')
        # # 计算场景大小
        # min_lat, max_lat = np.min(gt[:, 2]), np.max(gt[:, 2])
        # min_lon, max_lon = np.min(gt[:, 3]), np.max(gt[:, 3])
        # 时间除以频率
        # gt[:, 1] = gt[:, 1] / frame_rate
        # 转为像素，分辨率为10米，则 1度 = 111000米 = 11100 个像素
        # gt[:, 2] = (gt[:, 2] - min_lat) * 111000 / 10
        # gt[:, 3] = (gt[:, 3] - min_lon) * 111000 / 10
        # gt[:, 2] = gt[:, 2] * 111000 / 10
        # gt[:, 3] = gt[:, 3] * 111000 / 10
        # 按照mmsi处理
        gt_mmsi_list = np.unique(gt[:, 1])
        for mmsi in gt_mmsi_list:
            gt_mmsi_sort = np.sort(gt[gt[:, 1] == mmsi], axis=0)
            # pdb.set_trace()  # 检查是否按时间顺序
            last_t = -1
            for i in range(gt_mmsi_sort.shape[0]):
                gt_mmsi_i = gt_mmsi_sort[i]
                # gt_data_list.append([int(f), int(mmsi), str(label1), str(label2), float(lat), float(lon), float(s), float(c), str(behavior)])
                t,_,_,_,lat,lon,sog,cog,_ = gt_mmsi_i
                if int(t) == last_t:
                    continue
                # 假设每艘船大小为：长300米，宽50米，且都是朝北的
                bb_left = lon
                bb_top = lat
                bb_width, bb_height = 0.003, 0.001  # TODO使用GT的BBOX长宽
                conf, clas, visibility = 1, 1, 1
                # pdb.set_trace()
                gt_txt_wf.write(str(int(t+1)) + ',' + str(mmsi) + ',' + str(bb_left) + ',' +
                                str(bb_top) + ',' + str(bb_width) + ',' + str(bb_height) + ',' +
                                str(conf) + ',' + str(clas) + ',' + str(visibility) + '\n')
                last_t = int(t)
        gt_txt_wf.close()
        # seqinfo_ini_wf
        name = file_name.split('.')[0]
        imgDir = 'img1'
        frameRate = 1
        seqLength = np.max(gt[:, 0])
        imWidth=1920
        imHeight=1080
        imExt='.jpg'
        seqinfo_ini_wf.write(str('[Sequence]') + '\n')
        seqinfo_ini_wf.write('name=' + str(name) + '\n')
        seqinfo_ini_wf.write('imgDir=' + str(imgDir) + '\n')
        seqinfo_ini_wf.write('frameRate=' + str(frameRate) + '\n')
        seqinfo_ini_wf.write('seqLength=' + str(int(seqLength + 1)) + '\n')
        seqinfo_ini_wf.write('imWidth=' + str(imWidth) + '\n')
        seqinfo_ini_wf.write('imHeight=' + str(imHeight) + '\n')
        seqinfo_ini_wf.write('imExt=' + str(imExt) + '\n')
        # 处理measure
        measure_save_dir = os.path.join(save_root, split)
        os.makedirs(os.path.join(measure_save_dir, 'measure'), exist_ok=True)
        measure_txt_wf = open(os.path.join(measure_save_dir, 'measure', file_name.split('.')[0] + '.txt'), 'w')
        # batch,source,time,lat,lon,vel,cou --> time,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1
        # measure = pd.read_csv(os.path.join(measure_path, file_name)).values
        # 转为像素，分辨率为10米，则 1度 = 111000米 = 11100 个像素
        # measure[:, 2] = measure[:, 2] / frame_rate
        # measure[:, 3] = (measure[:, 3] - min_lat) * 111000 / 10
        # measure[:, 4] = (measure[:, 4] - min_lon) * 111000 / 10
        # measure[:, 3] = measure[:, 3] * 111000 / 10
        # measure[:, 4] = measure[:, 4] * 111000 / 10
        # data_list.append([int(t), int(mmsi), str(label2), float(length), float(width),
        #                           float(lat_bottle), float(lon_right), float(lat_top), float(lon_left),
        #                           float(s), float(c), str(source), str(satellite_id)])
        # 按照time处理
        measure[:, 0] = measure[:, 0].astype(np.int32)
        measure_time_list = np.unique(measure[:, 0])
        for t in measure_time_list:
            measure_t = measure[measure[:, 0] == t]
            # pdb.set_trace()  # 检查是否随机顺序
            for i in range(measure_t.shape[0]):
                measure_t_i = measure_t[i]
                _,_,ship_type,length,width,lat_bottle,lon_right,lat_top,lon_left,sog,cog, source,_ = measure_t_i
                # 假设每艘船大小为：长300米，宽50米，且都是朝北的
                bb_left = lon_left
                bb_top = lat_top
                bb_width, bb_height = abs(lon_left - lon_right), abs(lat_top - lat_bottle)  # 改为占用一个像素点
                if source == 'elec':
                    conf = 0.6
                else:
                    conf = 0.8
                measure_txt_wf.write(str(int(t+1)) + ',' + str(int(i)) + ',' + str(bb_left) + ',' +
                                str(bb_top) + ',' + str(bb_width) + ',' + str(bb_height) + ',' +
                                str(conf) + ',-1,-1,-1' + '\n')
        measure_txt_wf.close()
        # cam_para
        os.makedirs(os.path.join(cam_para_save_root, split), exist_ok=True)
        cam_para_txt_wf = open(os.path.join(cam_para_save_root, split, file_name.split('.')[0] + '.txt'), 'w')
        cam_para_txt_wf.write('RotationMatrices\n1 0 0\n0 0 -1\n0 1 0\n\n')
        cam_para_txt_wf.write('TranslationVectors\n0 1310 6097\n\n')
        cam_para_txt_wf.write('IntrinsicMatrix\n1300 0 960\n0 1300 540\n0 0 1\n')
        cam_para_txt_wf.close()





