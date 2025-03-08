import argparse
import pdb
import torch
import random
import numpy as np
import pandas as pd
from progress.bar import Bar
from torch.utils.data import DataLoader

# from dataset_jrdb import batch_process_coords, create_dataset, collate_batch
# from model_ship import create_model
# from utils.utils import create_logger

from tracking.dataset_xysc import collate_batch, batch_process_coords, get_datasets, create_dataset
from tracking.model_ship import create_model, hungarian_matching
from tracking.utils.utils import create_logger, load_default_config, load_config, AverageMeter
from tracking.utils.metrics import MSE_LOSS, MSE_LOSS_XYSC, ASSO_LOSS, CONTRASTIVE_LOSS

def inference(model, config, input_joints, padding_mask, out_len=14):
    model.eval()

    with torch.no_grad():
        pred_joints, out_asso = model(input_joints, padding_mask)

    pred_output_joints = pred_joints[:,-out_len:]
    pred_refine_joints = pred_joints[:,:pred_joints.shape[1]-out_len]

    return pred_output_joints, pred_refine_joints, out_asso


def evaluate(model, modality_selection, dataloader, bs, config, logger, return_all=False, bar_prefix="", per_joint=False, show_avg=False):
    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    bar = Bar(f"EVAL ADE_FDE", fill="#", max=len(dataloader))

    batch_size = bs
    batch_id = 0

    ade_pred = 0
    ade_refine = 0
    fde_pred = 0
    fde_refine = 0
    asso_acc = 0
    asso_sen = 0
    asso_pre = 0
    asso_f1 = 0

    ade_pred_batch = 0
    ade_refine_batch = 0
    fde_pred_batch = 0
    fde_refine_batch = 0
    asso_acc_batch = 0
    asso_sen_batch = 0
    asso_pre_batch = 0
    asso_f1_batch = 0

    gt_pred_list = []
    first_x = 0
    first_y = 0
    add_item_x = 0
    add_item_y = 0
    add_idx = 0
    sample_number = 0
    for i, batch in enumerate(dataloader):
        # torch.Size([64, 5, 21, 2, 4]) --> joints.shape  -> N=5
        joints, masks, mmsi_list, padding_mask = batch
        # print('i, joints.shape', i, joints.shape)
        padding_mask = padding_mask.to(config["DEVICE"])
        # torch.Size([64, 9, 10, 4]) --> in_joints.shape
        in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(joints, masks, padding_mask, config, modality_selection)

        pred_output_joints, pred_refine_joints, out_asso = inference(model, config, in_joints, padding_mask, out_len=out_F)

        out_joints = out_joints.cpu()  # torch.Size([64, 12, 10, 4])  10 == 5 * 2
        in_joints = in_joints.cpu()

        pred_output_joints = pred_output_joints.cpu().reshape(out_joints.size(0), out_F, 1, 2)  # torch.Size([64, 12, 1, 2])
        pred_refine_joints = pred_refine_joints.cpu().reshape(in_joints.size(0), in_F, 1, 2)  # torch.Size([64, 9, 1, 2])

        pred_xy = pred_refine_joints[:,:,0,:2]
        gt_xy = in_joints[:,:,0,:2]
        norm = torch.norm(pred_xy - gt_xy, p=2, dim=-1)
        mean_K = torch.mean(norm, dim=-1)
        mean_B = torch.mean(mean_K)
        ade_refine_batch += mean_B * in_joints.shape[0]
        pred_xy = pred_refine_joints[:,-1,0,:2]
        gt_xy = in_joints[:,-1,0,:2]
        norm = torch.norm(pred_xy - gt_xy, p=2, dim=-1)
        mean_K = torch.mean(norm, dim=-1)
        mean_B = torch.mean(mean_K)
        fde_refine_batch += mean_B * in_joints.shape[0]
        sample_number += out_joints.shape[0]
        # import pdb
        # pdb.set_trace()

        for k in range(len(out_joints)):
            add_idx += 1
            # pdb.set_trace()
            first_x = in_joints[0, 0, 0, 0].detach().cpu().numpy()
            first_y = in_joints[0, 0, 0, 1].detach().cpu().numpy()
            person_out_joints = out_joints[k,:,:3]
            person_pred_joints = pred_output_joints[k,:,:1]
            gt_xy = person_out_joints[:,0,:2]
            pred_xy = person_pred_joints[:,0,:2]
            sum_ade = 0
            for t in range(out_F):
                d1 = (gt_xy[t,0].detach().cpu().numpy() - pred_xy[t,0].detach().cpu().numpy())
                d2 = (gt_xy[t,1].detach().cpu().numpy() - pred_xy[t,1].detach().cpu().numpy())
                dist_ade = [d1,d2]
                sum_ade += np.linalg.norm(dist_ade)
            gt_xy_numpy_0 = gt_xy[0].detach().cpu().numpy()
            pred_xy_numpy_0 = pred_xy[0].detach().cpu().numpy()
            gt_pred_list.append([gt_xy_numpy_0[0] - first_x + add_item_x, gt_xy_numpy_0[1] - first_y + add_item_y,
                                 pred_xy_numpy_0[0] - first_x + add_item_x, pred_xy_numpy_0[1] - first_y + add_item_y])
            add_item_x = add_item_x + in_joints[k, 1, 0, 0].detach().cpu().numpy() - first_x
            add_item_y = add_item_y + in_joints[k, 1, 0, 1].detach().cpu().numpy() - first_y
            sum_ade /= out_F
            ade_pred_batch += sum_ade
            d1 = (gt_xy[-1,0].detach().cpu().numpy() - pred_xy[-1,0].detach().cpu().numpy())
            d2 = (gt_xy[-1,1].detach().cpu().numpy() - pred_xy[-1,1].detach().cpu().numpy())
            dist_fde = [d1,d2]
            scene_fde = np.linalg.norm(dist_fde)
            fde_pred_batch += scene_fde

        N = len(mmsi_list)
        mmsi_matrix = torch.zeros((N, N), dtype=torch.float32).to(out_asso.device)
        for i in range(N):
            for j in range(N):
                mmsi_matrix[i][j] = 1 if mmsi_list[i] == mmsi_list[j] else 0
        gt_asso_map = mmsi_matrix
        norms = torch.norm(out_asso, p=2, dim=1, keepdim=True)  # 计算每行的 L2 范数
        normalized_matrix = out_asso / norms  # 归一化
        association_map = torch.mm(normalized_matrix, normalized_matrix.t())
        association_map[association_map >= 0.5] = 1
        association_map[association_map < 0.5] = 0
        equal_mask = association_map == gt_asso_map
        TP = torch.sum((association_map == 1) & equal_mask)  # 预测为1，且与gt_asso_map相等
        FP = torch.sum((association_map == 1) & ~equal_mask) # 预测为1，但与gt_asso_map不相等
        TN = torch.sum((association_map == 0) & equal_mask)  # 预测为0，且与gt_asso_map相等
        FN = torch.sum((association_map == 0) & ~equal_mask) # 预测为0，但与gt_asso_map不相等
        # print(f"TP: {TP}, FP: {FP} \nTN: {TN}, FN: {FN}")
        ASSO_ACC = (TP + TN) / (TP + TN + FP + FN)
        ASSO_SEN = TP / (TP + FN + 1e-10)
        ASSO_PRE = TP / (TP + FP + 1e-10)
        asso_acc_batch += ASSO_ACC
        asso_sen_batch += ASSO_SEN
        asso_pre_batch += ASSO_PRE
        asso_f1_batch += 2 * (ASSO_PRE * ASSO_SEN) / (ASSO_PRE + ASSO_SEN)
        batch_id+=1
        # break
    gt_pred_list = pd.DataFrame(gt_pred_list)
    gt_pred_list.to_csv('gt_pred_list.csv')
    ade_pred = ade_pred_batch/sample_number
    fde_pred = fde_pred_batch/sample_number
    ade_refine = ade_refine_batch/sample_number
    fde_refine = fde_refine_batch/sample_number
    asso_acc = asso_acc_batch / batch_id
    asso_sen = asso_sen_batch / batch_id
    asso_pre = asso_pre_batch / batch_id
    asso_f1 = asso_f1_batch / batch_id
    # pdb.set_trace()
    return ade_pred, fde_pred, ade_refine, fde_refine, asso_acc, asso_sen, asso_pre, asso_f1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,  help="checkpoint path")
    parser.add_argument("--split", type=str, default="train", help="Split to use. one of [train, test, valid]")
    parser.add_argument("--metric", type=str, default="vim", help="Evaluation metric. One of (vim, mpjpe)")
    parser.add_argument("--modality", type=str, default="traj+all", help="available modality combination from['traj','traj+2dbox']")

    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    ################################
    # Load checkpoint
    ################################

    logger = create_logger('')
    print(f'Loading checkpoint from {args.ckpt}')
    logger.info(f'Loading checkpoint from {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location = torch.device('cpu'))
    config = ckpt['config']

    if torch.cuda.is_available():
        config["DEVICE"] = "cuda:1"
        torch.cuda.manual_seed(0)
        print('use device cuda:1')
    else:
        config["DEVICE"] = "cpu"
        print('use device cpu')
    # # 修改batch size为1，也就是按时序执行
    # config['TRAIN']['batch_size'] = 1
    config['mode'] = 'eval'


    logger.info("Initializing with config:")
    logger.info(config)

    ################################
    # Initialize model
    ################################

    model = create_model(config)
    model.load_state_dict(ckpt['model'])
    ################################
    # Load data
    ################################

    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    # assert in_F == 9
    # assert out_F == 12

    name = config['DATA']['train_datasets']

    dataset = create_dataset(name[0], split=args.split, track_size=(in_F+out_F), track_cutoff=in_F)



    bs = config['TRAIN']['batch_size']
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=config['TRAIN']['num_workers'], shuffle=False, collate_fn=collate_batch)
    ade_pred, fde_pred, ade_refine, fde_refine, asso_acc, asso_sen, asso_pre, asso_f1 = evaluate(model, args.modality, dataloader, bs, config, logger, return_all=True)

    print('ade_pred \t fde_pred \t ade_refine \t fde_refine \t asso_acc \t asso_sen \t asso_pre \t asso_f1')
    print(f'{ade_pred:.5f} \t {fde_pred:.5f} \t {ade_refine:.5f} \t {fde_refine:.5f} \t {asso_acc.item():.5f} \t {asso_sen.item():.5f} \t {asso_pre.item():.5f} \t {asso_f1.item():.5f}')


