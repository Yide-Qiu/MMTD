import argparse
from datetime import datetime
import numpy as np
import os
import pdb
import random
import time
import torch

from progress.bar import Bar
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from tracking.dataset_xysc import collate_batch, batch_process_coords, get_datasets, create_dataset
from tracking.model_ship import create_model, hungarian_matching
from tracking.utils.utils import create_logger, load_default_config, load_config, AverageMeter
from tracking.utils.metrics import MSE_LOSS, MSE_LOSS_XYSC, ASSO_LOSS, CONTRASTIVE_LOSS

def evaluate_loss(model, dataloader, config):
    bar = Bar(f"EVAL", fill="#", max=len(dataloader))
    loss_pred_avg = AverageMeter()
    loss_recon_avg = AverageMeter()
    loss_asso_avg = AverageMeter()
    asso_acc_avg = []
    asso_sen_avg = []
    asso_cm = [0, 0, 0, 0]
    dataiter = iter(dataloader)

    model.eval()
    with torch.no_grad():
        for i in range(len(dataloader)):
            try:
                joints, masks, mmsi_list, padding_mask = next(dataiter)
            except StopIteration:
                break

            in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(joints, masks, padding_mask, config)
            padding_mask = padding_mask.to(config["DEVICE"])

            loss_pred, loss_recon, loss_asso, _, out_asso, asso_cm_batch = compute_loss(model, config, in_joints, out_joints, mmsi_list, in_masks, out_masks, padding_mask)
            loss_pred_avg.update(loss_pred.item(), len(in_joints))
            loss_recon_avg.update(loss_recon.item(), len(in_joints))
            loss_asso_avg.update(loss_asso.item(), len(in_joints))
            # 计算关联准确率
            # pdb.set_trace()
            N = len(mmsi_list)
            mmsi_matrix = torch.zeros((N, N), dtype=torch.float32).to(out_asso.device)
            for i in range(N):
                for j in range(N):
                    mmsi_matrix[i][j] = 1 if mmsi_list[i] == mmsi_list[j] else 0
            # labels = mmsi_matrix.view(out_asso.shape[0])
            # out_asso[out_asso >= 0.5] = 1
            # out_asso[out_asso < 0.5] = 0
            # TP = torch.sum((out_asso[:, 0] == 1) & (labels == 1))  # 预测为1，且与gt_asso_map相等
            # FP = torch.sum((out_asso[:, 0] == 1) & (labels == 0)) # 预测为1，但与gt_asso_map不相等
            # TN = torch.sum((out_asso[:, 0] == 0) & (labels == 0))  # 预测为0，且与gt_asso_map相等
            # FN = torch.sum((out_asso[:, 0] == 0) & (labels == 1)) # 预测为0，但与gt_asso_map不相等
            # asso_cm[0] += TP
            # asso_cm[1] += FP
            # asso_cm[2] += TN
            # asso_cm[3] += FN
            # # print(f"TP: {TP}, FP: {FP} \nTN: {TN}, FN: {FN}")
            # ASSO_ACC = torch.sum(out_asso[:, 0] == labels) / (N * N)
            # pdb.set_trace()
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
            asso_cm[0] += TP
            asso_cm[1] += FP
            asso_cm[2] += TN
            asso_cm[3] += FN
            # print(f"TP: {TP}, FP: {FP} \nTN: {TN}, FN: {FN}")
            ASSO_ACC = (TP + TN) / (TP + TN + FP + FN)
            ASSO_SEN = TP / (TP + FN + 1e-10)
            asso_acc_avg.append(ASSO_ACC)
            asso_sen_avg.append(ASSO_SEN)
            # pdb.set_trace()
            summary = [
                f"({i + 1}/{len(dataloader)})",
                f"LOSS PRED: {loss_pred_avg.avg:.4f}",
                f"LOSS RECON: {loss_recon_avg.avg:.4f}",
                f"LOSS ASSO: {loss_asso_avg.avg:.4f}"
            ]

            bar.suffix = " | ".join(summary)
            bar.next()

        bar.finish()
    asso_acc_avg = torch.mean(torch.tensor(asso_acc_avg, dtype=torch.float32))
    asso_sen_avg = torch.mean(torch.tensor(asso_sen_avg, dtype=torch.float32))
    print(f"TP: {asso_cm[0]}, FP: {asso_cm[1]} \nTN: {asso_cm[2]}, FN: {asso_cm[3]}")

    return loss_pred_avg.avg, loss_recon_avg.avg, loss_asso_avg.avg, asso_acc_avg, asso_sen_avg

def compute_loss(model, config, in_joints, out_joints, mmsi_list, in_masks, out_masks, padding_mask, epoch=None, mode='val', loss_last=True, optimizer=None):

    _, in_F, _, _ = in_joints.shape

    metamask = (mode == 'train')
    # pdb.set_trace()

    pred_joints, out_asso = model(in_joints, padding_mask, metamask=metamask)
    # pdb.set_trace()
    loss_pred = MSE_LOSS(pred_joints[:,in_F:], out_joints, out_masks)
    loss_recon = MSE_LOSS(pred_joints[:,:in_F], in_joints, out_masks)
    # loss_asso = ASSO_LOSS(out_asso, mmsi_list)
    loss_asso = CONTRASTIVE_LOSS(out_asso, mmsi_list)
    N = len(mmsi_list)
    mmsi_matrix = torch.zeros((N, N), dtype=torch.float32).to(out_asso.device)
    for i in range(N):
        for j in range(N):
            mmsi_matrix[i][j] = 1 if mmsi_list[i] == mmsi_list[j] else 0
    # labels = mmsi_matrix.view(out_asso.shape[0])
    # out_asso[out_asso >= 0.5] = 1
    # out_asso[out_asso < 0.5] = 0
    # # equal_mask = labels == out_asso
    # TP = torch.sum((out_asso[:, 0] == 1) & (labels == 1))  # 预测为1，且与gt_asso_map相等
    # FP = torch.sum((out_asso[:, 0] == 1) & (labels == 0)) # 预测为1，但与gt_asso_map不相等
    # TN = torch.sum((out_asso[:, 0] == 0) & (labels == 0))  # 预测为0，且与gt_asso_map相等
    # FN = torch.sum((out_asso[:, 0] == 0) & (labels == 1)) # 预测为0，但与gt_asso_map不相等
    # asso_cm = [TP, FP, TN, FN]

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
    asso_cm = [TP, FP, TN, FN]
    # loss_asso = CONTRASTIVE_LOSS(out_asso, mmsi_list)
    # pdb.set_trace()

    return loss_pred, loss_recon, loss_asso, pred_joints, out_asso, asso_cm

def adjust_learning_rate(optimizer, epoch, config):
    """
    From: https://github.com/microsoft/MeshTransformer/
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs*2/3 = 100
    """
    # dct_multi_overfit_3dpw_allsize_multieval_noseg_rot_permute_id
    lr = config['TRAIN']['lr'] * (config['TRAIN']['lr_decay'] ** epoch) #  (0.1 ** (epoch // (config['TRAIN']['epochs']*4./5.)  ))
    if 'lr_drop' in config['TRAIN'] and config['TRAIN']['lr_drop']:
        lr = lr * (0.1 ** (epoch // (config['TRAIN']['epochs']*4./5.)  ))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    print('lr: ',lr)

def save_checkpoint(model, optimizer, epoch, config, filename, logger):
    logger.info(f'Saving checkpoint to {filename}.')
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'config': config
    }
    torch.save(ckpt, os.path.join(config['OUTPUT']['ckpt_dir'], filename))


def dataloader_for(dataset, config, **kwargs):
    return DataLoader(dataset,
                      batch_size=config['TRAIN']['batch_size'],
                      num_workers=config['TRAIN']['num_workers'],
                      collate_fn=collate_batch,
                      **kwargs)

def dataloader_for_val(dataset, config, **kwargs):
    return DataLoader(dataset,
                      batch_size=1,
                      num_workers=0,
                      collate_fn=collate_batch,
                      **kwargs)

def train(config, logger, experiment_name="", dataset_name=""):

    ################################
    # Load data
    ################################

    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    seq_num = config['TRAIN']['seq_num']
    dataset_train = ConcatDataset(get_datasets(config['DATA']['train_datasets'], config))
    dataloader_train = dataloader_for(dataset_train, config, shuffle=True, pin_memory=True)
    logger.info(f"Training on a total of {len(dataset_train)} annotations.")

    dataset_val = create_dataset(config['DATA']['train_datasets'][0], split="val", track_size=(in_F+out_F), track_cutoff=in_F, seq_num=seq_num)
    dataloader_val = dataloader_for(dataset_val, config, shuffle=True, pin_memory=True)
    logger.info(f"Evaluating on a total of {len(dataset_val)} annotations.")

    writer_name = experiment_name + "_" + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    writer_train = SummaryWriter(os.path.join(config["OUTPUT"]["runs_dir"], f"{writer_name}_TRAIN"))
    writer_valid =  SummaryWriter(os.path.join(config["OUTPUT"]["runs_dir"], f"{writer_name}_VALID"))
    config['mode'] = 'train'

    ################################
    # Create model, loss, optimizer
    ################################

    model = create_model(config)

    if config["MODEL"]["checkpoint"] != "":
        # logger.info(f"Loading checkpoint from {os.path.join(f'/data/ymh/MOT/codes/MSTA/tracking/experiments/{config["MODEL"]["checkpoint"]}/checkpoints', 'best_val_checkpoint.pth.tar')}")
        checkpoint = torch.load(os.path.join(f'/data/ymh/MOT/codes/MSTA/tracking/experiments/{config["MODEL"]["checkpoint"]}/checkpoints', 'best_val_checkpoint.pth.tar'))
        model.load_state_dict(checkpoint["model"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['TRAIN']['lr'])

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_parameters} parameters.")

    ################################
    # Begin Training
    ################################
    global_step = 0
    min_val_loss = 1e4
    max_acc_sen = 0

    for epoch in range(config["TRAIN"]["epochs"]):
        # val_loss_pred, val_loss_recon, val_loss_asso, asso_acc_avg, asso_sen_avg = evaluate_loss(model, dataloader_val, config)
        start_time = time.time()
        dataiter = iter(dataloader_train)

        timer = {"DATA": 0, "FORWARD": 0, "BACKWARD": 0}

        loss_pred_avg = AverageMeter()
        loss_recon_avg = AverageMeter()
        loss_asso_avg = AverageMeter()
        disc_loss_avg = AverageMeter()
        disc_acc_avg = AverageMeter()
        asso_cm = [0, 0, 0, 0]

        if config["TRAIN"]["optimizer"] == "adam":
            adjust_learning_rate(optimizer, epoch, config)

        train_steps =  len(dataloader_train)

        bar = Bar(f"TRAIN {epoch}/{config['TRAIN']['epochs'] - 1}", fill="#", max=train_steps)

        for i in range(train_steps):
            model.train()
            optimizer.zero_grad()

            ################################
            # Load a batch of data
            ################################
            start = time.time()

            try:
                joints, masks, mmsi_list, padding_mask = next(dataiter)

            except StopIteration:
                dataiter = iter(dataloader_train)
                joints, masks, mmsi_list, padding_mask = next(dataiter)

            # pdb.set_trace()  # loss_asso 不降反升，可能是数据集样本不均衡导致，需要查看数据
            in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(joints, masks, padding_mask, config, training=True)
            padding_mask = padding_mask.to(config["DEVICE"])

            timer["DATA"] = time.time() - start

            ################################
            # Forward Pass
            ################################
            start = time.time()
            # print('in_joints', in_joints.shape)
            # print('out_joints', out_joints.shape)
            # print('in_masks', in_masks.shape)
            # print('out_masks', out_masks.shape)
            # print('padding_mask', padding_mask.shape)
            # loss_s = 0
            # loss_c = 0
            loss_pred, loss_recon, loss_asso, pred_joints, out_asso, asso_cm_batch = compute_loss(model, config, in_joints, out_joints, mmsi_list, in_masks, out_masks, padding_mask, epoch=epoch, mode='train', optimizer=None)
            # pdb.set_trace()
            asso_cm[0] += asso_cm_batch[0]
            asso_cm[1] += asso_cm_batch[1]
            asso_cm[2] += asso_cm_batch[2]
            asso_cm[3] += asso_cm_batch[3]
            timer["FORWARD"] = time.time() - start

            ################################
            # Backward Pass + Optimization
            ################################
            start = time.time()
            # loss_asso = loss_asso * 1000
            # pdb.set_trace()
            loss_all = loss_pred + loss_asso
            # pdb.set_trace()
            loss_all.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["TRAIN"]["max_grad_norm"])
            optimizer.step()

            timer["BACKWARD"] = time.time() - start

            ################################
            # Logging
            ################################

            loss_pred_avg.update(loss_pred.item(), len(joints))
            loss_recon_avg.update(loss_recon.item(), len(in_joints))
            loss_asso_avg.update(loss_asso.item(), len(in_joints))

            summary = [
                f"{str(epoch).zfill(3)} ({i + 1}/{train_steps})",
                f"LOSS PRED: {loss_pred_avg.avg:.4f}",
                f"LOSS RECON: {loss_recon_avg.avg:.4f}",
                f"LOSS ASSO: {loss_asso_avg.avg:.4f}"
            ]


            for key, val in timer.items():
                 summary.append(f"{key}: {val:.2f}")

            bar.suffix = " | ".join(summary)
            bar.next()

            if cfg['dry_run']:
                break

        bar.finish()
        print(f"[Train]  TP: {asso_cm[0]}, FP: {asso_cm[1]} \n[Train]  TN: {asso_cm[2]}, FN: {asso_cm[3]}")

        ################################
        # Tensorboard logs
        ################################

        global_step += train_steps

        writer_train.add_scalar("loss", loss_pred_avg.avg + loss_recon_avg.avg + loss_asso_avg.avg, epoch)

        val_loss_pred, val_loss_recon, val_loss_asso, asso_acc_avg, asso_sen_avg = evaluate_loss(model, dataloader_val, config)
        writer_valid.add_scalar("loss", val_loss_pred + val_loss_recon + val_loss_asso, epoch)

        val_ade_pred = val_loss_pred/100
        val_ade_recon = val_loss_recon/100

        print('PRED: ', val_ade_pred, 'RECON: ', val_ade_recon, 'ASSO ACC: ', asso_acc_avg, 'ASSO SEN: ', asso_sen_avg)
        if max_acc_sen < asso_acc_avg + asso_sen_avg:

            max_acc_sen = asso_acc_avg + asso_sen_avg
            print('------------------------------BEST MODEL UPDATED------------------------------')
            print('Best PRED: ', val_ade_pred, 'Best RECON: ', val_ade_recon, 'Best ASSO ACC: ', asso_acc_avg, 'Best ASSO SEN: ', asso_sen_avg)
            save_checkpoint(model, optimizer, epoch, config, 'best_val'+'_checkpoint.pth.tar', logger)


        if cfg['dry_run']:
            break
        print('time for training: ', time.time()-start_time)
        print('epoch ', epoch, ' finished!')

    if not cfg['dry_run']:
        save_checkpoint(model, optimizer, epoch, config, 'checkpoint.pth.tar', logger)
    logger.info("All done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name. Otherwise will use timestamp")
    parser.add_argument("--cfg", type=str, default="", help="Config name. Otherwise will use default config")
    parser.add_argument('--dry-run', action='store_true', help="Run just one iteration")
    args = parser.parse_args()

    if args.cfg != "":
        cfg = load_config(args.cfg, exp_name=args.exp_name)
    else:
        cfg = load_default_config()

    cfg['dry_run'] = args.dry_run

    random.seed(cfg['SEED'])
    torch.manual_seed(cfg['SEED'])
    np.random.seed(cfg['SEED'])

    if torch.cuda.is_available():
        cfg["DEVICE"] = f"cuda:0"
        print('use device cuda:0')
    else:
        cfg["DEVICE"] = "cpu"
        print('use device cpu')

    dataset = cfg["DATA"]["train_datasets"]

    logger = create_logger(cfg["OUTPUT"]["log_dir"])
    logger.info("Initializing with config:")
    logger.info(cfg)

    train(cfg, logger, experiment_name=args.exp_name, dataset_name=dataset)






