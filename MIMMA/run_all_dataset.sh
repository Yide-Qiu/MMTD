# python train_msta.py --cfg tracking/configs/VISO.yaml --exp_name VISO
# # python train_msta_wo_asso_loss.py --cfg tracking/configs/VISO.yaml --exp_name VISO_wo_asso_loss
# python train_msta.py --cfg tracking/configs/SatSOT.yaml --exp_name SatSOT
# # python train_msta_wo_asso_loss.py --cfg tracking/configs/SatSOT.yaml --exp_name SatSOT_wo_asso_loss
# python train_msta.py --cfg tracking/configs/SAT_MTB.yaml --exp_name SAT_MTB
# # python train_msta_wo_asso_loss.py --cfg tracking/configs/SAT_MTB.yaml --exp_name SAT_MTB_wo_asso_loss
# python train_msta.py --cfg tracking/configs/MTAD.yaml --exp_name MTAD
# # python train_msta_wo_asso_loss.py --cfg tracking/configs/MTAD.yaml --exp_name MTAD_wo_asso_loss
# python train_msta.py --cfg tracking/configs/AccessAIS.yaml --exp_name AccessAIS
# # python train_msta_wo_asso_loss.py --cfg tracking/configs/AccessAIS.yaml --exp_name AccessAIS_wo_asso_loss
# python train_msta.py --cfg tracking/configs/OOTB.yaml --exp_name OOTB
# # python train_msta_wo_asso_loss.py --cfg tracking/configs/MTAD.yaml --exp_name MTAD_wo_asso_loss
# python train_msta.py --cfg tracking/configs/OTB100.yaml --exp_name OTB100
# # python train_msta_wo_asso_loss.py --cfg tracking/configs/AccessAIS.yaml --exp_name AccessAIS_wo_asso_loss
# python train_msta.py --cfg tracking/configs/MSTAv2.yaml --exp_name MSTAv2
# # python train_msta_wo_asso_loss.py --cfg tracking/configs/MSTAv2.yaml --exp_name MSTAv2_wo_asso_loss

python train_MFTP.py --cfg tracking/configs/MSTAv2_MFTP.yaml --exp_name MSTAv2_MFTP
python train_MSTA.py --cfg tracking/configs/MSTAv2_MSTA.yaml --exp_name MSTAv2_MSTA
python train_MIMMA.py --cfg tracking/configs/MSTAv2_MIMMA.yaml --exp_name MSTAv2_MIMMA

python evaluate_msta.py --ckpt ./tracking/experiments/MSTAv2_MFTP/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split test
python evaluate_msta.py --ckpt ./tracking/experiments/MSTAv2_MSTA/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split test

python evaluate_msta.py --ckpt ./tracking/experiments/SatSOT_MFTP/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split test

python train_msta_wo_pstp.py --cfg tracking/configs/VISO.yaml --exp_name VISO_wo_pstp
# python train_msta_wo_asso_loss.py --cfg tracking/configs/VISO.yaml --exp_name VISO_wo_asso_loss
python train_msta_wo_pstp.py --cfg tracking/configs/SatSOT.yaml --exp_name SatSOT_wo_pstp
# python train_msta_wo_asso_loss.py --cfg tracking/configs/SatSOT.yaml --exp_name SatSOT_wo_asso_loss
python train_msta_wo_pstp.py --cfg tracking/configs/SAT_MTB.yaml --exp_name SAT_MTB_wo_pstp
# python train_msta_wo_asso_loss.py --cfg tracking/configs/SAT_MTB.yaml --exp_name SAT_MTB_wo_asso_loss
python train_msta_wo_pstp.py --cfg tracking/configs/MTAD.yaml --exp_name MTAD_wo_pstp
# python train_msta_wo_asso_loss.py --cfg tracking/configs/MTAD.yaml --exp_name MTAD_wo_asso_loss
python train_msta_wo_pstp.py --cfg tracking/configs/AccessAIS.yaml --exp_name AccessAIS_wo_pstp
# python train_msta_wo_asso_loss.py --cfg tracking/configs/AccessAIS.yaml --exp_name AccessAIS_wo_asso_loss
python train_msta_wo_pstp.py --cfg tracking/configs/OOTB.yaml --exp_name OOTB_wo_pstp
# python train_msta_wo_asso_loss.py --cfg tracking/configs/MTAD.yaml --exp_name MTAD_wo_asso_loss
python train_msta_wo_pstp.py --cfg tracking/configs/OTB100.yaml --exp_name OTB100_wo_pstp
# python train_msta_wo_asso_loss.py --cfg tracking/configs/AccessAIS.yaml --exp_name AccessAIS_wo_asso_loss
python train_msta_wo_pstp.py --cfg tracking/configs/MSTAv2.yaml --exp_name MSTAv2_wo_pstp
# python train_msta_wo_asso_loss.py --cfg tracking/configs/MSTAv2.yaml --exp_name MSTAv2_wo_asso_loss

# python evaluate_msta.py --ckpt ./tracking/experiments/VISO/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split test
# python evaluate_msta.py --ckpt ./tracking/experiments/VISO/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split val
# python evaluate_msta.py --ckpt ./tracking/experiments/VISO/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split train
# python evaluate_msta.py --ckpt ./tracking/experiments/VISO_wo_asso_loss/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split test
# python evaluate_msta.py --ckpt ./tracking/experiments/VISO_wo_asso_loss/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split val
# python evaluate_msta.py --ckpt ./tracking/experiments/VISO_wo_asso_loss/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split train

# python evaluate_msta.py --ckpt ./tracking/experiments/SatSOT/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split test
# python evaluate_msta.py --ckpt ./tracking/experiments/SatSOT/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split val
# python evaluate_msta.py --ckpt ./tracking/experiments/SatSOT/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split train
# python evaluate_msta.py --ckpt ./tracking/experiments/SatSOT_wo_asso_loss/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split test
# python evaluate_msta.py --ckpt ./tracking/experiments/SatSOT_wo_asso_loss/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split val
# python evaluate_msta.py --ckpt ./tracking/experiments/SatSOT_wo_asso_loss/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split train

# python evaluate_msta.py --ckpt ./tracking/experiments/SAT_MTB/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split test
# python evaluate_msta.py --ckpt ./tracking/experiments/SAT_MTB/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split val
# python evaluate_msta.py --ckpt ./tracking/experiments/SAT_MTB/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split train
# python evaluate_msta.py --ckpt ./tracking/experiments/SAT_MTB_wo_asso_loss/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split test
# python evaluate_msta.py --ckpt ./tracking/experiments/SAT_MTB_wo_asso_loss/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split val
# python evaluate_msta.py --ckpt ./tracking/experiments/SAT_MTB_wo_asso_loss/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split train

# python evaluate_msta.py --ckpt ./tracking/experiments/MTAD/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split test
# python evaluate_msta.py --ckpt ./tracking/experiments/MTAD/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split val
# python evaluate_msta.py --ckpt ./tracking/experiments/MTAD/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split train
# python evaluate_msta.py --ckpt ./tracking/experiments/MTAD_wo_asso_loss/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split test
# python evaluate_msta.py --ckpt ./tracking/experiments/MTAD_wo_asso_loss/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split val
# python evaluate_msta.py --ckpt ./tracking/experiments/MTAD_wo_asso_loss/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split train

# python evaluate_msta.py --ckpt ./tracking/experiments/AccessAIS/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split test
# python evaluate_msta.py --ckpt ./tracking/experiments/AccessAIS/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split val
# python evaluate_msta.py --ckpt ./tracking/experiments/AccessAIS/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split train
# python evaluate_msta.py --ckpt ./tracking/experiments/AccessAIS_wo_asso_loss/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split test
# python evaluate_msta.py --ckpt ./tracking/experiments/AccessAIS_wo_asso_loss/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split val
# python evaluate_msta.py --ckpt ./tracking/experiments/AccessAIS_wo_asso_loss/checkpoints/best_val_checkpoint.pth.tar --metric ade_fde --modality traj+all --split train


















