import os
import pdb
import argparse
from torch.utils.tensorboard import SummaryWriter
from train import train

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default='/data4/ymh/MOT/codes/DetectionMethod/data/VISO/mot/car',  help="checkpoint root")
    parser.add_argument("--experiment_root", type=str, default='experiments/viso/car',  help="checkpoint root")
    parser.add_argument("--batch_size", type=int, default=1,  help="checkpoint root")
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = make_args()
    os.makedirs(args.experiment_root, exist_ok=True)
    train(args, writer=SummaryWriter(log_dir=args.experiment_root))

