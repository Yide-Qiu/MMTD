import pathlib
import os
import logging
import time
from tqdm import tqdm

import pdb
import cv2
import math
# import shapecheck
from typing import List
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader

from datasets import DataGenerator
from inference import (
    compute_detections_for_tensor,
    render_detections_to_image,
    detections_from_net,
    detections_from_network_output,
)
from models import FCOS, normalize_batch
from metrics import compute_metrics, IOULoss, SigmoidFocalLoss, WeightedFocalLoss

from targets import generate_targets

logger = logging.getLogger(__name__)


def train(args, writer: SummaryWriter):
    # val_loader = DataLoader(
    #     DataGenerator('val', args.dataroot, seq_len=10, image_transforms=[Resize(1024)]),
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=0
    # )

    train_loader = DataLoader(
        DataGenerator('train', args.dataroot, seq_len=10),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    if torch.cuda.is_available():
        logger.info("Using Cuda")
        device = torch.device("cuda:0")
    else:
        logger.warning("Cuda not available, falling back to cpu")
        device = torch.device("cpu")

    model = FCOS()
    model.to(device)
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = 1e9
    loss = 0

    for epoch in range(1, 10000):
        logger.info(f"Starting epoch {epoch}")
        with tqdm(train_loader, desc=f"[Training] epoch:{epoch}", unit="batch") as pbar:
            for batch_index, (tensor_image, tensor_traj_list) in enumerate(pbar):
                model.train()
                optimizer.zero_grad()
                # pdb.set_trace()
                tensor_image = tensor_image.to(device)
                batch = normalize_batch(tensor_image)
                # pdb.set_trace()
                # time1 = time.time()
                classes, centernesses, boxes = model(batch.squeeze(0)) # 0.02秒
                # time2 = time.time()
                # print('model time cost:', time2 - time1)
                img_height, img_width = tensor_image.shape[2:4]
                # time1 = time.time()
                class_targets, centerness_targets, box_targets = generate_targets(tensor_image.shape, tensor_traj_list, model.strides) # 0.2秒
                # pdb.set_trace()
                # time2 = time.time()
                # print('target gen time cost:', time2 - time1)
                # time1 = time.time()
                loss, class_losses, box_losses, centerness_losses = _compute_loss(
                    model.strides, classes, centernesses, boxes, class_targets, centerness_targets, box_targets
                ) # 0.02秒
                # time2 = time.time()
                # print('loss time cost:', time2 - time1)
                pbar.set_postfix(loss=loss.item())
                # train_loader.set_postfix(loss=loss.item())
                # print(loss)
                # pdb.set_trace()
                logging.info(f"[Training] Epoch: {epoch}, batch: {batch_index}/{len(train_loader)}, loss: {loss.item()}")

                writer.add_scalar("Loss/train", loss.item(), batch_index)
                loss.backward()
                optimizer.step()
            print(f"Epoch: {epoch}, loss: {loss.item()}")
        logger.info("Running validation...")
        if epoch % 2 == 0:
            detection_len_list = []
            for batch_index, (tensor_image, tensor_traj_list) in enumerate(tqdm(train_loader, desc="Evaluating Progress", unit="batch")):
                model.eval()
                tensor_image = tensor_image.to(device)
                batch = normalize_batch(tensor_image)
                classes, centernesses, boxes = model(batch.squeeze(0))
                img_height, img_width = tensor_image.shape[2:4]
                detections = detections_from_network_output(
                    img_height, img_width, classes, centernesses, boxes, model.scales, model.strides
                )
                class_targets, centerness_targets, box_targets = generate_targets(tensor_image.shape, tensor_traj_list, model.strides)
                loss, class_losses, box_losses, centerness_losses = _compute_loss(
                    model.strides, classes, centernesses, boxes, class_targets, centerness_targets, box_targets
                )
                logging.info(f"[Evaluating] Epoch: {epoch}, batch: {batch_index}/{len(train_loader)}, loss: {loss.item()}")

                writer.add_scalar("Loss/eval", loss.item(), batch_index)
                detection_len_list.append(len(detections))
                # pdb.set_trace()
                # logging.info(f"Epoch: {epoch}, batch: {batch_index}/{len(train_loader)}, loss: {loss.item()}")
                # writer.add_scalar("Loss/Eval", loss.item(), batch_index)
            detection_len_list = np.array(detection_len_list)
            print(f"Epoch: {epoch}, loss: {loss.item()}, detection_len: {np.mean(detection_len_list)}")
            if best_loss > loss:
                best_loss = loss
                path = os.path.join(writer.log_dir, f"fcos_epoch_{epoch}_loss_{best_loss}.ckpt")
                logger.info(f"Saving checkpoint to '{path}'")
                state = dict(
                    model=model.state_dict(),
                    epoch=epoch,
                    batch_index=batch_index,
                    optimizer_state=optimizer.state_dict(),
                )
                torch.save(state, path)



def _compute_loss(
    strides, classes, centernesses, boxes, class_targets, centerness_targets, box_targets
) -> torch.Tensor:
    batch_size = classes[0].shape[0]
    num_classes = classes[0].shape[-1]

    # class_loss = torch.nn.CrossEntropyLoss()  # 换成Focal Loss
    # box_loss = torch.nn.L1Loss()  # 换成IOU loss
    class_loss = WeightedFocalLoss(gamma=2, alpha=0.2)
    box_loss = IOULoss()
    centerness_loss = torch.nn.BCELoss()

    losses = []
    class_losses = []
    box_losses = []
    centerness_losses = []
    device = classes[0].device

    for feature_idx in range(len(classes)):
        cls_target = class_targets[feature_idx].to(device).view(batch_size, -1)
        centerness_target = centerness_targets[feature_idx].to(device).view(batch_size, -1)
        box_target = box_targets[feature_idx].to(device).view(batch_size, -1, 4)

        cls_view = classes[feature_idx].view(batch_size, -1, num_classes)
        box_view = boxes[feature_idx].view(batch_size, -1, 4)
        centerness_view = centernesses[feature_idx].view(batch_size, -1)
        if centerness_view.shape != centerness_target.shape:
            pdb.set_trace()
        cs = centerness_loss(centerness_view, centerness_target)
        losses.append(cs)
        centerness_losses.append(cs)
        # pdb.set_trace()
        ls = class_loss(cls_view.view(-1), cls_target.view(-1))
        losses.append(ls)
        class_losses.append(ls)

        for batch_idx in range(batch_size):
            mask = cls_target[batch_idx] > 0
            if torch.sum(cls_target[batch_idx]) > 0:
                l = box_loss(box_view[batch_idx][mask], box_target[batch_idx][mask]) * strides[feature_idx]
                # if torch.isnan(box_view[batch_idx][mask]).any():
                #     print("l contains NaN")
                # if torch.isnan(box_target[batch_idx][mask]).any():
                #     print("l contains NaN")
                # if torch.isnan(l):
                #     print("l is NaN")
                #     pdb.set_trace()
                if l < 0:
                    print(feature_idx, batch_idx)
                    # pdb.set_trace()
                losses.append(l)
                box_losses.append(l)
    if torch.stack(losses).mean() < 0:
        pdb.set_trace()
    return torch.stack(losses).mean(), class_losses, box_losses, centerness_losses


def _test_model(checkpoint, writer, model, loader, device):
    model.eval()

    all_detections = []
    all_box_labels = []
    all_class_labels = []

    # images = []
    for i, (tensor_image, tensor_traj, tensor_label) in enumerate(loader, 0):
        logging.info(f"Validation for {i}")
        tensor_image = tensor_image.to(device)
        tensor_image = normalize_batch(tensor_image)

        classes, centernesses, boxes = model(tensor_image)
        img_height, img_width = tensor_image.shape[2:4]
        detections = detections_from_network_output(
            img_height, img_width, classes, centernesses, boxes, model.scales, model.strides
        )
        # render_detections_to_image(img, detections[0])
        # _render_targets_to_image(img, box_labels[0])

        class_targets, centerness_targets, box_targets = generate_targets(
           tensor_image.shape, tensor_label[:, -1], tensor_label[:, 2:6], model.strides
        )
        if i == 0:
            for j in range(len(classes)):
                writer.add_image(f"class {i} feat {j}", classes[j][0][:, :, 1], checkpoint, dataformats="HW")
                writer.add_image(
                    f"class target {i} feat {j}", class_targets[j][0], checkpoint, dataformats="HW"
                )
                writer.add_image(f"centerness {i} feat {j}", centernesses[j][0], checkpoint, dataformats="HW")
                writer.add_image(
                    f"centerness target {i} feat {j}", centerness_targets[j][0], checkpoint, dataformats="HW"
                )

        loss, class_losses, box_losses, centerness_losses = _compute_loss(
            model.strides, classes, centernesses, boxes, class_targets, centerness_targets, box_targets
        )
        logging.info(f"Validation loss: {loss.item()}")

        writer.add_scalar("Loss/val", loss.item(), checkpoint)

        # images.append(img)
        # all_detections.extend(detections)
        # all_box_labels.extend(box_labels)
        # all_class_labels.extend(class_labels)

    # grid = _image_grid(images[0:24], 3, 2048)
    # writer.add_image(f"fcos test {i}", grid, checkpoint, dataformats="HWC")

    metrics = compute_metrics(all_detections, all_class_labels, all_box_labels)
    logging.info(
        f"Pascal voc metrics: TP: {metrics.true_positive_count}, FP: {metrics.false_positive_count}, mAP: {metrics.mean_average_precision}, total gt: {metrics.total_ground_truth_detections}"
    )
    writer.add_scalar("Metrics/mAP", metrics.mean_average_precision, checkpoint)

    writer.flush()


def _image_grid(images: List[np.ndarray], images_per_row: int, image_width: int) -> np.ndarray:
    max_image_width = int(image_width / images_per_row)
    images_per_col = int(math.ceil(len(images) / images_per_row))

    rescale = min(max_image_width / image.shape[1] for image in images)

    max_height = max(int(image.shape[0] * rescale) for image in images)

    image_height = images_per_col * max_height
    result = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        resized_image = cv2.resize(img, (int(img.shape[1] * rescale), int(img.shape[0] * rescale)))

        r, c = divmod(i, images_per_row)
        c *= max_image_width
        r *= max_height

        h = resized_image.shape[0]
        w = resized_image.shape[1]

        result[r : r + h, c : c + w] = resized_image

    return result


# @shapecheck.check_args(box_labels=("N", ("min_x", "min_y", "max_x", "max_y")))
def _render_targets_to_image(img: np.ndarray, box_labels: torch.Tensor):
    for i in range(box_labels.shape[0]):
        start_point = (int(box_labels[i][0].item()), int(box_labels[i][1].item()))
        end_point = (int(box_labels[i][2].item()), int(box_labels[i][3].item()))
        img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), 1)

    return img



