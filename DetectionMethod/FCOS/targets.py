from typing import Tuple, List
import torch
import math
import pdb


def generate_targets(
    img_shape: torch.LongTensor,
    tensor_traj_list: List[torch.LongTensor],
    strides: List[int],
) -> List[Tuple[torch.FloatTensor, torch.LongTensor]]:
    """
    Given the shape of the input image, the box and class labels, and the stride of each
    feature map, construct the model targets for FCOS at each feature map scale.
    """

    batch_size = img_shape[1]

    class_targets_by_feature = []
    centerness_target_by_feature = []
    box_targets_by_feature = []

    # m = (0, 64, 128, 256, 512, math.inf)
    m = (0, 8, 16, 32, 128, math.inf)
    # pdb.set_trace()
    # [8, 16, 32, 64, 128] --> strides
    for i, stride in enumerate(strides):
        feat_h = int(img_shape[-2] / stride)  # 1024 / 8 = 128
        feat_w = int(img_shape[-1] / stride)

        class_target_for_feature = torch.zeros(batch_size, feat_h, feat_w, dtype=int)
        # print('class_target_for_feature.shape', class_target_for_feature.shape)
        centerness_target_for_feature = torch.zeros(batch_size, feat_h, feat_w)
        box_target_for_feature = torch.zeros(batch_size, feat_h, feat_w, 4)

        min_box_side = m[i]
        max_box_side = m[i + 1]

        for batch_idx, (tensor_traj) in enumerate(
            zip(tensor_traj_list)
        ):
            # pdb.set_trace()
            heights = tensor_traj[0][0, 0, :, -1, 3] - tensor_traj[0][0, 0, :, -1, 1]  # ~6
            widths = tensor_traj[0][0, 0, :, -1, 2] - tensor_traj[0][0, 0, :, -1, 0]  # ~6
            # heights = box_labels[:, 3]
            # widths = box_labels[:, 2]
            areas = torch.mul(widths, heights)

            for j in torch.argsort(areas, dim=0, descending=True):
                # 大目标使用前N层
                if max(heights[j], widths[j]) < min_box_side:
                    continue

                min_x = max(int(tensor_traj[0][0, 0, j, -1, 0] / stride) - 1, 0)
                min_y = max(int(tensor_traj[0][0, 0, j, -1, 1] / stride) - 1, 0)
                max_x = min(int(tensor_traj[0][0, 0, j, -1, 2] / stride) + 2, feat_w)
                max_y = min(int(tensor_traj[0][0, 0, j, -1, 3] / stride) + 2, feat_h)
                # pdb.set_trace()
                for x in range(min_x, max_x):
                    for y in range(min_y, max_y):
                        if x == 0 or y == 0 or x == max_x - 1 or y == max_y - 1:
                            dist = 0
                        else:
                            l = x - min_x
                            r = max_x - 1 - x
                            t = y - min_y
                            b = max_y - 1 - y
                            dist = math.sqrt(min(l, r) / float(max(l, r)) * min(t, b) / float(max(t, b)))

                        centerness = dist
                        # print(dist, x, y)
                        # pdb.set_trace()
                        centerness_target_for_feature[batch_idx, x, y] = centerness

                # pdb.set_trace()
                class_target_for_feature[batch_idx, min_x : max_x, min_y : max_y] = 1
                # print(stride, batch_idx, min_y, max_y, min_x, max_x, class_labels[j])
                # print(class_target_for_feature[
                #     batch_idx, min_y + 1 : max_y - 1, min_x + 1 : max_x - 1
                # ])
                # pdb.set_trace()

                for x in range(min_x, max_x):
                    for y in range(min_y, max_y):
                        target = torch.Tensor(
                            [
                                x - float(tensor_traj[0][0, 0, j, -1, 0]) / stride,
                                y - float(tensor_traj[0][0, 0, j, -1, 1]) / stride,
                                float(tensor_traj[0][0, 0, j, -1, 2]) / stride - x,
                                float(tensor_traj[0][0, 0, j, -1, 3]) / stride - y,
                            ]
                        )
                        # pdb.set_trace()
                        box_target_for_feature[batch_idx, x, y] = target
                #         pdb.set_trace()
                # pdb.set_trace()

        class_targets_by_feature.append(class_target_for_feature)
        # pdb.set_trace()
        centerness_target_by_feature.append(centerness_target_for_feature)
        box_targets_by_feature.append(box_target_for_feature)

    # pdb.set_trace()
    # Return [BHWC, BHW[min_x, min_y, max_x, max_y]]
    return class_targets_by_feature, centerness_target_by_feature, box_targets_by_feature
