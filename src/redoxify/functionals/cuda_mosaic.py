# functions used in dali python_function
import torch
import random
import numpy as np
from typing import List, Tuple, Sequence

from nvidia.dali.plugin.pytorch.fn import torch_python_function
from nvidia.dali.pipeline import DataNode as DALINode


def dali_mosaic_images(
    images: DALINode,
    boxes: DALINode,
    labels: DALINode,
    pad_val: DALINode,
    sub_image_scale: DALINode,
    center_ratio_range: DALINode,
    probability: DALINode,
):
    func = torch_python_function(
        images,
        boxes,
        labels,
        pad_val,
        sub_image_scale,
        center_ratio_range,
        probability,
        function=_dali_mosaic_images,
        batch_processing=True,
        num_outputs=3,
        device="gpu",
    )
    return func


def _dali_mosaic_images(
    images_list: List[torch.Tensor],
    boxes_list: List[torch.Tensor],
    labels_list: List[torch.Tensor],
    pad_val_list: List[torch.Tensor],
    sub_image_scale_list: List[torch.Tensor],
    center_ratio_range_list: List[torch.Tensor],
    probabilities_list: List[torch.Tensor],
):
    """Mosaic augmentation.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.
    """
    # images: list of N images, all of them have the same shape. assert the shape
    ori_img_dtype = images_list[0].dtype
    center_ratio_range = center_ratio_range_list[0]
    img_scale = sub_image_scale_list[0]
    img_scale_w, img_scale_h = img_scale[0].item(), img_scale[1].item()
    for i, img_i in enumerate(images_list):
        h_i, w_i = img_i.shape[:2]
        scale_ratio_i = min(img_scale_h / h_i, img_scale_w / w_i)
        img_i = img_i.permute(2, 0, 1).to(torch.float32)
        img_i = torch.nn.functional.interpolate(
            img_i.unsqueeze(0),
            size=(int(h_i * scale_ratio_i), int(w_i * scale_ratio_i)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        img_i = img_i.permute(1, 2, 0).to(ori_img_dtype)
        images_list[i] = img_i

    num_images = len(images_list)
    if num_images <= 1:
        indices = list(range(num_images)) * 4
    elif num_images < 4:
        indices = list(range(num_images)) * 2
    else:
        indices = list(range(num_images))
    mosaic_img_list = []
    mosaic_bboxes_list = []
    mosaic_bboxes_labels_list = []
    for img_idx, cur_img in enumerate(images_list):
        selected_indices = random.sample(indices, 4)
        center_x = int(
            random.uniform(center_ratio_range[0], center_ratio_range[1]) * img_scale_w
        )
        center_y = int(
            random.uniform(center_ratio_range[0], center_ratio_range[1]) * img_scale_h
        )
        center_position = (center_x, center_y)
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_img = torch.full(
            (int(img_scale_h * 2), int(img_scale_w * 2), 3),
            pad_val_list[0],
            dtype=cur_img.dtype,
            device=cur_img.device,
        )
        loc_strs = ("top_left", "top_right", "bottom_left", "bottom_right")
        for i, loc in enumerate(loc_strs):
            selected_index = selected_indices[i]
            img_i = images_list[selected_index]
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(img_scale_h / h_i, img_scale_w / w_i)
            img_i = img_i.permute(2, 0, 1).to(torch.float32)
            img_i = torch.nn.functional.interpolate(
                img_i.unsqueeze(0),
                size=(int(h_i * scale_ratio_i), int(w_i * scale_ratio_i)),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            img_i = img_i.permute(1, 2, 0).to(ori_img_dtype)
            # compute the combine parameters
            paste_coord, crop_coord = _mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1], img_scale
            )
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = boxes_list[selected_index].clone()
            gt_bboxes_i *= torch.tensor(
                [w_i, h_i, w_i, h_i], dtype=gt_bboxes_i.dtype, device=gt_bboxes_i.device
            )
            gt_bboxes_labels_i = labels_list[selected_index]

            padw = x1_p - x1_c
            padh = y1_p - y1_c
            gt_bboxes_i[:, 0::2] += padw
            gt_bboxes_i[:, 1::2] += padh
            # clamp the boxes to the mosaic image
            gt_bboxes_i[:, 0::2] = torch.clamp(
                gt_bboxes_i[:, 0::2], min=0, max=img_scale_w * 2
            )
            gt_bboxes_i[:, 1::2] = torch.clamp(
                gt_bboxes_i[:, 1::2], min=0, max=img_scale_h * 2
            )
            boxes_w = gt_bboxes_i[:, 2] - gt_bboxes_i[:, 0]
            boxes_h = gt_bboxes_i[:, 3] - gt_bboxes_i[:, 1]
            valid_inds = (boxes_w > 1) & (boxes_h > 1)
            gt_bboxes_i = gt_bboxes_i[valid_inds]
            gt_bboxes_labels_i = gt_bboxes_labels_i[valid_inds]
            gt_bboxes_i /= torch.tensor(
                [img_scale_w * 2, img_scale_h * 2, img_scale_w * 2, img_scale_h * 2],
                dtype=gt_bboxes_i.dtype,
                device=gt_bboxes_i.device,
            )
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)
        mosaic_bboxes = torch.cat(mosaic_bboxes, dim=0)
        mosaic_bboxes_labels = torch.cat(mosaic_bboxes_labels, dim=0)
        mosaic_img_list.append(mosaic_img)
        mosaic_bboxes_list.append(mosaic_bboxes)
        mosaic_bboxes_labels_list.append(mosaic_bboxes_labels)
    for img_idx in range(len(images_list)):
        probability = probabilities_list[img_idx]
        if torch.rand(probability.shape, device=probability.device) > probability:
            mosaic_img_list[img_idx] = images_list[img_idx]
            mosaic_bboxes_list[img_idx] = boxes_list[img_idx]
            mosaic_bboxes_labels_list[img_idx] = labels_list[img_idx]
    return mosaic_img_list, mosaic_bboxes_list, mosaic_bboxes_labels_list


def _mosaic_combine(
    loc: str,
    center_position_xy: Sequence[float],
    img_shape_wh: Sequence[int],
    img_scale: Sequence[int],
) -> Tuple[Tuple[int], Tuple[int]]:
    """Calculate global coordinate of mosaic image and local coordinate of
    cropped sub-image.

    Args:
        loc (str): Index for the sub-image, loc in ('top_left',
            'top_right', 'bottom_left', 'bottom_right').
        center_position_xy (Sequence[float]): Mixing center for 4 images,
            (x, y).
        img_shape_wh (Sequence[int]): Width and height of sub-image

    Returns:
        tuple[tuple[float]]: Corresponding coordinate of pasting and
            cropping
            - paste_coord (tuple): paste corner coordinate in mosaic image.
            - crop_coord (tuple): crop corner coordinate in mosaic image.
    """
    assert loc in ("top_left", "top_right", "bottom_left", "bottom_right")
    if loc == "top_left":
        # index0 to top left part of image
        x1, y1, x2, y2 = (
            max(center_position_xy[0] - img_shape_wh[0], 0),
            max(center_position_xy[1] - img_shape_wh[1], 0),
            center_position_xy[0],
            center_position_xy[1],
        )
        crop_coord = (
            img_shape_wh[0] - (x2 - x1),
            img_shape_wh[1] - (y2 - y1),
            img_shape_wh[0],
            img_shape_wh[1],
        )

    elif loc == "top_right":
        # index1 to top right part of image
        x1, y1, x2, y2 = (
            center_position_xy[0],
            max(center_position_xy[1] - img_shape_wh[1], 0),
            min(center_position_xy[0] + img_shape_wh[0], img_scale[0] * 2),
            center_position_xy[1],
        )
        crop_coord = (
            0,
            img_shape_wh[1] - (y2 - y1),
            min(img_shape_wh[0], x2 - x1),
            img_shape_wh[1],
        )

    elif loc == "bottom_left":
        # index2 to bottom left part of image
        x1, y1, x2, y2 = (
            max(center_position_xy[0] - img_shape_wh[0], 0),
            center_position_xy[1],
            center_position_xy[0],
            min(img_scale[1] * 2, center_position_xy[1] + img_shape_wh[1]),
        )
        crop_coord = (
            img_shape_wh[0] - (x2 - x1),
            0,
            img_shape_wh[0],
            min(y2 - y1, img_shape_wh[1]),
        )

    else:
        # index3 to bottom right part of image
        x1, y1, x2, y2 = (
            center_position_xy[0],
            center_position_xy[1],
            min(center_position_xy[0] + img_shape_wh[0], img_scale[0] * 2),
            min(img_scale[1] * 2, center_position_xy[1] + img_shape_wh[1]),
        )
        crop_coord = 0, 0, min(img_shape_wh[0], x2 - x1), min(y2 - y1, img_shape_wh[1])

    paste_coord = x1, y1, x2, y2
    return paste_coord, crop_coord


# for debugging
if __name__ == "__main__":
    import cv2

    img = cv2.imread("/workspace/redoxify-dataloader/tests/fuji.jpg")
    img = img.copy()
    print(img.shape)
    # img = img.astype(np.float32)
    img_num = 16
    images = [torch.from_numpy(img).to("cuda") for _ in range(img_num)]
    boxes = [
        torch.tensor([[0.25, 0.25, 0.75, 0.75]], device="cuda") for _ in range(img_num)
    ]
    labels = [torch.randint(0, 10, (1,), device="cuda") for _ in range(img_num)]
    probabilities = [torch.tensor(0.8, device="cuda") for _ in range(img_num)]
    pad_val_list = [torch.tensor(114, device="cuda") for _ in range(img_num)]
    sub_image_scale = [torch.tensor([640, 640], device="cuda") for _ in range(img_num)]
    center_ratio_range_list = [
        torch.tensor([0.5, 1.5], device="cuda") for _ in range(img_num)
    ]
    mosaics, mosaic_boxes, mosaic_labels = _dali_mosaic_images(
        images,
        boxes,
        labels,
        pad_val_list,
        sub_image_scale,
        center_ratio_range_list,
        probabilities,
    )
    for i, mosaic in enumerate(mosaics):
        mosaic = mosaic.cpu().numpy().astype(np.uint8)
        mosaic = mosaic.copy()
        print(mosaic.shape)
        # cv2.imwrite(f'temp/mosaic_{i}.jpg', mosaic)
        for j, box in enumerate(mosaic_boxes[i]):
            box = box.cpu().numpy()
            box *= np.array(
                [mosaic.shape[1], mosaic.shape[0], mosaic.shape[1], mosaic.shape[0]]
            )
            box = box.astype(np.int32)
            label = mosaic_labels[i][j].item()
            cv2.rectangle(mosaic, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(
                mosaic,
                str(label),
                (box[0], box[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        cv2.imwrite(f"temp/mosaic_{i}_bbox.jpg", mosaic)
