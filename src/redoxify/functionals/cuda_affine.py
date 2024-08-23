import torch
from torch.nn import functional as F
from kornia.geometry.transform import warp_perspective
from typing import Tuple
from nvidia.dali.plugin.pytorch.fn import torch_python_function
from nvidia.dali.pipeline import DataNode as DALINode


def dali_random_affine(
    image: DALINode,
    bboxes: DALINode,
    labels: DALINode,
    max_rotate_degree: DALINode,
    max_translate_ratio: DALINode,
    scaling_ratio_range: DALINode,
    max_shear_degree: DALINode,
    border: DALINode,
    border_val: DALINode,
    min_bbox_size: DALINode,
):
    func = torch_python_function(
        image,
        bboxes,
        labels,
        max_rotate_degree,
        max_translate_ratio,
        scaling_ratio_range,
        max_shear_degree,
        border,
        border_val,
        min_bbox_size,
        function=_random_affine,
        batch_processing=False,
        num_outputs=3,
        device="gpu",
    )
    return func


def _random_affine(
    img: torch.Tensor,
    bboxes: torch.Tensor,
    labels: torch.Tensor,
    max_rotate_degree: torch.Tensor,
    max_translate_ratio: torch.Tensor,
    scaling_ratio_range: torch.Tensor,
    max_shear_degree: torch.Tensor,
    border: torch.Tensor,
    border_val: torch.Tensor,
    min_bbox_size: torch.Tensor,
):
    height = img.shape[0] + border[1] * 2
    width = img.shape[1] + border[0] * 2
    centr_matrix = torch.eye(3, dtype=torch.float32)
    centr_matrix[0, 2] = -img.shape[1] / 2
    centr_matrix[1, 2] = -img.shape[0] / 2

    warp_matrix, scaling_ratio = _get_random_homography_matrix(
        height,
        width,
        max_rotate_degree,
        max_translate_ratio,
        scaling_ratio_range,
        max_shear_degree,
        device=img.device,
    )
    warp_matrix = warp_matrix @ centr_matrix

    new_img = warp_perspective(
        img.permute(2, 0, 1).unsqueeze(0).float(),
        warp_matrix.unsqueeze(0),
        dsize=(height, width),
        padding_mode="fill",
        fill_value=border_val,
    )
    new_img = new_img.squeeze(0).permute(1, 2, 0).byte()

    num_bboxes = bboxes.shape[0]
    if num_bboxes > 0:
        ori_bboxes = bboxes.clone()
        # convert normalized bbox to absolute bbox
        ori_bboxes[:, 0::2] = ori_bboxes[:, 0::2] * img.shape[1]
        ori_bboxes[:, 1::2] = ori_bboxes[:, 1::2] * img.shape[0]
        corners = bbox2corners(ori_bboxes)
        corners = torch.cat([corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
        corners_T = torch.transpose(corners, -1, -2)
        corners_T = torch.matmul(warp_matrix, corners_T)
        corners = torch.transpose(corners_T, -1, -2)
        # Convert to homogeneous coordinates by normalization
        corners = corners[..., :2] / corners[..., 2:3]
        new_bboxes = corners2bbox(corners)
        new_bboxes[..., 0::2] = new_bboxes[..., 0::2].clamp(0, width)
        new_bboxes[..., 1::2] = new_bboxes[..., 1::2].clamp(0, height)
        valid_index = (new_bboxes[:, 2] - new_bboxes[:, 0] > min_bbox_size) & (
            new_bboxes[:, 3] - new_bboxes[:, 1] > min_bbox_size
        )
        bboxes = new_bboxes[valid_index]
        # convert absolute bbox to normalized bbox
        bboxes[:, 0::2] = bboxes[:, 0::2] / width
        bboxes[:, 1::2] = bboxes[:, 1::2] / height
        labels = labels[valid_index]
    return new_img, bboxes, labels


def corners2bbox(corners: torch.Tensor) -> torch.Tensor:
    """Convert box coordinates from corners ((x1, y1), (x2, y1), (x1, y2),
    (x2, y2)) to (x1, y1, x2, y2).

    Args:
        corners (Tensor): Corner tensor with shape of (..., 4, 2).

    Returns:
        Tensor: Horizontal box tensor with shape of (..., 4).
    """
    if corners.numel() == 0:
        return corners.new_zeros((0, 4))
    min_xy = corners.min(dim=-2)[0]
    max_xy = corners.max(dim=-2)[0]
    return torch.cat([min_xy, max_xy], dim=-1)


def bbox2corners(bboxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = torch.split(bboxes, 1, dim=-1)
    corners = torch.cat([x1, y1, x2, y1, x2, y2, x1, y2], dim=-1)
    return corners.reshape(*corners.shape[:-1], 4, 2)


def _torch_uniform(left: float, right: float) -> torch.Tensor:
    return torch.rand(1) * (right - left) + left


def _get_random_homography_matrix(
    height: int,
    width: int,
    max_rotate_degree: float,
    max_translate_ratio: float,
    scaling_ratio_range: Tuple[float, float],
    max_shear_degree: float,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    rotation_degree = _torch_uniform(-max_rotate_degree, max_rotate_degree)
    rotation_matrix = _get_rotation_matrix(rotation_degree)

    # Scaling
    scaling_ratio = _torch_uniform(scaling_ratio_range[0], scaling_ratio_range[1])
    scaling_matrix = _get_scaling_matrix(scaling_ratio)

    # Shear
    x_degree = _torch_uniform(-max_shear_degree, max_shear_degree)
    y_degree = _torch_uniform(-max_shear_degree, max_shear_degree)
    shear_matrix = _get_shear_matrix(x_degree, y_degree)

    # Translation
    trans_x = (
        _torch_uniform(0.5 - max_translate_ratio, 0.5 + max_translate_ratio) * width
    )
    trans_y = (
        _torch_uniform(0.5 - max_translate_ratio, 0.5 + max_translate_ratio) * height
    )
    translate_matrix = _get_translation_matrix(trans_x, trans_y)
    warp_matrix = translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix
    return warp_matrix.to(device=device), scaling_ratio.to(device=device)


def _get_rotation_matrix(rotation_degree: float) -> torch.Tensor:
    rotation_rad = rotation_degree * (torch.pi / 180)
    rotation_matrix = torch.tensor(
        [
            [torch.cos(rotation_rad), -torch.sin(rotation_rad), 0],
            [torch.sin(rotation_rad), torch.cos(rotation_rad), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    return rotation_matrix


def _get_scaling_matrix(scaling_ratio: float) -> torch.Tensor:
    scaling_matrix = torch.tensor(
        [
            [scaling_ratio, 0, 0],
            [0, scaling_ratio, 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    return scaling_matrix


def _get_shear_matrix(x_degree: float, y_degree: float) -> torch.Tensor:
    x_rad = x_degree * (torch.pi / 180)
    y_rad = y_degree * (torch.pi / 180)
    shear_matrix = torch.tensor(
        [
            [1, torch.tan(x_rad), 0],
            [torch.tan(y_rad), 1, 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    return shear_matrix


def _get_translation_matrix(x: float, y: float) -> torch.Tensor:
    translation_matrix = torch.tensor(
        [
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    return translation_matrix


if __name__ == "__main__":
    # Test the blur functions
    import cv2
    import numpy as np

    border = 320
    img = cv2.imread("tests/fuji.jpg")
    img = cv2.resize(img, (1280, 1280))
    img_tensor = torch.from_numpy(img)
    bboxes = torch.tensor([[0.1, 0.4, 0.9, 0.9]])
    labels = torch.tensor([1])
    # draw bbox and label on image
    box = bboxes[0].numpy()
    img = cv2.rectangle(img, (int(box[0] * img.shape[1]), int(box[1] * img.shape[0])), (int(box[2] * img.shape[1]), int(box[3] * img.shape[0])), (0, 255, 0), 2)
    img = cv2.putText(
        img,
        "1",
        (int(box[0] * img.shape[1]), int(box[1] * img.shape[0])),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite("tests/fuji_bbox.jpg", img)
    for i in range(10):
        warp_img, warp_bboxes, warp_labels = _random_affine(
            img_tensor,
            bboxes,
            labels,
            max_rotate_degree=torch.tensor(0.0),
            max_shear_degree=torch.tensor(0.0),
            scaling_ratio_range=torch.tensor([0.5, 1.5]),
            max_translate_ratio=torch.tensor(0.1),
            border=torch.tensor([-border, -border]),
            border_val=torch.tensor([114, 114, 114]),
            min_bbox_size=torch.tensor(10),
        )
        warp_img = warp_img.numpy().copy()
        warp_bboxes = warp_bboxes.numpy()
        warp_labels = warp_labels.numpy()
        box = warp_bboxes[0]
        warp_img = cv2.rectangle(warp_img, (int(box[0] * warp_img.shape[1]), int(box[1] * warp_img.shape[0])), (int(box[2] * warp_img.shape[1]), int(box[3] * warp_img.shape[0])), (0, 255, 0), 2)
        warp_img = cv2.putText(
            warp_img,
            "1",
            (int(box[0] * warp_img.shape[1]), int(box[1] * warp_img.shape[0])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        print(warp_img.shape)
        cv2.imwrite(f"temp/fuji_bbox_warp{i}.jpg", warp_img)
        