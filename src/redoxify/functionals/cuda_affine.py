import torch
from torch.nn import functional as F
from kornia.geometry.transform import warp_perspective
from typing import Tuple, Union, List
from nvidia.dali.plugin.pytorch.fn import torch_python_function
from nvidia.dali.pipeline import DataNode as DALINode


def dali_affine(
    image: DALINode,
    bboxes: DALINode,
    labels: DALINode,
    rotate_degree: DALINode,
    translate_ratio: DALINode,
    scaling_ratio: DALINode,
    shear_degree_xy: DALINode,
    border: DALINode,
    border_val: DALINode,
    min_bbox_size: DALINode,
    min_area_ratio: DALINode,
    max_aspect_ratio: DALINode,
):
    func = torch_python_function(
        image,
        bboxes,
        labels,
        rotate_degree,
        translate_ratio,
        scaling_ratio,
        shear_degree_xy,
        border,
        border_val,
        min_bbox_size,
        min_area_ratio,
        max_aspect_ratio,
        function=_torch_affine,
        batch_processing=False,
        num_outputs=3,
        device="gpu",
    )
    return func


def _torch_affine(
    img: torch.Tensor,
    bboxes: torch.Tensor,
    labels: torch.Tensor,
    rotate_degree: torch.Tensor,
    translate_ratio_xy: torch.Tensor,
    scaling_ratio: torch.Tensor,
    shear_degree_xy: torch.Tensor,
    border: torch.Tensor,
    border_val: torch.Tensor,
    min_bbox_size: torch.Tensor,
    min_area_ratio: torch.Tensor,
    max_aspect_ratio: torch.Tensor,
):
    device = img.device
    height = img.shape[0] + border[1] * 2
    width = img.shape[1] + border[0] * 2
    centr_matrix = torch.eye(3, dtype=torch.float32, device=device)
    centr_matrix[0, 2] = -img.shape[1] / 2
    centr_matrix[1, 2] = -img.shape[0] / 2
    warp_matrix, scaling_ratio = _get_homography_matrix(
        height,
        width,
        rotate_degree,
        translate_ratio_xy,
        scaling_ratio,
        shear_degree_xy,
        device=device,
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
        valid_index = _filter_bboxes(
            ori_bboxes, new_bboxes, min_bbox_size, min_area_ratio, max_aspect_ratio
        )
        bboxes = new_bboxes[valid_index]
        # convert absolute bbox to normalized bbox
        bboxes[:, 0::2] = bboxes[:, 0::2] / width
        bboxes[:, 1::2] = bboxes[:, 1::2] / height
        labels = labels[valid_index]
    return new_img, bboxes, labels


def _filter_bboxes(
    original_bboxes: torch.Tensor,
    warpped_bboxes: torch.Tensor,
    min_bbox_size: torch.Tensor,
    min_area_ratio: torch.Tensor,
    max_aspect_ratio: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ori_w = original_bboxes[:, 2] - original_bboxes[:, 0]
    ori_h = original_bboxes[:, 3] - original_bboxes[:, 1]
    warp_w = warpped_bboxes[:, 2] - warpped_bboxes[:, 0]
    warp_h = warpped_bboxes[:, 3] - warpped_bboxes[:, 1]
    wh_valid_index = (warp_w > min_bbox_size) & (warp_h > min_bbox_size)
    area_valid_index = (warp_w * warp_h) / (ori_w * ori_h + 1e-8) > min_area_ratio
    aspect_ratio_valid_index = (
        torch.max(warp_w / (warp_h + 1e-8), warp_h / (warp_w + 1e-8)) < max_aspect_ratio
    )
    return wh_valid_index & area_valid_index & aspect_ratio_valid_index


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


def _get_homography_matrix(
    height: int,
    width: int,
    rotation_degree: Union[float, torch.Tensor],
    translate_ratio_xy: Union[List[float], Tuple[float], torch.Tensor],
    scaling_ratio: Union[float, torch.Tensor],
    shear_degree_xy: Union[List[float], Tuple[float], torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    rotation_matrix = _get_rotation_matrix(rotation_degree)
    scaling_matrix = _get_scaling_matrix(scaling_ratio)
    x_degree = shear_degree_xy[0]
    y_degree = shear_degree_xy[1]
    shear_matrix = _get_shear_matrix(x_degree, y_degree)
    # Translation
    trans_x = translate_ratio_xy[0] * width
    trans_y = translate_ratio_xy[1] * height
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
