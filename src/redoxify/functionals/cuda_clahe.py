import torch
import random
from kornia.color import rgb_to_lab, lab_to_rgb
from kornia.enhance import equalize_clahe
from nvidia.dali.plugin.pytorch.fn import torch_python_function
from nvidia.dali.pipeline import DataNode as DALINode


def dali_clahe_image(
    image: DALINode,
    clip_limit: DALINode,
    tile_grid_size: DALINode,
    probability: DALINode,
):
    func = torch_python_function(
        image,
        clip_limit,
        tile_grid_size,
        probability,
        function=_clahe,
        batch_processing=False,
        num_outputs=1,
        output_layouts=["HWC"],
        device="gpu",
    )
    return func


def _clahe(
    img: torch.Tensor,
    clip_limit: torch.Tensor,
    tile_grid_size: torch.Tensor,
    probability: torch.Tensor,
):
    """
    Apply CLAHE to an image.

    Args:
        img (Tensor): Input image tensor of shape (H, W, C).
        clip_limit (Tensor): Clip limit for contrast limiting.
        tile_grid_size (Tensor): 2D tensor specifying the grid size for CLAHE.

    Returns:
        Tensor: CLAHE applied image.
    """
    if torch.rand(1, device=probability.device) > probability:
        return img
    img = (
        img.permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255
    )  # Convert to shape (1, C, H, W) and normalize
    if clip_limit.shape == (2,):
        clip_limit = random.uniform(clip_limit[0].item(), clip_limit[1].item())
    else:
        clip_limit = random.uniform(1.0, clip_limit.item())
    tile_grid_size = tuple(tile_grid_size.int().tolist())
    # kornia.color.rgb_to_lab() returns L in [0, 100] and ab in [-127, 128]
    # but kornia.enhance.equalize() expects input in [0, 1]
    img_lab = rgb_to_lab(img)
    img_lab[:, 0, ...] = equalize_clahe(
        img_lab[:, 0, ...]/100.0, clip_limit, tile_grid_size, 
        slow_and_differentiable=False
    )*100.0
    img_rgb = lab_to_rgb(img_lab)
    clahe_img = (
        img_rgb.squeeze(0).permute(1, 2, 0) * 255
    ).byte()  # Convert back to shape (H, W, C) and denormalize
    return clahe_img 


def old_clahe(
    img: torch.Tensor,
    clip_limit: torch.Tensor,
    tile_grid_size: torch.Tensor,
    probability: torch.Tensor,
):
    """
    Apply CLAHE to an image.

    Args:
        img (Tensor): Input image tensor of shape (H, W, C).
        clip_limit (Tensor): Clip limit for contrast limiting.
        grid_size (Tensor): 2D tensor specifying the grid size for CLAHE.

    Returns:
        Tensor: CLAHE applied image.
    """
    if torch.rand(1, device=probability.device) > probability:
        return img
    img = (
        img.permute(2, 0, 1).unsqueeze(0) / 255
    )  # Convert to shape (1, C, H, W) and normalize
    if clip_limit.shape == (2,):
        clip_limit = random.uniform(clip_limit[0].item(), clip_limit[1].item())
    else:
        clip_limit = random.uniform(1.0, clip_limit.item())
    tile_grid_size = tuple(tile_grid_size.int().tolist())
    clahe_img = equalize_clahe(
        img, clip_limit, tile_grid_size, slow_and_differentiable=False
    )
    clahe_img = (
        clahe_img.squeeze(0).permute(1, 2, 0) * 255
    ).byte()  # Convert back to shape (H, W, C) and denormalize
    return clahe_img
