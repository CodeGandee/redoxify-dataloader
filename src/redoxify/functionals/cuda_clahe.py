import torch
from kornia.enhance import equalize_clahe
from nvidia.dali.plugin.pytorch.fn import torch_python_function
from nvidia.dali.pipeline import DataNode as DALINode


def dali_blur_image(image: DALINode, clip_limit: DALINode, grid_size: DALINode):
    func = torch_python_function(
        image, clip_limit, grid_size,
        function=_clahe,
        batch_processing=False,
        num_outputs=1,
        device='gpu'
    )
    return func


def _clahe(img: torch.Tensor, clip_limit: torch.Tensor, grid_size: torch.Tensor):
    """
    Apply CLAHE to an image.
    
    Args:
        img (Tensor): Input image tensor of shape (H, W, C).
        clip_limit (Tensor): Clip limit for contrast limiting.
        grid_size (Tensor): 2D tensor specifying the grid size for CLAHE.
    
    Returns:
        Tensor: CLAHE applied image.
    """
    H, W, C = img.shape
    img = img.permute(2, 0, 1).unsqueeze(0)/255  # Convert to shape (1, C, H, W) and normalize
    clip_limit = clip_limit.float().item()
    grid_size = tuple(grid_size.int().tolist())
    clahe_img = equalize_clahe(img, clip_limit, grid_size, slow_and_differentiable=False)
    return clahe_img.squeeze(0).permute(1, 2, 0)*255  # Convert back to shape (H, W, C) and denormalize