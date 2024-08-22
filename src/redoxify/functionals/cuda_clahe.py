import torch
from kornia.enhance import equalize_clahe
from nvidia.dali.plugin.pytorch.fn import torch_python_function
from nvidia.dali.pipeline import DataNode as DALINode


def dali_clahe_image(image: DALINode, clip_limit: DALINode, grid_size: DALINode, probability: DALINode):
    func = torch_python_function(
        image, clip_limit, grid_size, probability,
        function=_clahe,
        batch_processing=False,
        num_outputs=1,
        device='gpu'
    )
    return func


def _clahe(img: torch.Tensor, clip_limit: torch.Tensor, grid_size: torch.Tensor, probability: torch.Tensor):
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
        return image
    img = img.permute(2, 0, 1).unsqueeze(0)/255  # Convert to shape (1, C, H, W) and normalize
    clip_limit = clip_limit.float().item()
    grid_size = tuple(grid_size.int().tolist())
    clahe_img = equalize_clahe(img, clip_limit, grid_size, slow_and_differentiable=False)
    return clahe_img.squeeze(0).permute(1, 2, 0)*255  # Convert back to shape (H, W, C) and denormalize




if __name__ == '__main__':
    # Test the blur functions
    import cv2
    import numpy as np
    image = cv2.imread("tests/富士山.jpg")
    image = cv2.resize(image, (512, 512))
    cv2.imwrite("tests/富士山_resized.jpg", image)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # cv_clahe_image = clahe.apply(image)
    print(image.shape)
    image = torch.from_numpy(image).to(device='cuda')
    clip_limit = torch.tensor(2)
    grid_size = torch.tensor([8, 8])
    blurred_image = _clahe(image, clip_limit, grid_size)
    blurred_image = blurred_image.cpu().numpy().astype(np.uint8)
    cv2.imwrite("tests/富士山_clahe.jpg", blurred_image)
    
    