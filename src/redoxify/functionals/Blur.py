# functions used in dali python_function
import torch
import torch.nn.functional as F
import random
import numpy as np

from nvidia.dali.plugin.pytorch.fn import torch_python_function
from nvidia.dali.pipeline import DataNode as DALINode

def dali_blur_image(image: DALINode, blur_limit: DALINode):
    func = torch_python_function(
        image, blur_limit,
        function=_blur_image,
        batch_processing=False,
        num_outputs=1,
        device='gpu'
    )
    return func

def _blur_image(image: torch.Tensor, blur_limit: torch.Tensor):
    """
    Apply a blur filter to an image using PyTorch. The blur kernel size is randomly
    chosen between 3 and `blur_limit`.
    
    Args:
    - image (torch.Tensor): Input image of shape (H, W, 3).
    - blur_limit (torch.Tensor): Maximum blur kernel size.
    
    Returns:
    - torch.Tensor: Blurred image.
    """
    if image.ndim != 3 or image.size(2) != 3:
        raise ValueError("Input image must be of shape (H, W, 3)")
    kernel_size = random.randint(3, blur_limit.item())
    kernel_size = kernel_size - 1 if kernel_size % 2 == 0 else kernel_size
    #create the blur kernel
    kernel = torch.ones(kernel_size, kernel_size) / kernel_size**2
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    #pad the image to ensure the output size matches the input size
    padding = kernel_size // 2
    image = F.pad(image.permute(2, 0, 1).unsqueeze(0), (padding, padding, padding, padding), mode='reflect')
    
    #apply the convolution
    blurred_image = F.conv2d(image, kernel, groups=3)
    
    #remove extra dimensions and permute back to (H, W, 3)
    blurred_image = blurred_image.squeeze(0).permute(1, 2, 0)
    
    return blurred_image

def dali_median_blur_image(image: DALINode, blur_limit: DALINode):
    func = torch_python_function(
        image, blur_limit,
        function=_median_blur_image,
        batch_processing=False,
        num_outputs=1,
        device='gpu'
    )
    return func

def _median_blur_image(image: torch.Tensor, blur_limit: int) -> torch.Tensor:
    """
    Apply a median blur filter to an image using PyTorch.
    
    Args:
    - image (torch.Tensor): Input image of shape (H, W, 3).
    - kernel_size (int): Size of the median blur kernel. Must be an odd number.
    
    Returns:
    - torch.Tensor: Median blurred image.
    """
    if image.ndim != 3 or image.size(2) != 3:
        raise ValueError("Input image must be of shape (H, W, 3)")
    #randomly choose the kernel size while ensuring it is odd
    kernel_size = random.randint(3, blur_limit)
    kernel_size = kernel_size - 1 if kernel_size % 2 == 0 else kernel_size
    
    padding = kernel_size // 2
    image_padded = F.pad(image.permute(2, 0, 1).unsqueeze(0), (padding, padding, padding, padding), mode='reflect')
    
    patches = image_padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    patches = patches.contiguous().view(3, image.size(0), image.size(1), -1)
    
    median_values = patches.median(dim=-1)[0]
    
    return median_values.permute(1, 2, 0)