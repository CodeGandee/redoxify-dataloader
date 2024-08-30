# functions used in dali python_function
import torch
import torch.nn.functional as F
import random
import numpy as np

from nvidia.dali.plugin.pytorch.fn import torch_python_function
from nvidia.dali.pipeline import DataNode as DALINode


def dali_blur_image(image: DALINode, blur_limit: DALINode, probability: DALINode):
    func = torch_python_function(
        image,
        blur_limit,
        probability,
        function=_blur_image,
        batch_processing=False,
        num_outputs=1,
        device="gpu",
    )
    return func


def _blur_image(
    image: torch.Tensor, blur_limit: torch.Tensor, probability: torch.Tensor
) -> torch.Tensor:
    """
    Apply blurring effect to an image with a given probability.

    Args:
        image (torch.Tensor): The input image tensor.
        blur_limit (torch.Tensor): The maximum limit for blurring.
        probability (torch.Tensor): The probability of applying blurring.

    Returns:
        torch.Tensor: The blurred image tensor.
    """
    if torch.rand(1, device=probability.device) > probability:
        return image
    kernel_size = random.randint(3, blur_limit.item())
    kernel_size = kernel_size - 1 if kernel_size % 2 == 0 else kernel_size
    # create the blur kernel
    kernel = (
        torch.ones(3, 1, kernel_size, kernel_size, device=image.device) / kernel_size**2
    )
    # pad the image to ensure the output size matches the input size
    padding = kernel_size // 2
    image = F.pad(
        image.permute(2, 0, 1).unsqueeze(0).float(),
        (padding, padding, padding, padding),
        mode="reflect",
    )

    # apply the convolution
    blurred_image = F.conv2d(image, kernel, groups=3)
    blurred_image = blurred_image.clamp(0, 255).to(torch.uint8)
    # remove extra dimensions and permute back to (H, W, 3)
    blurred_image = blurred_image.squeeze(0).permute(1, 2, 0)

    return blurred_image


def dali_median_blur_image(
    image: DALINode, blur_limit: DALINode, probability: DALINode
):
    func = torch_python_function(
        image,
        blur_limit,
        probability,
        function=_median_blur_image,
        batch_processing=False,
        output_layouts=["HWC"],
        num_outputs=1,
        device="gpu",
    )
    return func


def _median_blur_image(
    image: torch.Tensor, blur_limit: torch.Tensor, probability: torch.Tensor
) -> torch.Tensor:
    """
    Apply a median blur filter to an image using PyTorch.

    Args:
    - image (torch.Tensor): Input image of shape (H, W, 3).
    - kernel_size (int): Size of the median blur kernel. Must be an odd number.

    Returns:
    - torch.Tensor: Median blurred image.
    """
    if torch.rand(1, device=probability.device) > probability:
        return image
    # randomly choose the kernel size while ensuring it is odd
    kernel_size = random.randint(3, blur_limit.item())
    kernel_size = kernel_size - 1 if kernel_size % 2 == 0 else kernel_size

    padding = kernel_size // 2
    image_padded = F.pad(
        image.permute(2, 0, 1).unsqueeze(0).float(),
        (padding, padding, padding, padding),
        mode="reflect",
    )

    patches = image_padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    patches = patches.contiguous().view(3, image.size(0), image.size(1), -1)

    median_values = patches.median(dim=-1)[0]

    return median_values.permute(1, 2, 0).byte()


if __name__ == "__main__":
    # Test the blur functions
    import cv2

    image = cv2.imread("tests/富士山_resized.jpg")
    print(image.shape)
    image = torch.from_numpy(image).to(device="cuda")
    blur_limit = torch.tensor(7)
    blurred_image = _blur_image(image, blur_limit, torch.tensor(0.99))
    median_blurred_image = _median_blur_image(image, blur_limit, torch.tensor(0.99))
    blurred_image = blurred_image.cpu().numpy().astype(np.uint8)
    median_blurred_image = median_blurred_image.cpu().numpy().astype(np.uint8)
    cv2.imwrite("tests/富士山_blurred.jpg", blurred_image)
    cv2.imwrite("tests/富士山_median_blurred.jpg", median_blurred_image)
