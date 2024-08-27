import torch
from kornia.color.hsv import rgb_to_hsv, hsv_to_rgb
from nvidia.dali.plugin.pytorch.fn import torch_python_function
from nvidia.dali.pipeline import DataNode as DALINode


def dali_hsv_image(
    image: DALINode,
    hue_multiplier: DALINode,
    saturation_multiplier: DALINode,
    value_multiplier: DALINode,
):
    """
    Apply HSV transformation to the input image using DALI.

    Args:
        image (DALINode): The input image.
        hue_multiplier (DALINode): The multiplier for the hue channel.
        saturation_multiplier (DALINode): The multiplier for the saturation channel.
        value_multiplier (DALINode): The multiplier for the value channel.

    Returns:
        DALINode: The transformed image.
    """
    func = torch_python_function(
        image,
        hue_multiplier,
        saturation_multiplier,
        value_multiplier,
        function=_hsv_image,
        batch_processing=False,
        output_layouts=["HWC"],
        num_outputs=1,
        device="gpu",
    )
    return func


def _hsv_image(
    img: torch.Tensor,
    hue_multiplier: torch.Tensor,
    saturation_multiplier: torch.Tensor,
    value_multiplier: torch.Tensor,
):
    """
    Applies HSV transformation to an image tensor.

    Args:
        img (torch.Tensor): The input image tensor.
        hue_multiplier (torch.Tensor): The hue multiplier tensor.
        saturation_multiplier (torch.Tensor): The saturation multiplier tensor.
        value_multiplier (torch.Tensor): The value multiplier tensor.

    Returns:
        torch.Tensor: The transformed image tensor.
    """
    img_rgb = img.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_hsv = rgb_to_hsv(img_rgb)
    img_hsv[:, 0, :, :] = (img_hsv[:, 0, :, :] * hue_multiplier) % (2 * torch.pi)
    img_hsv[:, 1, :, :] = (img_hsv[:, 1, :, :] * saturation_multiplier).clamp(0, 1.0)
    img_hsv[:, 2, :, :] = (img_hsv[:, 2, :, :] * value_multiplier).clamp(0, 1.0)
    img_rgb = hsv_to_rgb(img_hsv)
    img_rgb = (img_rgb * 255.0).byte()
    return img_rgb.squeeze(0).permute(1, 2, 0)


if __name__ == "__main__":
    # Test the blur functions
    import cv2
    import numpy as np

    image = cv2.imread("tests/fuji.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).to(device="cuda")
    image2 = _hsv_image(image, torch.tensor(1.4), torch.tensor(1.0), torch.tensor(1.0))
    image2 = image2.cpu().numpy()
    cv2.imwrite("temp/hsv_kor2.jpg", cv2.cvtColor(image2, cv2.COLOR_RGB2BGR))
