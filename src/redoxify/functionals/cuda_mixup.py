# functions used in dali python_function
import torch
import random
import numpy as np
from typing import List

from nvidia.dali.plugin.pytorch.fn import torch_python_function
from nvidia.dali.pipeline import DataNode as DALINode

def dali_mixup_images(images: DALINode, boxes: DALINode, labels: DALINode, ratioes: DALINode):
    func = torch_python_function(
        images,
        boxes,
        labels,
        ratioes,
        function=_dali_mixup_images,
        batch_processing=True,
        num_outputs=3,
        output_layouts=["HWC"],
        device="gpu",
    )
    return func

def _dali_mixup_images(images: List[torch.Tensor], boxes: List[torch.Tensor], labels: List[torch.Tensor], ratioes: List[torch.Tensor]):
    shapes = [img.shape for img in images]
    assert len(set(shapes)) == 1, "All images should have the same shape in mixup transform"
    num_images = len(images)
    assert num_images >= 2, "Mixup transform requires at least 2 images"
    indices = list(range(num_images))
    mxiup_images = []
    mixup_boxes = []
    mixup_labels = []
    for idx in range(num_images):
        ratio = ratioes[idx]
        selected_indices = random.sample(indices, 2)
        mxiup_image = images[selected_indices[0]] * ratio + images[selected_indices[1]] * (1 - ratio)
        mxiup_images.append(mxiup_image)
        mixup_boxes.append(torch.cat([boxes[i] for i in selected_indices], dim=0))
        mixup_labels.append(torch.cat([labels[i] for i in selected_indices], dim=0))
    return mxiup_images, mixup_boxes, mixup_labels
