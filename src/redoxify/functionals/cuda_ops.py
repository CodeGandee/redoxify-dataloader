# functions used in dali python_function
import torch
import random
import numpy as np

from nvidia.dali.plugin.pytorch.fn import torch_python_function
from nvidia.dali.pipeline import DataNode as DALINode

def dali_mosaic_images(images: DALINode, boxes: DALINode, labels: DALINode):
    '''
    gather the data by indices
    '''
    func = torch_python_function(
        images, boxes, labels,
        function=_dali_mosaic_images,
        batch_processing=True,
        num_outputs=3,
        device='gpu'
    )
    return func

def _dali_mosaic_images(images: list[torch.Tensor], boxes: list[torch.Tensor], labels: list[torch.Tensor]):
    """Mosaic augmentation.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.
    """
    # images: list of N images, all of them have the same shape. assert the shape
    shapes = [img.shape for img in images]
    assert len(set(shapes)) == 1, "All images should have the same shape in mosaic transform"
    image_shape = shapes[0]
    mosaic_shape = (image_shape[0] * 2, image_shape[1] * 2, image_shape[2])
    num_images = len(images)
    mosaics = [torch.zeros(mosaic_shape, dtype=images[0].dtype, device=images[0].device) for _ in range(num_images)]
    if num_images <= 1:
        indices = list(range(num_images)) * 4
    elif num_images < 4:
        indices = list(range(num_images)) * 2
    else:
        indices = list(range(num_images))
    mosaic_boxes = []
    mosaic_labels = []
    offsets = torch.tensor([[0., 0., 0., 0.], [1.0, 0., 1.0, 0.], [0., 1.0, 0., 1.0], [1.0, 1.0, 1.0, 1.0]], device=images[0].device)
    for mosaic in mosaics:
        selected_indices = random.sample(indices, 4)
        mosaic[:image_shape[0], :image_shape[1]] = images[selected_indices[0]]
        mosaic[:image_shape[0], image_shape[1]:] = images[selected_indices[1]]
        mosaic[image_shape[0]:, :image_shape[1]] = images[selected_indices[2]]
        mosaic[image_shape[0]:, image_shape[1]:] = images[selected_indices[3]]
        mosaic_boxes.append(torch.cat([(boxes[i]+offsets[off_idx])/2 for off_idx, i in enumerate(selected_indices)], dim=0))
        mosaic_labels.append(torch.cat([labels[i] for i in selected_indices], dim=0))
    return mosaics, mosaic_boxes, mosaic_labels


def dali_mixup_images(images: DALINode, boxes: DALINode, labels: DALINode, ratioes: DALINode):
    '''
    gather the data by indices
    '''
    func = torch_python_function(
        images, boxes, labels, ratioes,
        function=_dali_mixup_images,
        batch_processing=True,
        num_outputs=3,
        device='gpu'
    )
    return func

def _dali_mixup_images(images: list[torch.Tensor], boxes: list[torch.Tensor], labels: list[torch.Tensor], ratioes: list[torch.Tensor]):
    """Mosaic augmentation.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.
    """
    # images: list of N images, all of them have the same shape. assert the shape
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