# functions used in dali python_function
import torch
import random
import numpy as np

from nvidia.dali.plugin.pytorch.fn import torch_python_function
from nvidia.dali.pipeline import DataNode as DALINode

def dali_mosaic_images(images: DALINode, boxes: DALINode, labels: DALINode, probabilities: DALINode):
    func = torch_python_function(
        images, boxes, labels, probabilities,
        function=_dali_mosaic_images,
        batch_processing=True,
        num_outputs=3,
        device='gpu'
    )
    return func

def _dali_mosaic_images(images_list: list[torch.Tensor], boxes_list: list[torch.Tensor], 
                        labels_list: list[torch.Tensor], probabilities_list: list[torch.Tensor]):
    """Mosaic augmentation.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.
    """
    # images: list of N images, all of them have the same shape. assert the shape
    shapes = [img.shape for img in images_list]
    assert len(set(shapes)) == 1, "All images should have the same shape in mosaic transform"
    image_shape = shapes[0]
    mosaic_shape = (image_shape[0] * 2, image_shape[1] * 2, image_shape[2])
    num_images = len(images_list)
    mosaics = [torch.zeros(mosaic_shape, dtype=images_list[0].dtype, device=images_list[0].device) for _ in range(num_images)]
    if num_images <= 1:
        indices = list(range(num_images)) * 4
    elif num_images < 4:
        indices = list(range(num_images)) * 2
    else:
        indices = list(range(num_images))
    mosaic_boxes = []
    mosaic_labels = []
    offsets = torch.tensor([[0., 0., 0., 0.], [1.0, 0., 1.0, 0.], [0., 1.0, 0., 1.0], [1.0, 1.0, 1.0, 1.0]], device=images_list[0].device)
    for mosaic in mosaics:
        selected_indices = random.sample(indices, 4)
        mosaic[:image_shape[0], :image_shape[1]] = images_list[selected_indices[0]]
        mosaic[:image_shape[0], image_shape[1]:] = images_list[selected_indices[1]]
        mosaic[image_shape[0]:, :image_shape[1]] = images_list[selected_indices[2]]
        mosaic[image_shape[0]:, image_shape[1]:] = images_list[selected_indices[3]]
        boxes_to_cat = []
        labels_to_cat = []
        for off_idx, i in enumerate(selected_indices):
            if boxes_list[i].shape[1] != 0:
                boxes_to_cat.append((boxes_list[i]+offsets[off_idx])/2)
                labels_to_cat.append(labels_list[i])
        if len(boxes_to_cat) == 0:
            mosaic_boxes.append(torch.zeros((0, 4), device=images_list[0].device, dtype=boxes_list[0].dtype))
            mosaic_labels.append(torch.zeros((0,), device=images_list[0].device, dtype=labels_list[0].dtype))
        else:
            mosaic_boxes.append(torch.cat(boxes_to_cat, dim=0))
            mosaic_labels.append(torch.cat(labels_to_cat, dim=0))
    for img_idx in range(len(images_list)):
        probability = probabilities_list[img_idx]
        if torch.rand(probability.shape, device=probability.device) > probability:
            print(images_list)
            mosaics[img_idx] = images_list[img_idx]
            mosaic_boxes[img_idx] = boxes_list[img_idx]
            mosaic_labels[img_idx] = labels_list[img_idx]
    return mosaics, mosaic_boxes, mosaic_labels


def dali_mixup_images(images: DALINode, boxes: DALINode, labels: DALINode, ratioes: DALINode):
    func = torch_python_function(
        images, boxes, labels, ratioes,
        function=_dali_mixup_images,
        batch_processing=True,
        num_outputs=3,
        device='gpu'
    )
    return func

def _dali_mixup_images(images: list[torch.Tensor], boxes: list[torch.Tensor], labels: list[torch.Tensor], ratioes: list[torch.Tensor]):
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




#for debugging
if __name__ == '__main__':
    images = [torch.rand((3, 224, 224), device='cuda') for _ in range(4)]
    boxes = [torch.rand((4, 0), device='cuda') for _ in range(3)] + [torch.rand((4, 0), device='cuda')]
    labels = [torch.randint(0, 10, (0,), device='cuda') for _ in range(4)]
    probabilities = [torch.tensor(0.5, device='cuda') for _ in range(4)]
    mosaics, mosaic_boxes, mosaic_labels = _dali_mosaic_images(images, boxes, labels, probabilities)
    print(mosaics, mosaic_boxes, mosaic_labels)