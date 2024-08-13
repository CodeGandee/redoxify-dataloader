from typing import Any, Mapping, Sequence
import torch

def pseudo_collate(batch: Sequence) -> Any:
    assert len(batch)==1, "Batch size of dataloader should be 1 if you are using RedoxDetDataset"
    keys = batch[0][0].keys()
    data_batch = {key: [data[key] for data in batch[0]] for key in keys}
    return data_batch

def yolov5_collate(data_batch: Sequence,
                   use_ms_training: bool = False) -> dict:
    """Rewrite collate_fn to get faster training speed.

    Args:
       data_batch (Sequence): Batch of data.
       use_ms_training (bool): Whether to use multi-scale training.
    """
    data_batch = data_batch[0]
    batch_imgs = []
    batch_bboxes_labels = []
    batch_masks = []
    batch_keyponits = []
    batch_keypoints_visible = []
    for i in range(len(data_batch)):
        datasamples = data_batch[i]['data_samples']
        inputs = data_batch[i]['inputs']
        batch_imgs.append(inputs)

        gt_bboxes = datasamples.gt_instances.bboxes
        gt_labels = datasamples.gt_instances.labels
        if 'masks' in datasamples.gt_instances:
            masks = datasamples.gt_instances.masks
            batch_masks.append(masks)
        if 'gt_panoptic_seg' in datasamples:
            batch_masks.append(datasamples.gt_panoptic_seg.pan_seg)
        if 'keypoints' in datasamples.gt_instances:
            keypoints = datasamples.gt_instances.keypoints
            keypoints_visible = datasamples.gt_instances.keypoints_visible
            batch_keyponits.append(keypoints)
            batch_keypoints_visible.append(keypoints_visible)

        batch_idx = gt_labels.new_full((len(gt_labels), 1), i)
        bboxes_labels = torch.cat((batch_idx, gt_labels[:, None], gt_bboxes),
                                  dim=1)
        batch_bboxes_labels.append(bboxes_labels)
    collated_results = {
        'data_samples': {
            'bboxes_labels': torch.cat(batch_bboxes_labels, 0)
        }
    }
    if len(batch_masks) > 0:
        collated_results['data_samples']['masks'] = torch.cat(batch_masks, 0)

    if len(batch_keyponits) > 0:
        collated_results['data_samples']['keypoints'] = torch.cat(
            batch_keyponits, 0)
        collated_results['data_samples']['keypoints_visible'] = torch.cat(
            batch_keypoints_visible, 0)

    if use_ms_training:
        collated_results['inputs'] = batch_imgs
    else:
        collated_results['inputs'] = torch.stack(batch_imgs, 0)
    return collated_results
