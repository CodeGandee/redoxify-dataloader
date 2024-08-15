
import cv2
import torch
from typing import List, Union

import numpy as np
from mmcv.transforms import to_tensor
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmengine.structures import InstanceData
from mmengine.dataset.base_dataset import Compose
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import BaseBoxes

from typing import Union

from torch.utils.data import IterableDataset

from redoxify.readers import _READER_CLASS_MAP
from redoxify.transforms import _TRANSFORM_CLASS_MAP
from redoxify.datasets.RedoxBaseDataset import RedoxBaseDataset
from redoxify.pipelines.PipelineBuilder import (
    RedoxPipelineConfig, RedoxPipelineBuilder, RedoxDataIterator
)

class PackRedoxInputs(PackDetInputs):
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks'
    }
    def transform(self, results: dict) -> dict:
        if 'inputs' in results and 'data_samples' in results:
            return results
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if isinstance(img, torch.Tensor):
                img = img.permute(2, 0, 1).contiguous()
                
            elif isinstance(img, np.ndarray):
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if not img.flags.c_contiguous:
                    img = np.ascontiguousarray(img.transpose(2, 0, 1))
                    img = to_tensor(img)
                else:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
            else:
                raise TypeError('img should be torch.Tensor or np.ndarray, '
                                f'but got {type(img)}')
            packed_results['inputs'] = img

        if 'gt_ignore_flags' in results:
            gt_ignore_flags_results = results['gt_ignore_flags']
            if isinstance(gt_ignore_flags_results, torch.Tensor):
                valid_idx = torch.where(gt_ignore_flags_results == 0)[0]
                ignore_idx = torch.where(gt_ignore_flags_results == 1)[0]
            elif isinstance(gt_ignore_flags_results, np.ndarray):
                valid_idx = np.where(gt_ignore_flags_results == 0)[0]
                ignore_idx = np.where(gt_ignore_flags_results == 1)[0]
            else:
                raise TypeError('gt_ignore_flags should be torch.Tensor or np.ndarray, '
                                f'but got {type(gt_ignore_flags_results)}')

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()
        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                if 'gt_ignore_flags' in results:
                    instance_data[
                        self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[
                        self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key])
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str

class RedoxMMDetDataset(RedoxBaseDataset):
    @classmethod
    def from_redox_cfg(cls, redox_config : dict, num_gpus=1, device_id: Union[int, None]=0) -> 'RedoxBaseDataset':
        pipe_config = RedoxPipelineConfig(batch_size=redox_config['pipeline_cfg']['batch_size'],
                                        num_workers=redox_config['pipeline_cfg']['num_workers'],
                                        device_id=device_id)
        builder = RedoxPipelineBuilder(pipe_config)
        builder.set_output_map(redox_config['output_map'])
        reader = _READER_CLASS_MAP[redox_config['reader']['type']](**redox_config['reader'], num_gpus=num_gpus, device_id=device_id)
        builder.set_reader(reader)
        transform_sequence = []
        for transform_cfg in redox_config['transform_sequence']:
            transform = _TRANSFORM_CLASS_MAP[transform_cfg['type']](transform_cfg['config'], transform_cfg['inout_map'])
            transform_sequence.append(transform)
        builder.set_transform_sequence(transform_sequence)
        redox_pipeline = builder.build_pipeline()
        iterator = RedoxDataIterator.from_redox_pipelines([redox_pipeline])
        return cls(iterator, **redox_config['mm_config'])
    
    def __init__(self, 
                 redox_iterator: RedoxDataIterator,
                 normalized_bbox=True, 
                 mm_pipeline=None,
                 image_key="images",
                 bbox_key="bboxes",
                 label_key="labels",
                 ignore_flags=None,
                 mm_key_mapping={}, **kwargs):
        '''
        out_shape: 
                if out_shape is int, then the image will be resized to (out_shape, out_shape)
                if out_shape is tuple, it should be (height, width)
        '''
        self.redox_iterator = redox_iterator
        self.packer = PackRedoxInputs(meta_keys=(['img_shape']))
        self.normalized_bbox = normalized_bbox
        self.mm_transform = self.build_mm_transforms(mm_pipeline)
        self.image_key = image_key
        self.bbox_key = bbox_key
        self.label_key = label_key
        self.ignore_flags = ignore_flags
        self.mm_key_mapping = mm_key_mapping

    def build_mm_transforms(self, mm_pipeline):
        if mm_pipeline is not None and len(mm_pipeline) > 0:
            return Compose(mm_pipeline)
        else:
            return None
        
    def __iter__(self):
        for idx, data in enumerate(self.redox_iterator):
            yield self.postprocess(data)
    
    def apply_mm_transforms(self, data):
        if self.mm_transform is None:
            return data
        self.tensor_to_numpy(data)
        data = self.mm_transform(data)
        return data 
    
    def tensor_to_numpy(self, data):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = np.ascontiguousarray(value.cpu().numpy())
                
    def postprocess(self, data):
        assert data[0].tensors.get(self.image_key, None) is not None, f"Key {self.image_key} not found in data"
        assert data[0].tensors.get(self.bbox_key, None) is not None, f"Key {self.bbox_key} not found in data"
        images = data[0].tensors[self.image_key]
        bboxes = data[0].tensors[self.bbox_key]
        if data[0].tensors.get(self.ignore_flags, None) is not None:
            ignore_flags = data[0].tensors[self.ignore_flags]
        else:
            ignore_flags = [torch.zeros(len(box), dtype=torch.int32, device=box.device) for box in bboxes]
        images_shapes = data[0].shapes[self.image_key]
        data_batch = []
        for idx in range(len(images)):
            image = images[idx]
            img_height = images_shapes[idx][0]
            img_width = images_shapes[idx][1]
            bbox = bboxes[idx]
            if self.normalized_bbox:
                bbox = bbox * torch.tensor([img_width, img_height, img_width, img_height], dtype=bbox.dtype, device=bbox.device)
            result =dict(
                img=image,
                gt_bboxes=bbox,
                gt_bboxes_labels=data[0].tensors.get(self.label_key, None)[idx],
                gt_ignore_flags=ignore_flags[idx],
                img_shape=(img_height.item(), img_width.item()),
            )
            for key, value in self.mm_key_mapping.items():
                if key not in [self.image_key, self.bbox_key, self.label_key, ignore_flags]:
                    result[value] = data[0].tensors[key][idx]
            result = self.apply_mm_transforms(result)
            data_batch.append(self.packer.transform(result))
        return data_batch
    
    def update_skip_type_keys(self, skip_type_keys):
        pass