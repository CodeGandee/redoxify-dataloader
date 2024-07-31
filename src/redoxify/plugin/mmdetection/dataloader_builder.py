
import cv2
import copy

import numpy as np

import torch
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader

from redoxify.readers import TFRecordReader
from redoxify.transforms.RandomCropWithBoxes import RandomCropWithBoxes 
from redoxify.transforms.Resize import Resize 
from redoxify.transforms.Pad import Pad
from redoxify.transforms.Mosaic import Mosaic
from redoxify.transforms.RandomSingleDirectionFlip import RandomSingleDirectionFlip
from redoxify.transforms.RandomHSVAug import RandomHSVAug
from redoxify.transforms.DataBlockGroupingTransform import DataBlockGroupingTransform
from redoxify.pipelines.PipelineBuilder import (
    RedoxPipelineConfig, RedoxPipelineBuilder, RedoxDataIterator
)

from coco_dataset_cfg import cfg_dict

import numpy as np
from mmcv.transforms import to_tensor
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmengine.structures import InstanceData
from mmengine.dataset.base_dataset import Compose
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import BaseBoxes

class PackDALIDetInputs(PackDetInputs):
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
    }
    def transform(self, results: dict) -> dict:
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
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

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
        # data_sample.ignored_instances = ignore_instance_data

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

class RedoxDetDataset(IterableDataset):
    def __init__(self, redox_iterator, 
                 normalize_bbox=True, 
                 mm_pipeline=None,
                 image_key="images",
                 bbox_key="bboxes",
                 key_mappping={}):
        '''
        out_shape: 
                if out_shape is int, then the image will be resized to (out_shape, out_shape)
                if out_shape is tuple, it should be (height, width)
        '''
        self.redox_iterator = redox_iterator
        self.packer = PackDALIDetInputs(meta_keys=(['img_shape']))
        self.normalize_bbox = normalize_bbox
        self.mm_transform = self.build_mm_transforms(mm_pipeline)
        self.image_key = image_key
        self.bbox_key = bbox_key
        self.key_mappping = key_mappping

    
    def build_mm_transforms(self, mm_pipeline):
        if mm_pipeline is not None:
            return Compose(mm_pipeline)
        else:
            return None
        
    @classmethod
    def _load_metainfo(cls) -> dict:
        cls_metainfo = copy.deepcopy(cls.METAINFO)
        return cls_metainfo
    
    @property
    def metainfo(self) -> dict:
        return copy.deepcopy(self._metainfo)
    
    def __iter__(self):
        for data in self.redox_iterator:
            yield self.preprocess(data)
    
    def __len__(self):
        return self.redox_iterator.size//self.redox_iterator.batch_size
    
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
                # data[key] = value.cpu().numpy()
    
    def update_skip_type_keys(self, skip_type_keys):
        return      
       
    def preprocess(self, data):
        images = data[0].tensors[self.image_key]
        bboxes = data[0].tensors[self.bbox_key]
        images_shapes = data[0].shapes[self.image_key]
        data_batch = []
        for idx in range(len(images)):
            image = images[idx]
            img_height = images_shapes[idx][0]
            img_width = images_shapes[idx][1]
            bbox = bboxes[idx]
            if self.normalize_bbox:
                bbox = bbox * torch.tensor([img_width, img_height, img_width, img_height], dtype=bbox.dtype, device=bbox.device)
            result =dict(
                img=image,
                gt_bboxes=bbox,
                img_shape=(img_height.item(), img_width.item()),
            )
            for key, value in self.key_mappping.items():
                if key != "images" and key != "bboxes":
                    result[value] = data[0].tensors[key][idx]
            if result.get("gt_ignore_flags") is None:
                result["gt_ignore_flags"] = np.zeros(len(result["gt_bboxes"]), dtype=np.int64)
            result = self.apply_mm_transforms(result)
            data_batch.append(result)
        return data_batch
        
def redox_data_collate_fn(batch):
    assert len(batch)==1, "Batch size of dataloader should be 1 if you are using RedoxDetDataset"
    keys = batch[0][0].keys()
    data_batch = {key: [data[key] for data in batch[0]] for key in keys}
    return data_batch

def build_dataloader_from_cfg(cfg, world_size=1, device_id=0):
    pipe_config = RedoxPipelineConfig(batch_size=cfg['pipeline_cfg']['batch_size'],
                                      num_workers=cfg['pipeline_cfg']['num_workers'],
                                      device_id=device_id)

    builder = RedoxPipelineBuilder(pipe_config)
    builder.set_output_map(cfg['output_map'])
    tf_files = copy.deepcopy(cfg['reader']['tf_files'])
    if not cfg['reader'].get('do_not_split_tfrec', False) and len(tf_files)%world_size == 0:
        tf_files = tf_files[device_id::world_size]
    reader = TFRecordReader(tf_files, cfg['reader']['reader_cfg'])
    builder.set_reader(reader)
    transform_sequence = []
    for transform_cfg in cfg['transform_sequence']:
        if transform_cfg['type'] == 'RandomCropWithBoxes':
            transform = RandomCropWithBoxes(transform_cfg['config'], transform_cfg['inout_map'])
        elif transform_cfg['type'] == 'Resize':
            transform = Resize(transform_cfg['config'], transform_cfg['inout_map'])
        elif transform_cfg['type'] == 'Pad':
            transform = Pad(transform_cfg['config'], transform_cfg['inout_map'])
        elif transform_cfg['type'] == 'RandomSingleDirectionFlip':
            transform = RandomSingleDirectionFlip(transform_cfg['config'], transform_cfg['inout_map'])
        elif transform_cfg['type'] == 'RandomHSVAug':
            transform = RandomHSVAug(transform_cfg['config'], transform_cfg['inout_map'])
        elif transform_cfg['type'] == 'DataBlockGroupingTransform':
            transform = DataBlockGroupingTransform(transform_cfg['config'], transform_cfg['inout_map'])
        elif transform_cfg['type'] == 'Mosaic':
            transform = Mosaic(transform_cfg['config'], transform_cfg['inout_map'])
        else:
            raise ValueError(f"Unknown transform type: {transform_cfg['type']}")
        transform_sequence.append(transform)
    builder.set_transform_sequence(transform_sequence)
    rpipeline = builder.build_pipeline()
    iterator = RedoxDataIterator.from_redox_pipelines([rpipeline])
    dataset = RedoxDetDataset(iterator, mm_pipeline=cfg.get('mm_pipeline', None), image_key=cfg['image_key'], bbox_key=cfg.get('bbox_key', None), key_mappping=cfg.get('mm_key_mapping', {}))
    dataloader = DataLoader(dataset, collate_fn=redox_data_collate_fn)
    return dataloader

def build_redoxiterator_from_cfg(cfg, world_size=1, device_id=0):
    pipe_config = RedoxPipelineConfig(batch_size=cfg['pipeline_cfg']['batch_size'],
                                      num_workers=cfg['pipeline_cfg']['num_workers'],
                                      device_id=device_id)

    builder = RedoxPipelineBuilder(pipe_config)
    builder.set_output_map(cfg['output_map'])
    tf_files = copy.deepcopy(cfg['reader']['tf_files'])
    if not cfg['reader'].get('do_not_split_tfrec', False) and len(tf_files)%world_size == 0:
        tf_files = tf_files[device_id::world_size]
    reader = TFRecordReader(tf_files, cfg['reader']['reader_cfg'])
    builder.set_reader(reader)
    transform_sequence = []
    for transform_cfg in cfg['transform_sequence']:
        if transform_cfg['type'] == 'RandomCropWithBoxes':
            transform = RandomCropWithBoxes(transform_cfg['config'], transform_cfg['inout_map'])
        elif transform_cfg['type'] == 'Resize':
            transform = Resize(transform_cfg['config'], transform_cfg['inout_map'])
        elif transform_cfg['type'] == 'Pad':
            transform = Pad(transform_cfg['config'], transform_cfg['inout_map'])
        elif transform_cfg['type'] == 'RandomSingleDirectionFlip':
            transform = RandomSingleDirectionFlip(transform_cfg['config'], transform_cfg['inout_map'])
        elif transform_cfg['type'] == 'RandomHSVAug':
            transform = RandomHSVAug(transform_cfg['config'], transform_cfg['inout_map'])
        elif transform_cfg['type'] == 'DataBlockGroupingTransform':
            transform = DataBlockGroupingTransform(transform_cfg['config'], transform_cfg['inout_map'])
        elif transform_cfg['type'] == 'Mosaic':
            transform = Mosaic(transform_cfg['config'], transform_cfg['inout_map'])
        else:
            raise ValueError(f"Unknown transform type: {transform_cfg['type']}")
        transform_sequence.append(transform)
    builder.set_transform_sequence(transform_sequence)
    rpipeline = builder.build_pipeline()
    iterator = RedoxDataIterator.from_redox_pipelines([rpipeline])
    return iterator


if __name__ == "__main__":
    def save_labeled_image(img: torch.Tensor, boxes:torch.Tensor, save_path:str):
        # draw and save by cv2
        img = img.cpu().numpy()
        boxes = boxes.cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1*img.shape[1]), int(y1*img.shape[0]), int(x2*img.shape[1]), int(y2*img.shape[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(save_path, img)
        
    loader = build_dataloader_from_cfg(cfg_dict, device_id=0)


    for idx, data_batch in enumerate(loader):
        print(111)

