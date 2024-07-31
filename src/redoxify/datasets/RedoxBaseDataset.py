
import cv2
import copy

from typing import Union

from torch.utils.data import IterableDataset

from redoxify.readers import _READER_CLASS_MAP
from redoxify.transforms import _TRANSFORM_CLASS_MAP

from redoxify.pipelines.PipelineBuilder import (
    RedoxPipelineConfig, RedoxPipelineBuilder, RedoxDataIterator
)

class RedoxBaseDataset(IterableDataset):
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
        return cls(iterator)
        
    def __init__(self, 
                 redox_iterator: RedoxDataIterator):
        self.redox_iterator = redox_iterator

    def __iter__(self):
        for data in self.redox_iterator:
            yield self.postprocess(data)
    
    def __len__(self):
        return self.redox_iterator.size//self.redox_iterator.batch_size

    def postprocess(self, data):
        # Not implemented
        return data