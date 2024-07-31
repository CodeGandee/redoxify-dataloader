import uuid
from typing import Dict, List
from attrs import define, field
import copy

import nvidia.dali.fn as fn
from nvidia.dali.tfrecord import Feature as TFRecordFeature

from redoxify.datablocks.DataBlock import DataBlock, DALINode
from redoxify.datablocks import (
    ImagesBlock as ImageDataBlock, ImageSpec,
    BoxesBlock as BoxDataBlock, BoxSpec,
    VectorsBlock as VectorDataBlock, VectorSpec,
    DataBlock
)

from redoxify.readers.ReaderSpecs import DataSpecType
import redoxify.GlobalConst as C


@define(kw_only=True, eq=False)
class TFReaderConfig:
    # define how to interpret each key in the tfrecord
    tf_feature_spec: Dict[str, TFRecordFeature] = field()
    
    # define how to convert each key to the data block
    datablock_spec: Dict[str, DataSpecType] = field()
    
    # if not given, use random uuid
    reader_name: str = field(default=str(uuid.uuid4()))
    
    # shuffle the data when reading
    random_shuffle: bool = field(default=False)
    
    def __attrs_post_init__(self):
        # tf_feature_spec and datablock_spec should have the same keys
        assert set(self.tf_feature_spec.keys()) == set(self.datablock_spec.keys()), "tf_feature_spec and datablock_spec should have the same keys"
    
@define(kw_only=True, eq=False)
class TFRecordFile:
    record_file : str = field()
    index_file : str = field()

class TFRecordReader:
    def __init__(self, tf_files : List[TFRecordFile], config : TFReaderConfig, num_gpus=1, device_id=0, **kwargs):
        tf_files = copy.deepcopy(tf_files)
        if device_id is not None and len(tf_files)%num_gpus == 0:
            tf_files = tf_files[device_id::num_gpus]
        self.m_files = tf_files
        self.m_config = config
        
    def read_raw(self) -> DALINode:
        record_list = [file.record_file for file in self.m_files]
        index_list = [file.index_file for file in self.m_files]
        
        out = fn.readers.tfrecord(
            path=record_list, index_path=index_list, 
            features=self.m_config.tf_feature_spec,
            name=self.m_config.reader_name,
            random_shuffle=self.m_config.random_shuffle,
            device='cpu')
        return out
    
    def read_as_datablocks(self) -> Dict[str, DataBlock]:
        raw_data = self.read_raw()
        out = {}
        for key in self.m_config.datablock_spec.keys():
            # copy the spec because we might need to fill information in it
            spec = self.m_config.datablock_spec[key].clone()
            
            if isinstance(spec, ImageSpec):
                x = ImageDataBlock()
                # fill size information in cpu for later use
                spec.set_size_by_image_data(raw_data[key], skip_if_set=True)
                spec.encoding = 'jpg'
                spec.data_layout = 'hwc'
                x.add_data(key=C.DefaultDatablockKey, data=raw_data[key], spec=spec)
                out[key] = x
                
            elif isinstance(spec, BoxSpec):
                x = BoxDataBlock()
                x.add_data(key=C.DefaultDatablockKey, data=raw_data[key], spec=spec)
                out[key] = x
            elif isinstance(spec, VectorSpec):
                x = VectorDataBlock()
                x.add_data(key=C.DefaultDatablockKey, data=raw_data[key], spec=spec)
                out[key] = x
        return out