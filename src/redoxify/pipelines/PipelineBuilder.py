# build data loading pipeline with transforms
from typing import Dict, List, Union
from attrs import define, field
from rich import print as pprint

from nvidia.dali.pipeline import (
    Pipeline as DALI_Pipeline, 
    pipeline_def as DALI_pipeline_def,
)
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import do_not_convert

import torch

import redoxify.transforms as T
import redoxify.datablocks as B
from redoxify.RedoxTypes import DataKey
from redoxify.readers import (
    TFRecordReader, TFReaderConfig, TFRecordFile
)

# required input for pipeline builder
# 1. tf-record file / reader object
# 2. output datablocks and their keys, order
# 3. dali pipeline configuration
# 4. transforms to apply


@define(kw_only=True, eq=False)
class RedoxPipelineConfig:
    batch_size : int = field()
    num_workers : int = field()
    device_id : int = field(default=0)
    seed : int = field(default=-1)
    _enable_conditionals : bool = field(default=True)
    
@define(kw_only=True, eq=False)
class SingleOutputSpec:
    # key in the output data of the last transform
    # if sub_key is None, it is assumed that the datablock only has one sub_key
    input_key : DataKey = field()
    
    # should we use DALI pad to ensure uniform tensor shape inside a batch?
    # cases as follows:
    # - pad_for_batch = False, no padding is applied to the tensor, you have to ensure different samples having the same shape
    # - pad_for_batch = True, padding is applied to the tensor so that they have the same shape, with pad_fill_value filled into the sample
    # - split_batch_into_list = True
    #       the output will be a list of tensors, each tensor is a sample in the batch.
    #       if the tensor is padded, it will be trimmed to the original shape
    # - split_batch_into_list = False, 
    #       the output will be a single tensor, with all samples in the batch stacked together, 
    #       if the tensor is padded, it remains as such, and the output data contains shape data
    pad_for_batch : bool = field()
    pad_fill_value : float = field(default=0.0)
    split_batch_into_list : bool = field(default=False)
    
    # if input_key.sub_key is None, then stack all sub_keys of the main_key into a tensor
    stack_all_sub_keys : bool = field(default=False)
    
    # for things that have normalized/absolute coordinates, which one should we use?
    # True - output normalized coords
    # False - output absolute coords
    # None - do not care, keep the same as input
    use_normalized_coords : Union[bool, None] = field(default=None)
    
@define(kw_only=True, eq=False)
class PipelineTransformContext(T.TransformContext):
    previous_transforms : List[T.TransformParams] = field(factory=list)
    
@define(kw_only=True, eq=False)
class RedoxIteratorOutput:
    tensors : Dict[str, torch.Tensor] = field(factory=dict)
    shapes : Dict[str, torch.Tensor] = field(factory=dict)
    
@define(kw_only=True, eq=False)
class RedoxPipelineData:
    dali_pipeline : DALI_Pipeline = field()
    pipeline_config: RedoxPipelineConfig = field()
    output_map : Dict[str, SingleOutputSpec] = field()
    reader_name : Union[str, None] = field(default=None)
    
class RedoxPipelineBuilder:
    def __init__(self, config : RedoxPipelineConfig) -> None:
        self.m_config = config
        self.m_reader : TFRecordReader = None
        self.m_transforms : List[T.BaseTransform] = []
        self.m_output_map : Dict[str, SingleOutputSpec] = {}
        
    def set_output_map(self, output_map : Dict[str, SingleOutputSpec]):
        self.m_output_map = output_map
        pass
    
    def set_transform_sequence(self, transforms : List[T.BaseTransform]):
        self.m_transforms = transforms
    
    def set_reader(self, reader : TFRecordReader):
        self.m_reader = reader
    
    def build_pipeline(self) -> RedoxPipelineData:
        p = self._build_pipeline()
        return RedoxPipelineData(dali_pipeline=p, 
                                 pipeline_config=self.m_config,
                                 output_map=self.m_output_map,
                                 reader_name=self.m_reader.m_config.reader_name)
        
    def _build_pipeline(self) -> DALI_Pipeline:
        @DALI_pipeline_def(batch_size=self.m_config.batch_size, 
                           num_threads=self.m_config.num_workers, 
                           device_id=self.m_config.device_id, 
                           seed=self.m_config.seed,
                           enable_conditionals=self.m_config._enable_conditionals)
        def dali_pipe():
            result = _dali_pipe()
            return result
        
        # this is a hack to prevent dali from converting the function to a pipeline, for debugging purposes
        # to use this, you have to remove all branches that uses DALINode as conditionals
        
        # @do_not_convert
        def _dali_pipe():
            # read data
            results = self.m_reader.read_as_datablocks()
            
            # apply transforms
            context = PipelineTransformContext()
            
            if self.m_transforms is not None:
                for t in self.m_transforms:
                    input_data = T.TransformInput(data_blocks=results)
                    output_data = t.do_transform(input_data=input_data, context=context)
                    context.previous_transforms.append(output_data.params)
                    results = output_data.data_blocks
                    
            # map to output
            output_data = []
            output_shape = []
            for key, spec in self.m_output_map.items():
                main_key = spec.input_key.main_key
                sub_key = spec.input_key.sub_key
                block = results[main_key]
                if isinstance(block, B.ImagesBlock):
                    _data = block.get_decoded_tensor(sub_key)
                elif isinstance(block, B.BoxesBlock):
                    if spec.use_normalized_coords is True:
                        _data = block.get_boxes_normalized(sub_key)
                    elif spec.use_normalized_coords is False:
                        raise NotImplementedError('Absolute coordinates not implemented yet')
                    else:
                        _data = block.get_boxes(as_Nx4=True, key = sub_key)
                elif isinstance(block, B.VectorsBlock):
                    _data = block.get_vector(sub_key)
                else:
                    raise ValueError(f"Unknown datablock type {type(block)}")
                
                original_shape = fn.shapes(_data)
                
                if spec.pad_for_batch:
                    _data = fn.pad(_data, fill_value=spec.pad_fill_value)
                    
                output_data.append(_data.gpu())
                output_shape.append(original_shape.gpu())
            
            # output them
            final_output = output_data + output_shape
            return tuple(final_output)
        
        p = dali_pipe()
        return p
    
@define(kw_only=True, eq=False)
class RedoxIteratorConfig:
    reader_name : str = field(default=None)
    auto_reset : bool = field(default=True)
    
class RedoxDataIterator(DALIGenericIterator):
    @classmethod
    def from_redox_pipelines(cls, 
                             pipeline_data : List[RedoxPipelineData],
                             pipeline_output_map : Dict[str, SingleOutputSpec] = None,
                             config : RedoxIteratorConfig = None) -> 'RedoxDataIterator':
        if config is None:
            config = RedoxIteratorConfig()
        pipelines = [x.dali_pipeline for x in pipeline_data]
        
        # FIXME: make sure all pipelines has the same output map
        
        # by default, use the first pipeline's output map
        if pipeline_output_map is None:
            pipeline_output_map = pipeline_data[0].output_map
        
        # FIXME: should be respect reader_name==None ?
        if config.reader_name is None:
            config.reader_name = pipeline_data[0].reader_name
            
        return cls(pipelines=pipelines, 
                   pipeline_output_map = pipeline_output_map,
                   config=config)
    
    def __init__(self, pipelines : List[DALI_Pipeline], 
                 pipeline_output_map : Dict[str, SingleOutputSpec],
                 config : RedoxIteratorConfig = None):
        
        assert isinstance(pipelines, list), "Pipelines must be a list"
        
        if config is None:
            config = RedoxIteratorConfig()
        
        # data outputs
        data_output_map = list(pipeline_output_map.keys())
        
        # shape outputs
        shape_output_map = [f"{k}.shape" for k in data_output_map]
        dali_output_map = data_output_map + shape_output_map
        super().__init__(
            pipelines=pipelines,
            output_map=dali_output_map,
            reader_name=config.reader_name,
            auto_reset=False,
            # auto_reset=config.auto_reset,
        )
        
        self.m_output_map = pipeline_output_map
        self.m_pipelines = pipelines
        self.m_config = config
        
        self.m_data_keys : List[str] = data_output_map
        self.m_shape_keys : List[str] = shape_output_map
    
    def __next__(self) -> List[RedoxIteratorOutput]:
        # TODO: test this
        parent_data : List = super().__next__()
        output = []
        for sample in parent_data:
            this_output = RedoxIteratorOutput()
            output.append(this_output)
            for data_key, shape_key in zip(self.m_data_keys, self.m_shape_keys):
                spec = self.m_output_map[data_key]
                data : torch.Tensor = sample[data_key]
                shape : torch.Tensor = sample[shape_key]
                
                # unpadding
                # slice data
                if spec.pad_for_batch:  # padded
                    if spec.split_batch_into_list:
                        _data : List[torch.Tensor] = []
                        
                        # unpad each sample in a batch
                        for x_data, x_shape in zip(data, shape):
                            # repeated narrows to unpad
                            
                            # if shape is already the same, no need to unpad
                            if x_data.shape == x_shape:
                                _data.append(x_data)
                            else:
                                assert len(x_data.shape) == len(x_shape), "dimension mismatch"
                                
                                # for each dimension, narrow the tensor
                                # equivalent to x_data[:x_shape[0], :x_shape[1], :x_shape[2], ...]
                                for ith_dim in range(len(x_shape)):
                                    x_data = torch.narrow(x_data, dim=ith_dim, start=0, length=x_shape[ith_dim])
                                _data.append(x_data)
                    else:   # cannot unpad, user handles unpadding themselves
                        _data = data
                else: #no unpadding
                    if spec.split_batch_into_list:
                        _data = [x for x in data]
                    else:   # no need to do anything
                        _data = data
                this_output.tensors[data_key] = _data
                this_output.shapes[data_key] = shape
        return output
                
                
                
        