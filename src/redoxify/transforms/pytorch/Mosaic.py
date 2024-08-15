from typing import Union, Dict, List
from attrs import define, field
import attrs.validators as av
import copy

import nvidia.dali.fn as fn
from nvidia.dali.types import DALIDataType

from redoxify.datablocks.DataBlock import DataBlock, DALINode
from redoxify.functionals.cuda_ops import dali_mosaic_images
from redoxify.RedoxTypes import DataKey

from redoxify.datablocks.ImagesBlock import (
    ImagesBlock, ImageSpec )

from redoxify.datablocks.BoxesBlock import (
    BoxesBlock, BoxSpec )

from redoxify.datablocks.VectorsBlock import (
    VectorsBlock, VectorSpec )

from redoxify.transforms.BaseTransform import (
    BaseTransform, 
    TransformParams,
    TransformInput,
    TransformOutput,
    TransformContext
)

@define(kw_only=True, eq=False)
class ImageMosaicSetting:
    image_key : DataKey = field()
    output_key : DataKey = field()

@define(kw_only=True, eq=False)
class BoxMosaicSetting:
    box_key : DataKey = field()
    output_key : DataKey = field()

@define(kw_only=True, eq=False)
class LabelMosaicSetting:
    label_key : DataKey = field()
    output_key : DataKey = field()

@define(kw_only=True, eq=False)
class MosaicInputOutputMap:
    image_mosaic_settings : List[ImageMosaicSetting] = field(factory=list)
    box_mosaic_settings : List[BoxMosaicSetting] = field(factory=list)
    label_mosaic_settings : List[LabelMosaicSetting] = field(factory=list)
    
    
@define(kw_only=True, eq=False)
class MosaicConfig:
    mosaic_height : Union[int, float, DALINode] = field(validator=av.instance_of((int, float, DALINode)))
    mosaic_width : Union[int, float, DALINode] = field(validator=av.instance_of((int, float, DALINode)))
    fill_values : Union[int, float, DALINode, List[int], List[float]] = field(validator=av.instance_of((int, float, DALINode, List)))
    probability : float = field(default=0.5, validator=av.instance_of(float))
    
@define(kw_only=True, eq=False)
class MosaicParams(TransformParams):
    mosaic_indices: List[int] = field(factory=list)
    
class Mosaic(BaseTransform):
    def __init__(self, mosaic_config : MosaicConfig, 
                 inout_map : MosaicInputOutputMap):
        self.m_mosaic_config = mosaic_config
        self.m_inout_map = inout_map
        assert len(self.m_inout_map.image_mosaic_settings) > 0, "image_mosaic_settings must be provided"
        
    def do_transform(self, input_data : TransformInput, 
                     context : TransformContext = None) -> TransformOutput:
        '''
        perform random crop on the images and boxes
        '''
        
        output = self._do_transform(input_data.data_blocks)
            
        mosaic_params = MosaicParams(
            mosaic_indices=[0,0,0,0]
        )
        final_output = TransformOutput(data_blocks=output, params=mosaic_params)
        return final_output
        
        
    def _do_transform(self, input_data : Dict[str, DataBlock], *args, **kwargs) -> Dict[str, DataBlock]:
        '''
        perform random crop on the images and boxes
        '''
        for img_setting in self.m_inout_map.image_mosaic_settings:
            assert img_setting.image_key.main_key in input_data, f"image key {img_setting.image_key} not found in data"
        for box_setting in self.m_inout_map.box_mosaic_settings:
            assert box_setting.box_key.main_key in input_data, f"box key {box_setting.box_key} not found in data"
        for label_setting in self.m_inout_map.label_mosaic_settings:
            assert label_setting.label_key.main_key in input_data, f"label key {label_setting.label_key} not found in data"
        assert len(self.m_inout_map.image_mosaic_settings) == len(self.m_inout_map.box_mosaic_settings), "number of image and box settings must be the same"
        assert len(self.m_inout_map.image_mosaic_settings) == len(self.m_inout_map.label_mosaic_settings), "number of image and label settings must be the same"

        image_key = self.m_inout_map.image_mosaic_settings[0].image_key
        image_datablock : ImagesBlock = input_data[image_key.main_key]
        
        config = self.m_mosaic_config
        output_data = copy.copy(input_data)

        for img_setting, boxes_setting, labels_setting in zip(self.m_inout_map.image_mosaic_settings, 
                                                     self.m_inout_map.box_mosaic_settings,
                                                     self.m_inout_map.label_mosaic_settings):
            image_key = img_setting.image_key
            output_image_key = img_setting.output_key
            boxes_key = boxes_setting.box_key
            output_boxes_key = boxes_setting.output_key
            labels_key = labels_setting.label_key
            output_labels_key = labels_setting.output_key
            image_datablock = input_data[image_key.main_key]
            boxes_datablock = input_data[boxes_key.main_key]
            labels_datablock = input_data[labels_key.main_key]
            out_image_blk = ImagesBlock()
            out_boxes_blk = BoxesBlock()
            out_labels_blk = VectorsBlock()
            for sub_key in image_datablock.get_keys():
                image_data = image_datablock.get_data(sub_key)
                image_data_spec = image_datablock.get_spec(sub_key)
                boxes_data = boxes_datablock.get_data(sub_key)
                boxes_data_spec = boxes_datablock.get_spec(sub_key)
                labels_data = labels_datablock.get_data(sub_key)
                labels_data_spec = labels_datablock.get_spec(sub_key)
                if fn.random.coin_flip(probability=config.probability):
                    mosaic_image, mosaic_boxes, mosaic_labels = dali_mosaic_images(image_data.gpu(), boxes_data.gpu(), labels_data.gpu())
                    mosaic_image = fn.resize(mosaic_image, resize_x=config.mosaic_width, resize_y=config.mosaic_height)
                    image_data_spec.height = config.mosaic_height
                    image_data_spec.width = config.mosaic_width
                else:
                    mosaic_image = image_data.gpu()
                    mosaic_boxes = boxes_data.gpu()
                    mosaic_labels = labels_data.gpu()
                out_image_blk.add_data(sub_key, mosaic_image, image_data_spec)
                out_boxes_blk.add_data(sub_key, mosaic_boxes, boxes_data_spec)
                out_labels_blk.add_data(sub_key, mosaic_labels, labels_data_spec)
            output_data[output_image_key.main_key] = out_image_blk
            output_data[output_boxes_key.main_key] = out_boxes_blk
            output_data[output_labels_key.main_key] = out_labels_blk
        return output_data