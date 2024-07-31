from typing import Union, Dict, List
from attrs import define, field
import attrs.validators as av
import copy

import nvidia.dali.fn as fn
import nvidia.dali.math as dmath

from redoxify.datablocks.DataBlock import DataBlock, DALINode

from redoxify.RedoxTypes import DataKey

from redoxify.datablocks.ImagesBlock import (
    ImagesBlock, ImageSpec )

from redoxify.datablocks.BoxesBlock import (
    BoxesBlock, BoxSpec )

from redoxify.transforms.BaseTransform import (
    BaseTransform, 
    TransformParams,
    TransformInput,
    TransformOutput,
    TransformContext
)

@define(kw_only=True, eq=False)
class ImageResizeSetting:
    image_key : DataKey = field()
    output_key : DataKey = field()

@define(kw_only=True, eq=False)
class BoxResizeSetting:
    box_key : DataKey = field()
    output_key : DataKey = field()

@define(kw_only=True, eq=False)
class ResizeInputOutputMap:
    image_resize_settings : List[ImageResizeSetting] = field(factory=list)
    box_resize_settings : List[BoxResizeSetting] = field(factory=list)
    
    
@define(kw_only=True, eq=False)
class ResizeConfig:
    target_height : Union[int, float, DALINode] = field(validator=av.instance_of((int, float, DALINode)))
    target_width : Union[int, float, DALINode] = field(validator=av.instance_of((int, float, DALINode)))
    keep_aspect_ratio : bool = field(default=True)
    keep_aspect_ratio_mode : str = field(default='not_larger', validator=av.in_(['not_larger', 'not_smaller']))
    
@define(kw_only=True, eq=False)
class ResizeParams(TransformParams):
    resized_height: Union[int, float, DALINode] = field(validator=av.instance_of((int, float, DALINode)))
    resized_width: Union[int, float, DALINode] = field(validator=av.instance_of((int, float, DALINode)))
    scale_factor_x: Union[float, DALINode] = field(validator=av.instance_of((float, DALINode)))
    scale_factor_y: Union[float, DALINode] = field(validator=av.instance_of((float, DALINode)))

    
class Resize(BaseTransform):
    def __init__(self, resize_config : ResizeConfig, 
                 inout_map : ResizeInputOutputMap):
        self.m_resize_config = resize_config
        self.m_inout_map = inout_map
        assert len(self.m_inout_map.image_resize_settings) > 0, "image_resize_settings must be provided"
        
    def do_transform(self, input_data : TransformInput, 
                     context : TransformContext = None) -> TransformOutput:
        '''
        perform random crop on the images and boxes
        '''
        
        output, resize_width, resize_height, scale_factor_x, scale_factor_y = self._do_transform(input_data.data_blocks)
            
        resize_params = ResizeParams(
            resized_width=resize_width,
            resized_height=resize_height,
            scale_factor_x=scale_factor_x,
            scale_factor_y=scale_factor_y
        )
        final_output = TransformOutput(data_blocks=output, params=resize_params)
        
        return final_output
        
        
    def _do_transform(self, input_data : Dict[str, DataBlock], *args, **kwargs) -> Dict[str, DataBlock]:
        '''
        perform random crop on the images and boxes
        '''
        resized_width = self.m_resize_config.target_width
        resized_height = self.m_resize_config.target_height
        for img_setting in self.m_inout_map.image_resize_settings:
            assert img_setting.image_key.main_key in input_data, f"image key {img_setting.image_key} not found in data"
        for box_setting in self.m_inout_map.box_resize_settings:
            assert box_setting.box_key.main_key in input_data, f"box key {box_setting.box_key} not found in data"


        config = self.m_resize_config
        output_data = copy.copy(input_data)

        image_key = self.m_inout_map.image_resize_settings[0].image_key
        image_datablock : ImagesBlock = input_data[image_key.main_key]
        image_height, image_width, _ = image_datablock.get_size_hwc(size_source='spec')
        
        for img_setting in self.m_inout_map.image_resize_settings:
            image_key = img_setting.image_key
            output_key = img_setting.output_key
            image_datablock = input_data[image_key.main_key]
            output_data[output_key.main_key] = image_datablock._get_resized_images(target_height=resized_height, 
                                                                                   target_width=resized_width,
                                                                                   keep_aspect_ratio=config.keep_aspect_ratio,
                                                                                   resize_mode=config.keep_aspect_ratio_mode)
        
        for box_setting in self.m_inout_map.box_resize_settings:
            box_key = box_setting.box_key
            output_key = box_setting.output_key
            box_datablock : BoxesBlock = input_data[box_key.main_key]
            out_boxes_blk = BoxesBlock()
            for sub_key in box_datablock.get_keys():
                out_boxes_spec = box_datablock.get_spec(sub_key).clone()
                out_boxes_spec.is_normalized = True
                _boxes = box_datablock.get_boxes_normalized(sub_key, image_width=image_width, image_height=image_height)
                out_boxes_blk.add_data(sub_key, _boxes, out_boxes_spec)
            output_data[output_key.main_key] = out_boxes_blk
        
        scale_factor_x = resized_width / image_width
        scale_factor_y = resized_height / image_height
        
        return output_data, resized_width, resized_height, scale_factor_x, scale_factor_y