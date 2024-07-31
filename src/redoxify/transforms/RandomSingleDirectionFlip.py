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
class ImageRandomSingleDirectionFlipSetting:
    image_key : DataKey = field()
    output_key : DataKey = field()

@define(kw_only=True, eq=False)
class BoxRandomSingleDirectionFlipSetting:
    box_key : DataKey = field()
    output_key : DataKey = field()

@define(kw_only=True, eq=False)
class RandomSingleDirectionFlipInputOutputMap:
    image_flip_settings : List[ImageRandomSingleDirectionFlipSetting] = field(factory=list)
    box_flip_settings : List[BoxRandomSingleDirectionFlipSetting] = field(factory=list)
    
    
@define(kw_only=True, eq=False)
class RandomSingleDirectionFlipConfig:
    probability : float = field(default=0.5)
    flip_direction : str = field(default='horizontal', validator=av.in_(['horizontal', 'vertical']))
    
@define(kw_only=True, eq=False)
class RandomSingleDirectionFlipParams(TransformParams):
    flipped : Union[bool, DALINode] = field()
    flip_direction : Union[str, None] = field(default=None)
    
class RandomSingleDirectionFlip(BaseTransform):
    def __init__(self, flip_config : RandomSingleDirectionFlipConfig, 
                 inout_map : RandomSingleDirectionFlipInputOutputMap):
        self.m_flip_config = flip_config
        self.m_inout_map = inout_map
        assert len(self.m_inout_map.image_flip_settings) > 0, "image_flip_settings must be provided"
        
    def do_transform(self, input_data : TransformInput, 
                     context : TransformContext = None) -> TransformOutput:
        '''
        perform random crop on the images and boxes
        '''
        
        output, if_flip, flip_direction = self._do_transform(input_data.data_blocks)
            
        flip_params = RandomSingleDirectionFlipParams(
            flipped = if_flip,
            flip_direction = flip_direction
        )
        final_output = TransformOutput(data_blocks=output, params=flip_params)
        return final_output
        
        
    def _do_transform(self, input_data : Dict[str, DataBlock], *args, **kwargs) -> Dict[str, DataBlock]:
        '''
        perform random crop on the images and boxes
        '''
        for img_setting in self.m_inout_map.image_flip_settings:
            assert img_setting.image_key.main_key in input_data, f"image key {img_setting.image_key} not found in data"
        for box_setting in self.m_inout_map.box_flip_settings:
            assert box_setting.box_key.main_key in input_data, f"box key {box_setting.box_key} not found in data"
        image_key = self.m_inout_map.image_flip_settings[0].image_key
        image_datablock : ImagesBlock = input_data[image_key.main_key]
        image_height, image_width, _ = image_datablock.get_size_hwc(size_source='spec')
        config = self.m_flip_config
        output_data = copy.copy(input_data)

        if_flip = fn.random.coin_flip(probability=config.probability)
        
        # HACK: for testing, use this to remove branch
        # if_flip = True
        
        flip_direction = config.flip_direction
        horizontal = 0
        vertical = 0
        if flip_direction == 'horizontal':
            horizontal = 1
        else:
            vertical = 1
        
        for img_setting in self.m_inout_map.image_flip_settings:
            image_key = img_setting.image_key
            output_key = img_setting.output_key
            image_datablock = input_data[image_key.main_key]
            out_img_blk = ImagesBlock()
            for sub_key in image_datablock.get_keys():
                out_image_spec = image_datablock.get_spec(sub_key).clone()
                _image = image_datablock.get_decoded_tensor(sub_key)
                if if_flip:
                    _image = fn.flip(_image, horizontal=horizontal, vertical=vertical)
                out_img_blk.add_data(sub_key, _image, out_image_spec)
            output_data[output_key.main_key] = out_img_blk
                
        
        for box_setting in self.m_inout_map.box_flip_settings:
            box_key = box_setting.box_key
            output_key = box_setting.output_key
            box_datablock : BoxesBlock = input_data[box_key.main_key]
            out_boxes_blk = BoxesBlock()
            for sub_key in box_datablock.get_keys():
                out_boxes_spec = box_datablock.get_spec(sub_key).clone()
                out_boxes_spec.is_normalized = True
                _boxes = box_datablock.get_boxes_normalized(sub_key, image_width=image_width, image_height=image_height)
                if if_flip:
                    ltrb = True if out_boxes_spec.format == 'xyxy' else False
                    _boxes = fn.bb_flip(_boxes, ltrb=ltrb, horizontal=horizontal, vertical=vertical)
                out_boxes_blk.add_data(sub_key, _boxes, out_boxes_spec)
            output_data[output_key.main_key] = out_boxes_blk
        
        
        return output_data, if_flip, flip_direction