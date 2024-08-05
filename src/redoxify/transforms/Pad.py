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
class ImagePadSetting:
    image_key : DataKey = field()
    output_key : DataKey = field()

@define(kw_only=True, eq=False)
class BoxPadSetting:
    box_key : DataKey = field()
    output_key : DataKey = field()

@define(kw_only=True, eq=False)
class PadInputOutputMap:
    image_pad_settings : List[ImagePadSetting] = field(factory=list)
    box_pad_settings : List[BoxPadSetting] = field(factory=list)
    
    
@define(kw_only=True, eq=False)
class PadConfig:
    target_height : Union[int, float, DALINode] = field(validator=av.instance_of((int, float, DALINode)))
    target_width : Union[int, float, DALINode] = field(validator=av.instance_of((int, float, DALINode)))
    fill_values : Union[int, float, DALINode, List[int], List[float]] = field(validator=av.instance_of((int, float, DALINode, List)))
    aligh_center : bool = field(default=True)
    
@define(kw_only=True, eq=False)
class PadParams(TransformParams):
    pad_height: Union[int, float, DALINode] = field(validator=av.instance_of((int, float, DALINode)))
    pad_width: Union[int, float, DALINode] = field(validator=av.instance_of((int, float, DALINode)))
    
class Pad(BaseTransform):
    def __init__(self, pad_config : PadConfig, 
                 inout_map : PadInputOutputMap):
        self.m_pad_config = pad_config
        self.m_inout_map = inout_map
        assert len(self.m_inout_map.image_pad_settings) > 0, "image_pad_settings must be provided"
        
    def do_transform(self, input_data : TransformInput, 
                     context : TransformContext = None) -> TransformOutput:
        '''
        perform random crop on the images and boxes
        '''
        output, pad_width, pad_height = self._do_transform(input_data.data_blocks)
            
        pad_params = PadParams(
            pad_height=pad_height,
            pad_width=pad_width
        )
        final_output = TransformOutput(data_blocks=output, params=pad_params)
        return final_output
        
        
    def _do_transform(self, input_data : Dict[str, DataBlock], *args, **kwargs) -> Dict[str, DataBlock]:
        '''
        perform random crop on the images and boxes
        '''
        target_width = self.m_pad_config.target_width
        target_height = self.m_pad_config.target_height
        for img_setting in self.m_inout_map.image_pad_settings:
            assert img_setting.image_key.main_key in input_data, f"image key {img_setting.image_key} not found in data"
        for box_setting in self.m_inout_map.box_pad_settings:
            assert box_setting.box_key.main_key in input_data, f"box key {box_setting.box_key} not found in data"

        image_key = self.m_inout_map.image_pad_settings[0].image_key
        image_datablock : ImagesBlock = input_data[image_key.main_key]
        image_height, image_width, _ = image_datablock.get_size_hwc(size_source='spec')
        
        # assert target_height >= image_height, f"target height {target_height} must be greater than image height {image_height}"
        # assert target_width >= image_width, f"target width {target_width} must be greater than image width {image_width}"
        
        config = self.m_pad_config
        output_data = copy.copy(input_data)

        for img_setting in self.m_inout_map.image_pad_settings:
            image_key = img_setting.image_key
            output_key = img_setting.output_key
            image_datablock = input_data[image_key.main_key]
            output_data[output_key.main_key], anchor = image_datablock._get_pad_images(target_height=target_height, target_width=target_width, 
                                                                                fill_values=config.fill_values, align_center=config.aligh_center)
        
        for box_setting in self.m_inout_map.box_pad_settings:
            box_key = box_setting.box_key
            output_key = box_setting.output_key
            box_datablock : BoxesBlock = input_data[box_key.main_key]
            out_boxes_blk = BoxesBlock()
            for sub_key in box_datablock.get_keys():
                out_boxes_spec = box_datablock.get_spec(sub_key).clone()
                out_boxes_spec.is_normalized = True
                _boxes = box_datablock.get_boxes_normalized(sub_key, image_width=image_width, image_height=image_height)
                _col1 = (_boxes[:, 0] * image_width + anchor[0]) / target_width
                _col2 = (_boxes[:, 1] * image_height + anchor[1]) / target_height
                _col3 = (_boxes[:, 2] * image_width + anchor[0]) / target_width
                _col4 = (_boxes[:, 3] * image_height + anchor[1]) / target_height
                new_boxes = fn.stack(_col1, _col2, _col3, _col4, axis=-1)
                out_boxes_blk.add_data(sub_key, new_boxes, out_boxes_spec)
            output_data[output_key.main_key] = out_boxes_blk
        
        
        return output_data, target_width - image_width, target_height - image_height