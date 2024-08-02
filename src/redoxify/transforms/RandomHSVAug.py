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
class ImageRandomHSVSetting:
    image_key : DataKey = field()
    output_key : DataKey = field()

@define(kw_only=True, eq=False)
class RandomHSVInputOutputMap:
    image_hsv_settings : List[ImageRandomHSVSetting] = field(factory=list)
    
    
@define(kw_only=True, eq=False)
class RandomHSVConfig:
    hue_delta : Union[float, DALINode] = field(default=30.0)
    saturation_delta : Union[float, DALINode] = field(default=0.5)
    value_delta : Union[float, DALINode] = field(default=0.5)
    probability : Union[float, DALINode] = field(default=0.5)
    
@define(kw_only=True, eq=False)
class RandomHSVParams(TransformParams):
    aug : Union[bool, DALINode] = field(default=True)
    
class RandomHSVAug(BaseTransform):
    def __init__(self, hsv_config : RandomHSVConfig, 
                 inout_map : RandomHSVInputOutputMap):
        self.m_hsv_config = hsv_config
        self.m_inout_map = inout_map
        assert len(self.m_inout_map.image_hsv_settings) > 0, "image_hsv_settings must be provided"
        
    def do_transform(self, input_data : TransformInput, 
                     context : TransformContext = None) -> TransformOutput:
        '''
        perform random crop on the images and boxes
        '''
        
        output, aug = self._do_transform(input_data.data_blocks)
        hsv_params = RandomHSVParams(aug=aug)
        final_output = TransformOutput(data_blocks=output, params=hsv_params)
        return final_output
        
        
    def _do_transform(self, input_data : Dict[str, DataBlock], *args, **kwargs) -> Dict[str, DataBlock]:
        '''
        perform random crop on the images and boxes
        '''
        for img_setting in self.m_inout_map.image_hsv_settings:
            assert img_setting.image_key.main_key in input_data, f"image key {img_setting.image_key} not found in data"
            
        output_data = copy.copy(input_data)
        if not fn.random.coin_flip(probability=self.m_hsv_config.probability):
            output_data[output_key.main_key] = input_data[image_key.main_key]
            return output_data, False
        hue = fn.random.uniform(range=[-self.m_hsv_config.hue_delta, self.m_hsv_config.hue_delta])
        saturation = fn.random.uniform(range=[1-self.m_hsv_config.saturation_delta, 1+self.m_hsv_config.saturation_delta])
        value = fn.random.uniform(range=[1-self.m_hsv_config.value_delta, 1+self.m_hsv_config.value_delta])
        for img_setting in self.m_inout_map.image_hsv_settings:
            image_key = img_setting.image_key
            output_key = img_setting.output_key
            image_datablock = input_data[image_key.main_key]
            out_img_blk = ImagesBlock()
            for sub_key in image_datablock.get_keys():
                out_image_spec = image_datablock.get_spec(sub_key).clone()
                _image = image_datablock.get_decoded_tensor(sub_key)
                _image = fn.hsv(_image, hue=hue, saturation=saturation, value=value)
                out_img_blk.add_data(sub_key, _image, out_image_spec)
            output_data[output_key.main_key] = out_img_blk
        return output_data, True