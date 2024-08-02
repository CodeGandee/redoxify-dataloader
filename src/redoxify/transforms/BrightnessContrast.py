from typing import Union, Dict, List, Tuple
from attrs import define, field
import attrs.validators as av
import copy

import nvidia.dali.fn as fn
import nvidia.dali.math as dmath

from redoxify.datablocks.DataBlock import DataBlock, DALINode

from redoxify.RedoxTypes import DataKey

from redoxify.datablocks.ImagesBlock import (
    ImagesBlock, ImageSpec )

from redoxify.transforms.BaseTransform import (
    BaseTransform, 
    TransformParams,
    TransformInput,
    TransformOutput,
    TransformContext
)

@define(kw_only=True, eq=False)
class ImageRandomBrightContrastSetting:
    image_key : DataKey = field()
    output_key : DataKey = field()

@define(kw_only=True, eq=False)
class RandomBrightContrastInputOutputMap:
    image_bright_contrast_settings : List[ImageRandomBrightContrastSetting] = field(factory=list)
    
    
@define(kw_only=True, eq=False)
class RandomBrightContrastConfig:
    brightness_limit : Union[float, DALINode, Tuple[float, float], Tuple[DALINode, DALINode]] = field(default=30.0)
    contrast_limit : Union[float, DALINode, Tuple[float, float], Tuple[DALINode, DALINode]] = field(default=0.5)
    probability : Union[float, DALINode] = field(default=0.5)
    
@define(kw_only=True, eq=False)
class RandomBrightContrastParams(TransformParams):
    aug : Union[bool, DALINode] = field(default=True)
    
class RandomBrightContrastAug(BaseTransform):
    def __init__(self, bright_contrast_config : RandomBrightContrastConfig, 
                 inout_map : RandomBrightContrastInputOutputMap):
        self.m_bright_contrast_config = bright_contrast_config
        self.m_inout_map = inout_map
        assert len(self.m_inout_map.image_bright_contrast_settings) > 0, "image_bright_contrast_settings must be provided"
        
    def do_transform(self, input_data : TransformInput, 
                     context : TransformContext = None) -> TransformOutput:
        '''
        perform random crop on the images and boxes
        '''
        
        output, aug = self._do_transform(input_data.data_blocks)
        bright_contrast_params = RandomBrightContrastParams(aug=aug)
        final_output = TransformOutput(data_blocks=output, params=bright_contrast_params)
        return final_output
        
        
    def _do_transform(self, input_data : Dict[str, DataBlock], *args, **kwargs) -> Dict[str, DataBlock]:
        '''
        perform random crop on the images and boxes
        '''
        for img_setting in self.m_inout_map.image_bright_contrast_settings:
            assert img_setting.image_key.main_key in input_data, f"image key {img_setting.image_key} not found in data"
            
        output_data = copy.copy(input_data)
        if not fn.random.coin_flip(probability=self.m_bright_contrast_config.probability):
            output_data[output_key.main_key] = input_data[image_key.main_key]
            return output_data, False
        if isinstance(self.m_bright_contrast_config.brightness_limit, float) or isinstance(self.m_bright_contrast_config.brightness_limit, DALINode):
            brightness_range = (self.m_bright_contrast_config.brightness_limit, self.m_bright_contrast_config.brightness_limit)
        elif isinstance(self.m_bright_contrast_config.brightness_limit, Tuple):
            brightness_range = self.m_bright_contrast_config.brightness_limit
        if isinstance(self.m_bright_contrast_config.contrast_limit, float) or isinstance(self.m_bright_contrast_config.contrast_limit, DALINode):
            contrast_range = (self.m_bright_contrast_config.contrast_limit, self.m_bright_contrast_config.contrast_limit)
        elif isinstance(self.m_bright_contrast_config.contrast_limit, Tuple):
            contrast_range = self.m_bright_contrast_config.contrast_limit
        random_brightness = 1.0 + fn.random.uniform(range=brightness_range)
        random_contrast = 1.0 + fn.random.uniform(range=contrast_range)
        for img_setting in self.m_inout_map.image_bright_contrast_settings:
            image_key = img_setting.image_key
            output_key = img_setting.output_key
            image_datablock = input_data[image_key.main_key]
            out_img_blk = ImagesBlock()
            for sub_key in image_datablock.get_keys():
                out_image_spec = image_datablock.get_spec(sub_key).clone()
                _image = image_datablock.get_decoded_tensor(sub_key)
                _image = fn.brightness_contrast(_image, brightness=random_brightness, contrast=random_contrast)
                out_img_blk.add_data(sub_key, _image, out_image_spec)
            output_data[output_key.main_key] = out_img_blk
        return output_data, True