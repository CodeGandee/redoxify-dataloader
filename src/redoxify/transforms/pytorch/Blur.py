from typing import Union, Dict, List
from attrs import define, field
import attrs.validators as av
import copy

import nvidia.dali.fn as fn
import nvidia.dali.math as dmath

from redoxify.datablocks.DataBlock import DataBlock, DALINode
from redoxify.functionals.cuda_blur import dali_blur_image

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
class ImageBlurSetting:
    image_key : DataKey = field()
    output_key : DataKey = field()

@define(kw_only=True, eq=False)
class BlurInputOutputMap:
    image_blur_settings : List[ImageBlurSetting] = field(factory=list)
    
    
@define(kw_only=True, eq=False)
class BlurConfig:
    blur_limit : Union[int, DALINode] = field(default=7)
    probability : Union[float, DALINode] = field(default=0.5)
    
@define(kw_only=True, eq=False)
class BlurParams(TransformParams):
    aug : Union[bool, DALINode] = field(default=True)
    
class Blur(BaseTransform):
    def __init__(self, blur_config : BlurConfig, 
                 inout_map : BlurInputOutputMap):
        self.m_blur_config = blur_config
        self.m_inout_map = inout_map
        assert len(self.m_inout_map.image_blur_settings) > 0, "image_blur_settings must be provided"
        
    def do_transform(self, input_data : TransformInput, 
                     context : TransformContext = None) -> TransformOutput:
        '''
        perform random crop on the images and boxes
        '''
        
        output, aug = self._do_transform(input_data.data_blocks)
        blur_params = BlurParams(aug=aug)
        final_output = TransformOutput(data_blocks=output, params=blur_params)
        return final_output
        
        
    def _do_transform(self, input_data : Dict[str, DataBlock], *args, **kwargs) -> Dict[str, DataBlock]:
        '''
        perform random crop on the images and boxes
        '''
        for img_setting in self.m_inout_map.image_blur_settings:
            assert img_setting.image_key.main_key in input_data, f"image key {img_setting.image_key} not found in data"
            
        output_data = copy.copy(input_data)
        for img_setting in self.m_inout_map.image_blur_settings:
            image_key = img_setting.image_key
            output_key = img_setting.output_key
            image_datablock = input_data[image_key.main_key]
            out_img_blk = ImagesBlock()
            for sub_key in image_datablock.get_keys():
                out_image_spec = image_datablock.get_spec(sub_key).clone()
                #FIXME: solve the issue with the blur
                # if fn.random.coin_flip(probability=self.m_blur_config.probability):
                #     _image = image_datablock.get_decoded_tensor(sub_key)
                #     _image = dali_blur_image(_image, self.m_blur_config.blur_limit)
                # else:
                _image = image_datablock.get_decoded_tensor(sub_key)
                out_img_blk.add_data(sub_key, _image, out_image_spec)
            output_data[output_key.main_key] = out_img_blk
        return output_data, True