from typing import Union, Dict, List
from attrs import define, field
import attrs.validators as av
import copy

import nvidia.dali.fn as fn
import nvidia.dali.math as dmath
from redoxify.datablocks.DataBlock import DataBlock, DALINode

from redoxify.datablocks.ImagesBlock import (
    ImagesBlock as ImageDataBlock,
    ImageSpec
)
from redoxify.datablocks.BoxesBlock import (
    BoxesBlock as BoxesDataBlock,
    BoxSpec
)

from redoxify.transforms.BaseTransform import (
    BaseTransform, 
    TransformParams,
    TransformInput,
    TransformOutput,
    TransformContext
)

@define(kw_only=True, eq=False)
class CropConfig:
    # crop size, by default, it is normalized size
    crop_width_min : Union[int, float, DALINode] = field(validator=av.instance_of((int, float, DALINode)))
    crop_width_max : Union[int, float, DALINode] = field(validator=av.instance_of((int, float, DALINode)))
    crop_height_min : Union[int, float, DALINode] = field(validator=av.instance_of((int, float, DALINode)))
    crop_height_max : Union[int, float, DALINode] = field(validator=av.instance_of((int, float, DALINode)))
    is_crop_size_normalized : bool = field(default=True, 
                                           validator=av.instance_of(bool))
    
    # crop center, by default, it is normalized position
    crop_center_x_min : Union[int, float, DALINode, None] = field(validator=av.instance_of((int, float, DALINode, None)))
    crop_center_x_max : Union[int, float, DALINode, None] = field(validator=av.instance_of((int, float, DALINode, None)))
    crop_center_y_min : Union[int, float, DALINode, None] = field(validator=av.instance_of((int, float, DALINode, None)))
    crop_center_y_max : Union[int, float, DALINode, None] = field(validator=av.instance_of((int, float, DALINode, None)))
    is_crop_center_normalized : bool = field(default=True, 
                                             validator=av.instance_of(bool))
    def is_size_random_by_range(self) -> bool:
        width_random = self.crop_width_min != self.crop_width_max
        height_random = self.crop_height_min != self.crop_height_max
        return width_random and height_random
    
    def is_size_fixed(self) -> bool:
        width_fixed = self.crop_width_min == self.crop_width_max
        height_fixed = self.crop_height_min == self.crop_height_max
        return width_fixed and height_fixed
    
    def is_center_unspecified(self) -> bool:
        no_x = self.crop_center_x_min is None and self.crop_center_x_max is None
        no_y = self.crop_center_y_min is None and self.crop_center_y_max is None
        return no_x and no_y

    def __attrs_post_init__(self):
        # if normalized, the value should not be int
        if self.is_crop_size_normalized:
            assert not isinstance(self.crop_width_min, int), "crop_width_min should be float or DALINode"
            assert not isinstance(self.crop_width_max, int), "crop_width_max should be float or DALINode"
            assert not isinstance(self.crop_height_min, int), "crop_height_min should be float or DALINode"
            assert not isinstance(self.crop_height_max, int), "crop_height_max should be float or DALINode"
            
        if self.is_crop_center_normalized:
            assert not isinstance(self.crop_center_x_min, int), "crop_center_x_min should be float or DALINode"
            assert not isinstance(self.crop_center_x_max, int), "crop_center_x_max should be float or DALINode"
            assert not isinstance(self.crop_center_y_min, int), "crop_center_y_min should be float or DALINode"
            assert not isinstance(self.crop_center_y_max, int), "crop_center_y_max should be float or DALINode"
            
        # FIXME: check the range of the values, ensure min<=max


class RandomCropWithBoxes(BaseTransform):
    def __init__(self, crop_config : CropConfig, 
                 image_keys : List[str] = None, 
                 box_keys : List[str] = None):
        '''
        all images and boxes are considered as a single sample, and will be cropped in the same way
        
        parameters
        -------------
        crop_config : CropConfig
            the configuration of the crop
        image_keys : List[str]
            the keys of the images to be cropped, and they will be cropped in the same way
        box_keys : List[str]
            the keys of the boxes to be cropped, and they will be cropped in the same way as the image
        '''
        self.m_crop_config = crop_config
        self.m_image_keys : List[str] = image_keys
        self.m_box_keys : List[str] = box_keys
        
    def do_transform(self, input_data : TransformInput, 
                     context : TransformContext = None) -> TransformOutput:
        '''
        perform random crop on the images and boxes
        '''
        
        # determine we are in what case
        config = self.m_crop_config
        assert config.is_center_unspecified(), "only support fully random center now"
        assert self.m_image_keys is not None, "image_keys should be specified"
        
        db = input_data.data_blocks
        
        # get image size, we assume all images have the same size
        key = self.m_image_keys[0]
        image_height, image_width, _ = input_data.data_blocks[key].get_size_hwc(size_source='spec')
        
        # collect all boxes
        @define(kw_only=True, eq=False)
        class BoxKey:
            db_key : str = field()
            sub_key : str = field()
            count : DALINode = field()
        
        all_box_keys : List[BoxKey] = []
        box_Nx4 : List[DALINode] = []
        for main_key in self.m_box_keys:
            box_db : BoxesDataBlock = db[main_key]
            for sub_key in box_db.get_keys():
                #(N,4) boxes in normalized coordinate
                _boxes = box_db.get_boxes_normalized(sub_key, image_width, image_height)
                
                #record it
                box_Nx4.append(_boxes)
                all_box_keys.append(BoxKey(db_key=main_key, sub_key=sub_key, count = fn.shapes(_boxes)[0]))
                                    
        box_vstack_Nx4 = fn.cat(*box_Nx4, axis=0)
        boxes_cropped : List[DALINode] = []
        
        # random size
        if config.is_size_random_by_range():
            # compute crop shape
            crop_width = fn.random.uniform(range=[config.crop_width_min, config.crop_width_max])    
            crop_height = fn.random.uniform(range=[config.crop_height_min, config.crop_height_max])
            if config.is_crop_size_normalized:
                crop_width = crop_width * image_width
                crop_height = crop_height * image_height
                
            # crop all boxes, return everything, see DALI doc for detail
            _anchor, _shape, _bboxes, _boxes_indices = \
                fn.random_bbox_crop(
                    box_vstack_Nx4,
                    bbox_layout='xyXY', #box is given in (xmin, ymin, xmax, ymax)
                    shape_layout='WH', #shape is given in (width, height)
                    crop_shape=[crop_width, crop_height],
                    input_shape=[image_width, image_height],
                    output_bbox_indices=True,
                )
                
            # recover all boxes in original order
            
                
                
            
            
        
        output, crop_pos_x, crop_pos_y, crop_width, crop_height = self._do_transform(input_data.data_blocks)
        
        crop_params = RandomCenterCropParams(
            crop_width_pixel=crop_width,
            crop_height_pixel=crop_height,
            crop_pos_x=crop_pos_x,
            crop_pos_y=crop_pos_y
        )
        final_output = TransformOutput(data_blocks=output, params=crop_params)
        
        return final_output