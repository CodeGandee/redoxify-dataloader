from typing import Union, Dict, List
from attrs import define, field
import attrs.validators as av
import copy

import nvidia.dali.fn as fn
import nvidia.dali.math as dmath
import nvidia.dali.types as DT
from nvidia.dali.pipeline import Pipeline

import redoxify.GlobalConst as C
from redoxify.RedoxTypes import DataKey
from redoxify.datablocks.DataBlock import DataBlock, DALINode

from redoxify.datablocks.ImagesBlock import (
    ImagesBlock, ImageSpec
)
from redoxify.datablocks.BoxesBlock import (
    BoxesBlock, BoxSpec
)
from redoxify.datablocks.VectorsBlock import (
    VectorsBlock, VectorSpec
)

from redoxify.transforms.BaseTransform import (
    BaseTransform, 
    TransformParams,
    TransformInput,
    TransformOutput,
    TransformContext
)

@define(kw_only=True, eq=False)
class LabelCropSetting:
    ''' A group of labels of same length to be cropped by a reference boxes
    
    Protocol:
    Given a dict of datablocks db, we assume the followings:
    - let ref_box_data = db[ref_box_key.main_key].get_data(ref_box_key.sub_key), it contains N boxes
    - reference boxes themselves are NOT cropped, they are used to determine what to crop for box_key and label_key
    - crop_box_key.sub_key must be None, because all sub_keys are cropped in the same way
    - box_data = db[crop_box_key.main_key], it should also contains N boxes, same as primary boxes
    - if ref_box_data[t] is cropped, box_data[t] (including boxes in all sub keys) must be cropped in the same way
    - crop_label_key.sub_key must be None, because all sub_keys are cropped in the same way
    - label_data = db[crop_label_key.main_key], label_data contains N labels, one for each box    
    - if ref_box_data[t] is cropped, label_data[t] (including boxes in all sub keys) must be cropped in the same way
    - after cropping:
        - output_data[output_box_key.main_key] contains cropped version of db[crop_box_key.main_key]
        - output_data[output_label_key.main_key] contains cropped version of db[crop_label_key.main_key]
    '''
    ref_box_key : DataKey = field()
    crop_box_key : DataKey = field()
    crop_label_key : DataKey = field()
    output_box_key : DataKey = field()
    output_label_key : DataKey = field()
    
    @crop_box_key.validator
    def _validate_boxes(self, attribute, value):
        if value is None:
            return
        assert value.sub_key is None, "box sub_key must be None"
            
    @crop_label_key.validator
    def _validate_labels(self, attribute, value):
        if value is None:
            return
        assert value.sub_key is None, "label sub_key must be None"
        
@define(kw_only=True, eq=False)
class ImageCropSetting:
    ''' An image to be cropped and output by the transform.
    
    Given a dict of datablocks db, we assume the followings:
    - db[image_key] is cropped, and output to db[output_key]
    - image_key.sub_key and output_key.sub_key must be None, because all sub_keys are cropped in the same way
    - the output datablock has the same sub keys as input
    '''
    image_key : DataKey = field()
    output_key : DataKey = field()
    
@define(kw_only=True, eq=False)
class CropInputOutputMap:
    image_crop_settings : List[ImageCropSetting] = field(factory=list)
    label_crop_settings : List[LabelCropSetting] = field(factory=list)
    
    def empty(self) -> bool:
        return len(self.image_crop_settings) == 0 and len(self.label_crop_settings) == 0
    
    @label_crop_settings.validator
    def _validate_label_crop_settings(self, attribute, value):
        # no repeated output box keys
        output_box_keys : List[str] = [x.output_box_key.main_key for x in value]
        assert len(output_box_keys) == len(set(output_box_keys)), "output box keys should be unique"
        output_label_keys : List[str] = [x.output_label_key.main_key for x in value]
        assert len(output_label_keys) == len(set(output_label_keys)), "output label keys should be unique"
        
    @image_crop_settings.validator
    def _validate_image_crop_settings(self, attribute, value):
        # input and output sub_key must be None
        for x in value:
            assert x.image_key.sub_key is None, "image_key.sub_key must be None"
            assert x.output_key.sub_key is None, "output_key.sub_key must be None"
        
        # no repeated output key
        output_keys : List[str] = [x.output_key.main_key for x in value]
        assert len(output_keys) == len(set(output_keys)), "output keys should be unique"
    

@define(kw_only=True, eq=False)
class CropRuntimeParams(TransformParams):
    crop_anchor : DALINode = field(default=None)    # see fn.slice() for detail
    crop_shape : DALINode = field(default=None)
    
    # whether the crop_anchor and crop_shape are normalized or not
    is_normalized : bool = field(default=True) 
    
    # selected_box_indices[main_key][sub_key] = box indices selected by crop
    # selected_box_indices : Dict[str, DALINode] = field(factory=dict)

@define(kw_only=True, eq=False)
class CropConfig:
    # randomize aspect ratio width/height
    aspect_ratio_wh_min : float = field()
    aspect_ratio_wh_max : float = field()
    
    # size of the box relative to the image size
    # given as normalized box length, in range (0,1]
    box_length_min : float = field(validator=[av.gt(0), av.le(1.0)])
    box_length_max : float = field(validator=[av.gt(0), av.le(1.0)])
    
    # crop metric, if the metric for the target box is smaller than this, the box will be cropped
    # if multiple crop_metric_min is given, it will be applied by random choice
    # overlap = area(box & crop_window) / area(box)
    # iou = area(box & crop_window) / area(box | crop_window)
    crop_metric_type : str = field(default='overlap', validator=av.in_(['overlap', 'iou']))
    crop_metric_min : Union[float, List[float]] = field(default=0.0)
    
    # internal parameters of nvidia DALI, do not touch if you don't know what you are doing
    _num_attempts : int = field(default=C.Defaults.RandomCropNumAttempts)
    _all_boxes_above_threshold : bool = field(default=C.Defaults.RandomCropAllRemainBoxesSatisfyThreshold)

    def __attrs_post_init__(self):
        # check for min<=max
        assert self.aspect_ratio_wh_min <= self.aspect_ratio_wh_max, "aspect_ratio_wh_min should be less than aspect_ratio_wh_max"
        assert self.box_length_min <= self.box_length_max, "box_length_min should be less than box_length_max"
        
    @crop_metric_min.validator
    def _validate_crop_metric_min(self, attribute, value):
        if isinstance(value, float):
            assert value >= 0.0 and value <= 1.0, "crop_metric_min should be within [0, 1]"
        elif isinstance(value, list):
            for x in value:
                assert x >= 0.0 and x <= 1.0, "crop_metric_min should be within [0, 1]"
        else:
            raise ValueError("crop_metric_min should be float or list of float")

class RandomCropWithBoxes(BaseTransform):
    ''' Do the image cropping along with bboxes and labels.
    Protocol:
        - Images should be of the same size, and they are cropped in the same way
        - Boxes and labels are processed by groups, for each group of (boxes, labels), they satisify:
            - boxes = Nx4, labels = NxM, labels[i] is the label for boxes[i]
    '''
    def __init__(self, crop_config : CropConfig, 
                 inout_map : CropInputOutputMap):
        '''
        all images and boxes are considered as a single sample, and will be cropped in the same way
        '''
        self.m_crop_config = crop_config
        self.m_inout_map = inout_map
        
        # inout map must have image
        assert len(self.m_inout_map.image_crop_settings) > 0, "must have image to crop"
        
    def do_transform(self, input_data : TransformInput, 
                     context : TransformContext = None) -> TransformOutput:
        '''
        perform random crop on the images and boxes
        '''
        
        # all keys must exist
        for img_setting in self.m_inout_map.image_crop_settings:
            assert img_setting.image_key.main_key in input_data.data_blocks, f"image key {img_setting.image_key.main_key} not found"
        for lb_setting in self.m_inout_map.label_crop_settings:
            assert lb_setting.ref_box_key.main_key in input_data.data_blocks, f"ref box key {lb_setting.ref_box_key.main_key} not found"
            assert (lb_setting.crop_box_key is None or lb_setting.crop_box_key.main_key in input_data.data_blocks), f"crop box key {lb_setting.crop_box_key.main_key} not found"
            assert (lb_setting.crop_label_key is None or lb_setting.crop_label_key.main_key in input_data.data_blocks), f"crop label key {lb_setting.crop_label_key.main_key} not found"
        
        # determine we are in what case
        config = self.m_crop_config
        pipe = Pipeline.current()
        runtime_params = CropRuntimeParams()
        output = TransformOutput()
        
        # do not modify input dict
        output.data_blocks = copy.copy(input_data.data_blocks)
        output.params = runtime_params
        
        # get image and size
        image_key = self.m_inout_map.image_crop_settings[0].image_key
        image_datablock : ImagesBlock = input_data.data_blocks[image_key.main_key]
        image_height, image_width, _ = image_datablock.get_size_hwc(size_source='spec')
        
        # do you have box? if yes, crop the box and then images
        # otherwise, generate dummy crop and just crop the images
        if len(self.m_inout_map.label_crop_settings) > 0:
            for lb_setting in self.m_inout_map.label_crop_settings:
                # crop the reference boxes
                ref_box_key = lb_setting.ref_box_key
                ref_box_data : BoxesBlock = input_data.data_blocks[ref_box_key.main_key]
                ref_box_spec = ref_box_data.get_spec(ref_box_key.sub_key)
                ref_box_Nx4 = ref_box_data.get_boxes_normalized(
                    key = ref_box_key.sub_key,
                    image_width = image_width,
                    image_height = image_height
                )
                # HACK: clamp the boxes to [0,1] to avoid out of bounds error, for example, when xmax or ymax is 1.0+1e-6
                ref_box_Nx4 = dmath.clamp(ref_box_Nx4, lo=0.0, hi=1.0)
                # FIXME: optimize for single (box, label) input case where ref box is target box
                # avoid passing data to custom CPU function
                _anchor, _shape, _bboxes, _boxes_indices = \
                    fn.random_bbox_crop(
                        ref_box_Nx4,
                        aspect_ratio=[config.aspect_ratio_wh_min, config.aspect_ratio_wh_max],
                        bbox_layout=ref_box_spec.get_DALI_box_layout(),
                        scaling=[config.box_length_min, config.box_length_max],
                        shape_layout='WH',
                        output_bbox_indices=True,
                        seed = pipe.seed,
                        all_boxes_above_threshold=config._all_boxes_above_threshold,
                        num_attempts=config._num_attempts,
                        threshold_type=config.crop_metric_type,
                        thresholds=config.crop_metric_min,
                        bbox_prune_threshold=config.crop_metric_min,
                )
                    
                # store the crop window
                runtime_params.crop_anchor = _anchor
                runtime_params.crop_shape = _shape
                runtime_params.is_normalized = True
                
                # subset other boxes and labels
                if lb_setting.crop_box_key is not None:
                    crop_box_key = lb_setting.crop_box_key
                    
                    # special case: crop_box_key == ref_box_key, and it only has one sub_key
                    # the result is just _bboxes
                    crop_box_subset = BoxesBlock()
                    if crop_box_key.main_key == ref_box_key.main_key and len(ref_box_data.get_keys())==1:
                        # crop_box_subset = BoxesBlock()
                        crop_box_spec = ref_box_data.get_spec().clone()
                        crop_box_spec.is_normalized = True
                        crop_box_subset.add_data(
                            key=ref_box_data.get_keys()[0],
                            data=_bboxes,
                            spec=crop_box_spec,
                        )
                    else:
                        crop_box_data : BoxesBlock = input_data.data_blocks[crop_box_key.main_key]
                        _crop_box_subset : BoxesBlock = crop_box_data.subset_by_index(_boxes_indices)
                        
                        # HACK: you have to create crop_box_subset outside in order for DALI to work
                        crop_box_subset = _crop_box_subset
                        
                    output_box_key = lb_setting.output_box_key
                    output.data_blocks[output_box_key.main_key] = crop_box_subset
                
                if lb_setting.crop_box_key is not None:
                    crop_label_key = lb_setting.crop_label_key
                    crop_label_data : VectorsBlock = input_data.data_blocks[crop_label_key.main_key]
                    crop_label_subset : VectorsBlock = crop_label_data.subset_by_index(_boxes_indices)
                    output_label_key = lb_setting.output_label_key
                    output.data_blocks[output_label_key.main_key] = crop_label_subset
        else:
            # no boxes, generate random crop so that we can crop the images later
            image_shape_wh = [
                fn.cast(image_width, dtype=DT.DALIDataType.INT32),
                fn.cast(image_height, dtype=DT.DALIDataType.INT32)
            ]
            image_shape_wh = fn.cat(*image_shape_wh)
            _anchor, _shape = fn.random_crop_generator(
                image_shape_wh,
                random_aspect_ratio=[config.aspect_ratio_wh_min, config.aspect_ratio_wh_max],
                random_area=[config.box_length_min ** 2, config.box_length_max ** 2],
                seed = pipe.seed,
                num_attempts=config._num_attempts,
            )
            
            runtime_params.crop_anchor = _anchor
            runtime_params.crop_shape = _shape
            runtime_params.is_normalized = False
            
        # crop the images based on crop
        for img_inout_setting in self.m_inout_map.image_crop_settings:
            img_key = img_inout_setting.image_key
            img_data : ImagesBlock = input_data.data_blocks[img_key.main_key]
            img_output = img_data._get_cropped_images_by_anchor(
                anchor = runtime_params.crop_anchor,
                shape = runtime_params.crop_shape,
                is_normalized = runtime_params.is_normalized,
                out_of_bounds_policy='error',
            )
            output_key = img_inout_setting.output_key
            output.data_blocks[output_key.main_key] = img_output
            
        return output
    