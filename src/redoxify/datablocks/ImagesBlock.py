from typing import Dict, Tuple, Union, List
from attrs import define, field
import attrs.validators as av
import copy

import nvidia.dali.fn as fn
from nvidia.dali.types import DALIDataType
from nvidia.dali.pipeline import DataNode as DALINode
from nvidia.dali.types import DALIImageType

from redoxify.datablocks.DataBlock import DataBlock, DataSpec
import redoxify.GlobalConst as C

@define(kw_only=True, eq=False)
class ImageSpec(DataSpec):
    encoding: str = field(default=C.ImageEncoding.raw, 
                          validator=av.in_(C.ImageEncoding.get_all_encodings())) # 'raw' means no encoding
    data_layout : str = field(default='hwc',
                              validator=av.in_(['hwc', 'chw', 'hw']))    # can be 'hwc' or 'chw' or 'hw'
    decode_as : DALIImageType = field(default=DALIImageType.RGB)   # by default, decoded as RGB
    decode_to_device : str = field(default='auto', 
                                   validator=av.in_(['auto', 'cpu', 'gpu']))
    
    # size information in cpu, in pixels (even if the value is float)
    channel : Union[int, DALINode, None] = field(default=None)
    width : Union[int, DALINode, None] = field(default=None)
    height : Union[int, DALINode, None] = field(default=None)
    dtype : DALIDataType = field(default=DALIDataType.UINT8)

    
    def set_size_by_image_data(self, image_data : DALINode, skip_if_set : bool = False):
        ''' read size info from image_data, and set them to spec.
        
        parameters
        -------------
        image_data : DALINode
            the image data to read size from
        skip_if_set : bool
            if the size is already set, skip setting it
        '''
        
        # for image, we get its size before decoding
        if self.encoding == C.ImageEncoding.raw:
            # no encoding, get size directly
            _h, _w, _c = get_hwc_from_raw_image(image_data, self)
        else:
            # the image is encoded, peek its size
            _shape = fn.peek_image_shape(image_data)
            _h, _w = _shape[0], _shape[1]
            
            if self.decode_as == DALIImageType.RGB:
                _c = 3
            elif self.decode_as == DALIImageType.GRAY:
                _c = 1
            elif self.decode_as == DALIImageType.YCbCr:
                _c = 3
            elif self.decode_as == DALIImageType.BGR:
                _c = 3
            elif self.decode_as == DALIImageType.ANY_DATA:
                _c = None
            else:
                raise ValueError(f"unknown decode_as: {self.decode_as}")
        
        if skip_if_set:    # do not override user-provided data
            if self.height is None:
                self.height = _h
            if self.width is None:
                self.width = _w
            if self.channel is None:
                self.channel = _c
        else:   # override anyway
            self.height = _h
            self.width = _w
            self.channel = _c

class ImagesBlock(DataBlock):
    ''' A data block containing a set of images with the SAME width and height
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # decoded images
        self.m_decoded_images : Dict[str, DALINode] = {}
        
    def subset(self, keys: List[str]) -> 'ImagesBlock':
        out = ImagesBlock()
        for key in keys:
            assert key in self.m_content, f"key {key} not found in data block"
            out.m_specs[key] = self.m_specs[key]
            out.m_content[key] = self.m_content[key]
            out.m_decoded_images[key] = self.m_decoded_images[key]
        return out
    
    def gpu(self) -> 'ImagesBlock':
        # convert content to gpu
        out = super().gpu()
        
        # convert decoded images to gpu
        out.m_decoded_images = {}
        for key in self.m_decoded_images:
            out.m_decoded_images[key] = self.m_decoded_images[key].gpu()
            s : ImageSpec = self.m_specs[key].clone()
            s.decode_to_device = 'gpu'
            out.m_specs[key] = s
        return out
    
    def _get_resized_images(self,
                            target_width : Union[int, float, DALINode],
                            target_height : Union[int, float, DALINode],
                            keep_aspect_ratio : bool = True,
                            resize_mode : str = 'not_larger') -> 'ImagesBlock':
        if keep_aspect_ratio:
            assert resize_mode in ['not_larger', 'not_smaller'], "resize_mode must be 'not_larger' or 'not_smaller'"
        else:
            resize_mode = 'default'
        
        out = ImagesBlock()
        for sub_key in self.m_content:
            decoded_image = self.m_decoded_images[sub_key]
            new_spec : ImageSpec = self.m_specs[sub_key].clone()
            new_spec.encoding = C.ImageEncoding.raw
            new_spec.width = target_width
            new_spec.height = target_height
            print(f"target_width: {target_width}, target_height: {target_height}")
            resized_image = fn.resize(decoded_image, 
                                       resize_x=target_width, 
                                       resize_y=target_height,
                                       mode=resize_mode)
            new_spec.height, new_spec.width = fn.shapes(resized_image)[0], fn.shapes(resized_image)[1]
            out.add_data(sub_key, resized_image, new_spec)
        return out
    
    def _get_pad_images(self,
                        target_width : Union[int, float, DALINode],
                        target_height : Union[int, float, DALINode],
                        fill_values : Union[int, float, DALINode, List[int], List[float]] = 0.0,
                        align_center: bool = False) -> 'ImagesBlock':
        out = ImagesBlock()
        for sub_key in self.m_content:
            decoded_image = self.m_decoded_images[sub_key]
            new_spec : ImageSpec = self.m_specs[sub_key].clone()
            new_spec.encoding = C.ImageEncoding.raw
            source_width = self.m_specs[sub_key].width
            source_height = self.m_specs[sub_key].height
            new_spec.width = target_width
            new_spec.height = target_height
            if align_center:
                anch = fn.stack((target_width - source_width)/2, (target_height - source_height)/2)
                anch = fn.cast(anch, dtype=DALIDataType.INT32)
            else:
                anch = fn.zeros(shape=(2,), dtype=DALIDataType.INT32)
            images_pad = fn.slice(decoded_image, -anch, (target_width, target_height), out_of_bounds_policy="pad", fill_values=fill_values,
                                normalized_shape=False, normalized_anchor=False)
            # images_pad = fn.crop(decoded_image, crop=[target_height, target_width], 
            #                        crop_pos_x=0.0, crop_pos_y=0.0, out_of_bounds_policy='pad', fill_values=fill_values)
            out.add_data(sub_key, images_pad, new_spec)
        return out, anch
        
    def _get_cropped_images_by_anchor(self, 
                                      anchor: Union[DALINode, Tuple[int,int], Tuple[float, float]], 
                                      shape: Union[DALINode, Tuple[int,int], Tuple[float, float]],
                                      is_normalized : bool,
                                      shape_layout : str = 'WH',
                                      out_of_bounds_policy : str = 'error',
                                      fill_value : float = 0.0) -> 'ImagesBlock':
        ''' crop the images by anchor and shape, following nvidia DALI convention
        see fn.slice() for details
        
        parameters
        -------------
        anchor : DALINode | Tuple[int,int] | Tuple[float, float]
            the anchor of the crop, if DALINode, it must be a 1d vector with 2 elements
            if Tuple[int,int], it must be (x,y) in pixel.
            if Tuple[float, float], it can be (x,y) in normalized coordinate or pixels, depending on is_normalized.
        shape : DALINode | Tuple[int,int] | Tuple[float, float]
            the shape of the crop, if DALINode, it must be a 1d vector with 2 elements
            if Tuple[int,int], it must be (width, height) in pixel.
            if Tuple[float, float], it can be (width, height) in normalized coordinate or pixels, depending on is_normalized.
        shape_layout : str
            the layout of the shape, 'WH' or 'HW', how to interpret the shape tuple
        is_normalized : bool
            if the anchor and shape are in normalized coordinate
        out_of_bounds_policy : str
            the policy to handle out of bounds, 'pad' or 'trim_to_shape' or 'error',
            see dali documentation for details. The default value follows dali default.
        fill_value : float
            the fill value for out of bounds policy
            
        return
        ----------
        ImagesBlock
            the cropped images
        '''
        assert shape_layout in ['WH', 'HW'], "shape_layout must be 'WH' or 'HW'"
        
        out = ImagesBlock()
        for sub_key in self.m_content:
            decoded_image = self.m_decoded_images[sub_key]
            old_spec : ImageSpec = self.m_specs[sub_key]
            new_spec : ImageSpec = self.m_specs[sub_key].clone()
            new_spec.encoding = C.ImageEncoding.raw
            if shape_layout == 'WH':
                new_width, new_height = shape[0], shape[1]
            else:
                new_height, new_width = shape[0], shape[1]
            if is_normalized:
                # width and height are in float
                # convert width and height to int pixel                
                new_width = new_width * old_spec.width
                new_height = new_height * old_spec.height
                
            # crop image
            cropped_image = fn.slice(decoded_image, 
                     anchor, shape, 
                     normalized_anchor=is_normalized, 
                     normalized_shape=is_normalized,
                     axis_names=shape_layout,
                     fill_values=fill_value,
                     out_of_bounds_policy=out_of_bounds_policy)
            
            # write to output
            new_spec.width = new_width
            new_spec.height = new_height
            out.add_data(sub_key, cropped_image, new_spec)
        return out
        
    def rename_key(self, key_before: str, key_after : str):
        '''
        Rename the key of the data
        
        parameters
        ------------
        key_before: str
            the key to rename, if None, rename the only key in the data block (it must have only one key)
        key_after: str
            the new key
        '''
        super().rename_key(key_before, key_after)
        self.m_decoded_images[key_after] = self.m_decoded_images[key_before]
        del self.m_decoded_images[key_before]
        
    def get_size_hwc(self, key : str = None, size_source : str = 'data') -> Tuple[DALINode, DALINode, DALINode]:
        '''
        Get the height, width, and channel of the image.
        
        parameters
        ------------
        key: str
            the key to get image size, if None, get size of any image, this is useful when you know
            there is only one key in the data block
        size_source : 'spec' | 'data'
            Get the size information from data or spec.
            
        return
        ----------
        height : DALINode
            the height of the image
        width : DALINode
            the width of the image
        channel : DALINode
            the channel of the image
        '''
        assert len(self.m_decoded_images) > 0, "no image in the data block"
        assert size_source in ['spec', 'data', None], "size_source must be 'spec' or 'data' or None"
        
        # if key is not given, get the first key
        if key is None:
            key = next(iter(self.m_decoded_images.keys()))
        
        spec = self.m_specs[key]
        if size_source == 'spec':
            if spec.height is not None and spec.width is not None and spec.channel is not None:
                return spec.height, spec.width, spec.channel
            else:
                raise ValueError("size_source is 'spec', but size is not in spec")
        else:
            raw_image = self.get_decoded_tensor(key)
            return self._get_size_hwc_from_raw_image(raw_image, spec)
                    
    def add_data(self, key: str, data: DALINode, spec: ImageSpec):
        '''
        add data to data block, and decode them as needed
        '''
        # add to parent
        super().add_data(key, data, spec)
        
        # add decoded image
        if spec.encoding == C.ImageEncoding.raw:
            # BUG: raw_data might not be of the same format as spec.decode_as
            decoded_data = data
        else:
            device = spec.decode_to_device
            if device in ['gpu', 'auto']:
                device = 'mixed'
            decoded_data = fn.decoders.image(data, device=device, 
                                             output_type=spec.decode_as)
        self.m_decoded_images[key] = decoded_data
        
    def get_decoded_tensor(self, key: str = None) -> DALINode:
        '''
        get the decoded image.
        
        parameters
        ------------
        key: str
            the key to get the decoded image, if None, get any image, this is useful when you know
            there is only one key in the data block
            
        return
        ----------
        DALINode
            the decoded image
        '''
        if not self.m_decoded_images:
            return None
        
        if key is None:
            return next(iter(self.m_decoded_images.values()))
        else:
            return self.m_decoded_images.get(key)
    
    def get_spec(self, key: str = None) -> ImageSpec:
        return super().get_spec(key)
        
def get_hwc_from_raw_image(raw_image: DALINode, spec : ImageSpec) -> Tuple[DALINode, DALINode, DALINode]:
    ''' get height, width and channel from image without encoding
    
    parameters
    -------------
    raw_image : DALINode
        the raw image, no encoding
    spec : ImageSpec
        the specification of the image
        
    return
    --------
    height : DALINode
        the height of the image
    width : DALINode
        the width of the image
    channel : DALINode
        the channel of the image
    '''
    image_shape = fn.shapes(raw_image)
    
    output = []
    if spec.data_layout == "hwc":
        output = [image_shape[0], image_shape[1], image_shape[2]]
    elif spec.data_layout == "chw":
        output = [image_shape[1], image_shape[2], image_shape[0]]
    elif spec.data_layout == "hw":
        output = [image_shape[0], image_shape[1], 1]
    
    return tuple(output)