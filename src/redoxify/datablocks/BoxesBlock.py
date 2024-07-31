from typing import Dict, Tuple, Union, List
from attrs import define, field
import attrs.validators as av

import nvidia.dali.fn as fn
from nvidia.dali.pipeline import DataNode as DALINode
from nvidia.dali.types import DALIImageType

from redoxify.datablocks.DataBlock import DataBlock, DataSpec
import redoxify.functionals.cpu_ops as cpu_ops

@define(kw_only=True, eq=False)
class BoxSpec(DataSpec):
    # xywh or xyxy, where xyxy means xmin, ymin, xmax, ymax
    format: str = field(default='xyxy', validator=av.in_(['xyxy', 'xywh']))

    # whether the box coordinates are normalized (divided by image size) or not
    is_normalized : bool = field(default=True)
    
    # xywh or xyxy box in global coordinate, the xyxy boxes are relative to this box
    reference_box : Union[List[float], DALINode] = field(default=None)
    
    def get_DALI_box_layout(self) -> str:
        # 'xyxy' -> 'xyXY'
        # 'xywh' -> 'xyWH'
        if self.format is None:
            return None
        
        box_layout = 'xyXY' if self.format == 'xyxy' else 'xyWH'
        return box_layout
    

class BoxesBlock(DataBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass
    
    def subset_by_index(self, indices : Union[List[int], DALINode]) -> 'BoxesBlock':
        '''
        subset the boxes block by indices
        
        parameters
        ------------
        indices: List[int] or DALINode
            the indices to subset the boxes block
        
        return
        ---------
        BoxesBlock
            the subsetted boxes block
        '''
        #FIXME: indices not used, implement gather function by python
        out = BoxesBlock()
        for key, data in self.m_content.items():
            data_Nx4 = fn.reinterpret(data, shape=[-1, 4])            
            subset_Nx4 = cpu_ops.dali_gather_along_axis(data_Nx4, indices, axis=0)
            out.add_data(key, subset_Nx4, self.m_specs[key])
        return out
        
    def add_data(self, key: str, data: DALINode, spec: BoxSpec):
        '''
        add data to data block, and decode them as needed
        '''
        super().add_data(key, data, spec)
    
    def get_boxes(self, /, *, as_Nx4:bool, key: str = None) -> DALINode:
        '''
        Get the boxes
        '''
        assert isinstance(as_Nx4, bool), f"as_Nx4 should be bool, but got {as_Nx4}"
        
        if not self.m_content:
            return None
        
        if key is None:
            key = next(iter(self.m_content.keys()))
        raw_box = self.m_content.get(key)
        if raw_box is None:
            return raw_box
        
        if as_Nx4:
            return fn.reinterpret(raw_box, shape=[-1, 4])
        else:
            return raw_box
    
    def get_spec(self, key: str = None) -> BoxSpec:
        return super().get_spec(key)
    
    def get_boxes_normalized(self, key: Union[str, None],
                             image_width : Union[int, float, DALINode],
                             image_height : Union[int, float, DALINode]
                             ) -> DALINode:
        '''
        Get the boxes in normalized coordinate
        
        parameters
        ------------
        key: str
            the key of the boxes
        image_width: int, float, DALINode
            the width of the image
        image_height: int, float, DALINode
            the height of the image
        
        return
        ---------
        DALINode : (N,4)
            Nx4 boxes in normalized coordinate (normalized by width and height)
        '''
        raw_box = self.get_boxes(as_Nx4=True, key=key)
        if raw_box is None:
            return None
        
        spec = self.get_spec(key)
        if spec.is_normalized:  # already normalized, just return
            out = fn.reinterpret(raw_box, shape=[-1, 4])
            return out
        
        # normalize it
        box_rows = fn.reinterpret(raw_box, shape=[-1, 4])   # view as [N, 4]
        if spec.format == 'xyxy':   # input is xyxy
            xmin, xmax = box_rows[:,0], box_rows[:,2]
            ymin, ymax = box_rows[:,1], box_rows[:,3]
            new_xyxy = [
                xmin / image_width,
                ymin / image_height,
                xmax / image_width,
                ymax / image_height
            ]
            out = fn.stack(*new_xyxy, axis=-1)
        elif spec.format == 'xywh':
            x, y = box_rows[:,0], box_rows[:,1]
            w, h = box_rows[:,2], box_rows[:,3]
            new_xywh = [
                x / image_width,
                y / image_height,
                w / image_width,
                h / image_height
            ]
            out = fn.stack(*new_xywh, axis=-1)
        else:
            raise ValueError(f"Unknown box format {spec.format}")
            
        return out
            