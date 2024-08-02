from typing import Union
import nvidia.dali.fn as fn
import nvidia.dali.math as dmath
import copy
from typing import Dict, List
import attrs.validators as av
from attrs import define, field
from redoxify.RedoxTypes import DataKey

from redoxify.datablocks.DataBlock import DataBlock, DALINode

from redoxify.datablocks.ImagesBlock import (
    ImagesBlock, ImageSpec,
)
from redoxify.datablocks.BoxesBlock import (
    BoxesBlock, BoxSpec,
)
from redoxify.datablocks.VectorsBlock import (   
    VectorsBlock, VectorSpec,
)
from redoxify.transforms.BaseTransform import BaseTransform

from redoxify.transforms.BaseTransform import (
    BaseTransform, 
    TransformParams,
    TransformInput,
    TransformOutput,
    TransformContext
)

@define(kw_only=True, eq=False)
class GroupMemberSetting:
    member_key : DataKey = field()

@define(kw_only=True, eq=False)
class GroupingInputOutputMap:
    group_sub_keys : List[str] = field(factory=list)
    member_settings : List[GroupMemberSetting] = field(factory=list)

@define(kw_only=True, eq=False)
class GroupingConfig:
    group_name : str = field()
    data_type : str = field(default='image', validator=av.in_(['image', 'box', 'vector']))

@define(kw_only=True, eq=False) 
class GroupingParams:
    pass
    

class DataBlockGrouping(BaseTransform):
    def __init__(self, grouping_config : GroupingConfig,
                 inout_map : GroupingInputOutputMap):
        self.m_config = grouping_config
        self.inout_map = inout_map
        assert len(self.inout_map.group_sub_keys) == len(self.inout_map.member_settings), "group_sub_keys and member_settings must have the same length"
        self.group_map = {}
        for group_sub_key, member_setting in zip(self.inout_map.group_sub_keys, self.inout_map.member_settings):
            self.group_map[(member_setting.member_key.main_key, member_setting.member_key.sub_key)] = group_sub_key
            
    def do_transform(self, input_data : TransformInput, 
                     context : TransformContext = None) -> TransformOutput:
        if self.m_config.data_type == 'image':
            new_data_blocks = ImagesBlock()
        elif self.m_config.data_type == 'box':
            new_data_blocks = BoxesBlock()
        elif self.m_config.data_type == 'vector':
            new_data_blocks = VectorsBlock()
        else:
            raise ValueError(f"Unknown data type {self.m_config.data_type}")
        output = TransformOutput()
        output.data_blocks = copy.copy(input_data.data_blocks)
        output.params = GroupingParams()
        for main_key, sub_key in self.group_map.keys():
            sub_data = input_data.data_blocks[main_key].get_data(sub_key)
            sub_spec = input_data.data_blocks[main_key].get_spec(sub_key).clone()
            new_data_blocks.add_data(self.group_map[(main_key, sub_key)], sub_data, sub_spec)
        output.data_blocks[self.m_config.group_name] = new_data_blocks
        return output