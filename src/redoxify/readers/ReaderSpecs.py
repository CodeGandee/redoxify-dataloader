# specifications about how to read and interpret data from files
from typing import Union
from attrs import define, field
import attrs.validators as av
from redoxify.datablocks import (
    ImagesBlock, BoxesBlock, VectorsBlock,
    ImageSpec, BoxSpec, VectorSpec
)
# import redoxify.GlobalConst as C

DataSpecType = Union[ImageSpec, BoxSpec, VectorSpec]

# @define(kw_only=True, eq=False)
# class TF_ImageSpec:
#     encoding : str = field(validator=av.in_(C.ImageEncoding.get_all_encodings()))
#     width : Union[int, None] = field(default=None)
#     height : Union[int, None] = field(default=None)
#     channel : Union[int, None] = field(default=None)
    