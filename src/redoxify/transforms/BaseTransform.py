from typing import Dict
from attrs import define, field
from redoxify.datablocks.DataBlock import DataBlock

class TransformContext:
    ''' context of the transformation, this is used to store the parameters of previous transformations
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

@define(kw_only=True, eq=False)
class TransformParams:
    ''' params of the transformation that is actually applied to the data
    '''
    pass
    
@define(kw_only=True, eq=False)
class TransformInput:
    ''' input of the transformation
    '''
    data_blocks : Dict[str, DataBlock] = field()
    
@define(kw_only=True, eq=False)
class TransformOutput:
    ''' output of the transformation
    '''
    data_blocks : Dict[str, DataBlock] = field(factory=dict)
    params : TransformParams = field(default = None)

class BaseTransform:
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass
    
    def do_transform(self, input_data : TransformInput, 
                     context : TransformContext = None) -> TransformOutput:
        raise NotImplementedError("do_transform is not implemented")

