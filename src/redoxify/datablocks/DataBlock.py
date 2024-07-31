import nvidia.dali.fn as fn
from nvidia.dali.types import DALIDataType
from nvidia.dali.pipeline import DataNode as DALINode
from typing import Dict, List
from attrs import define, field
import copy
from typing_extensions import Self

import redoxify.GlobalConst as C

@define(kw_only=True, eq=False)
class DataSpec:
    '''
    The specification of the data
    '''
    dtype : DALIDataType = field(default=DALIDataType.FLOAT)
    def clone(self) -> Self:
        return copy.copy(self)

class DataBlock:
    def __init__(self, *args, **kwargs):
        self.m_content : Dict[str, DALINode] = {}
        self.m_specs : Dict[str, DataSpec] = {}
        
    @property
    def content(self) -> Dict[str, DALINode]:
        return self.m_content
    
    @property
    def specs(self) -> Dict[str, DataSpec]:
        return self.m_specs
    
    def gpu(self) -> Self:
        '''
        Create a new DataBlock with all the data on GPU
        '''
        output = copy.copy(self)
        for key, val in output.m_content.items():
            output.m_content[key] = val.gpu()
        return output
        
    def pad(self):
        '''
        Pad all the data so that they have the same dimensions in batch
        '''
        raise NotImplementedError("pad is not implemented")
        # for key, val in self.m_content.items():
        #     self.m_content[key] = fn.pad(val, axes=self.m_pad_axes, fill_value=self.m_pad_fill_value)
    
    def get_keys(self) -> List[str]:
        '''
        Get the keys of the data
        '''
        return list(self.m_content.keys())
            
    def add_data(self, key: str, data: DALINode, spec: DataSpec, auto_cast: bool = None):
        '''
        Add data to the DataBlock
        '''
        if auto_cast is None:
            auto_cast = C.Defaults.AutoCastDataBlock
        self.m_specs[key] = spec
        if auto_cast:
            data = fn.cast(data, dtype=spec.dtype)
        self.m_content[key] = data
        
    def subset(self, keys: List[str]) -> Self:
        '''create a subset of this datablock containing given keys, return a new DataBlock
        '''
        # raise NotImplementedError("subset is not implemented")
        assert self.m_content, "no data in the data block"
        if keys is None:
            keys = list(self.m_content.keys())
        assert all([key in self.m_content for key in keys]), "some keys are not in the data block"
        new_db = self.__class__()
        for key in keys:
            new_db.add_data(key, self.m_content[key], self.m_specs[key].clone())
        return new_db
        
        
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
        assert self.m_content, "no data in the data block"
        assert key_after not in self.m_content, "key_after already exists"
        
        if key_before is None:
            key_before = next(iter(self.m_content.keys()))
            
        assert key_before in self.m_content, "key_before does not exist"
        
        if key_before in self.m_content:
            self.m_content[key_after] = self.m_content[key_before]
            del self.m_content[key_before]
            
        if key_before in self.m_specs:
            self.m_specs[key_after] = self.m_specs[key_before]
            del self.m_specs[key_before]
        
    def get_data(self, key: str = None) -> DALINode:
        '''
        Get data from the DataBlock
        
        parameters
        ------------
        key: str
            the key of the data, if None, get any data, this is useful when you know
            there is only one key in the data block
            
        return
        ----------
        DALINode
            the dali data, None if the key is not found
        '''
        if not self.m_content:
            return None
        
        if key is None:
            return next(iter(self.m_content.values()))
        else:
            return self.m_content.get(key)
    
    def get_spec(self, key: str = None) -> DataSpec:
        '''
        Get the spec of the data
        
        parameters
        ------------
        key: str
            the key to get the spec, if None, get any spec, this is useful when you know
            there is only one key in the data block
        return
        ----------
        DataSpec
            the spec of the data
        '''
        if not self.m_specs:
            return None
        
        if key is None:
            return next(iter(self.m_specs.values()))
        else:
            return self.m_specs.get(key)