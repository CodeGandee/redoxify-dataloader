from attrs import define, field
from typing import Union, List

from nvidia.dali.pipeline import DataNode as DALINode
import nvidia.dali.fn as fn
from redoxify.datablocks.DataBlock import DataBlock, DataSpec
import redoxify.functionals.cpu_ops as cpu_ops


@define(kw_only=True, eq=False)
class VectorSpec(DataSpec):
    length: Union[int, None] = field(default=None)  # known length of the vector, if None, it is unknown
    

class VectorsBlock(DataBlock):
    ''' A collection of named vectors, the vectors are assumed to be 1d, of shape (N,)
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass
    
    def subset_by_index(self, indices : Union[List[int], DALINode]) -> 'VectorsBlock':
        '''
        subset the vectors block by indices
        
        parameters
        ------------
        indices: List[int] or DALINode
            the indices to subset the vectors block
        
        return
        ---------
        VectorsBlock
            the subsetted vectors block
        '''
        out = VectorsBlock()
        
        for key, data in self.m_content.items():
            data_1d = fn.reinterpret(data, shape=[-1])
            data_subset = cpu_ops.dali_gather_along_axis(data_1d, indices, axis=0)
            out.add_data(key, data_subset, self.m_specs[key])
        return out
    
    def add_data(self, key: str, data: DALINode, spec: VectorSpec = None):
        '''
        add data to data block, and decode them as needed
        '''
        if spec is None:
            spec = VectorSpec()
        super().add_data(key, data, spec)

    def get_spec(self, key: str = None) -> VectorSpec:
        return super().get_spec(key)
        
    def get_vector(self, key: str = None) -> DALINode:
        '''
        Get the vectors
        '''
        if not self.m_content:
            return None
        
        if key is None:
            key = next(iter(self.m_content.keys()))
        raw_vector = self.m_content.get(key)
        if raw_vector is None:
            return raw_vector
        
        return raw_vector
