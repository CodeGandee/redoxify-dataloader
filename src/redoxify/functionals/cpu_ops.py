# functions used in dali python_function
import numpy as np
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import DataNode as DALINode

def dali_gather_along_axis(data : DALINode, indices: DALINode, axis:int = 0):
    '''
    gather the data by indices
    '''
    func = fn.python_function(
        data, indices, axis,
        function=_dali_gather_along_axis,
        num_outputs=1,
        device='cpu'
    )
    return func

def _dali_gather_along_axis(data : np.ndarray, indices: np.ndarray, axis:int):
    '''
    gather the data by indices
    '''
    return data.take(indices.astype(int), axis=axis)