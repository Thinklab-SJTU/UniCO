import numpy as np
from jittor import Var
from typing import Union


def check_dim(array: Union[np.ndarray, Var], dim: int):
    if isinstance(array, np.ndarray):
        if array.ndim != dim:
            raise ValueError("Dimension mismatch!")
    if isinstance(array, Var):
        if array.ndim != dim:
            raise ValueError("Dimension mismatch!")
        
def to_numpy(
    x: Union[np.ndarray, Var, list]
) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, Var):
        return x.numpy()
    elif isinstance(x, list):
        return np.array(x)