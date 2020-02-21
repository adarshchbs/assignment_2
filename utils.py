import torch
import params
import numpy as np

def make_tensor(array : np.ndarray) -> torch.Tensor:
    array = torch.tensor(array)
    if(params.gpu_flag):
        array = array.cuda(params.gpu_name)

    return array

def cuda(model):
    if(params.gpu_flag):
        model.cuda(params.gpu_name)