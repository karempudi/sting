import torch
from sting.utils.types import RecursiveNamespace
from typing import Union


def get_model(param: RecursiveNamespace):
    """
    Function to return a list of models and a list of 
    transforms applied before and after, each of the model is run 

    Arguments:
        param: parameters used for the analysis
    
    Returns:
        models: a list of models
        before_transforms: a list of transforms corresponding to each model
        after_transfroms: a list of transforms corresponding to each model
                to get to final result, to recover inferences on the original
                data
    """
    return None, None, None