from src.rnode import Rnode
import numpy as np

"""Elementary functions"""
def sin(x):
    """
    Elementary function sin

    Parameters:
    x: Value or Rnode at which the sin function is to be evaluated

    Returns:
    For Rnodes, a new Rnode object with sin computed
    For values, the sin function evaluated at that value
    """
    try:
        return Fnode(np.sin(x._val), np.cos(x._val) * x._deriv)
    except AttributeError:
        return np.sin(x)

def arcsin(x):
    """
    Elementary function arcsin

    Parameters:
    x: Value or Rnode at which the arcsin function is to be evaluated

    Returns:
    For Rnodes, a new Rnode object with arcsin computed
    For values, the arcsin function evaluated at that value
    """
    try:
        if x._val < -1 or x._val > 1: # range of valid values for arc_sin values is -1 ≤ x ≤ 1
            raise ValueError("The domain of arcsin is between -1 and 1 inclusive")
        return Fnode(np.arcsin(x._val), (1/((1 - x._val ** 2)) ** 0.5) * x._deriv)
    except AttributeError:  # This is not a Fnode
        return np.arcsin(x)

def sinh(x):
    """
    Elementary function sinh

    Parameters:
    x: Value or Rnode at which the sinh function is to be evaluated

    Returns:
    For Rnodes, a new Fnode object with sinh computed
    For values, the sinh function evaluated at that value
    """
    try:
        return Fnode(np.sinh(x._val), np.cosh(x._val) * x._deriv)
    except AttributeError:
        return np.sinh(x)