"""Elementary functions"""

from src.fnode import Fnode
import numpy as np

def ln(x):
    try:
        if x._val <= 0:
            raise ValueError("The natural log is not defined for negative numbers")
        return Fnode(np.log(x._val), 1 / x._val * x._deriv)
    except AttributeError: #This is not a Fnode 
        return np.log(x)

def log(x, base):
    try:
        if x._val <= 0:
            raise ValueError("Log is not defined for negative numbers")
        return Fnode(np.log(x._val) / np.log(base), 1 /(x._val * np.log(base)) * x._deriv)
    except AttributeError:  # This is not a Fnode
        return np.log(x) / np.log(base)

def sqrt(x):
    try:
        if x._val <= 0:
            raise ValueError("sqrt is not real for negative numbers")
        return Fnode(x._val**0.5, 0.5 * x._val**-0.5 * x._deriv)
    except AttributeError:  # This is not a Fnode
        return np.sqrt(x)
