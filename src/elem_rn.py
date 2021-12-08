from src.rnode import Rnode
import numpy as np

"""Elementary functions for reverse mode"""


def tan(x):
    z = Rnode(np.tan(x._val))
    x._children.append((1 / (np.cos(x._val) ** 2), z))
    return z


def arctan(x):
    z = Rnode(np.arctan(x._val))
    x._children.append((1 / (1 + (x._val ** 2)), z))
    return z


def tanh(x):
    z = Rnode(np.tanh(x._val))
    x._children.append((1 / (np.cosh(x._val) ** 2), z))
    return z


def ln(x):
    if(np.all(x._val > 0)):
        z = Rnode(np.log(x._val))
        x._children.append((1/x._val, z))
        return z
    else:
        raise ValueError("The natural log is not defined for negative numbers")


def log(x, base):
    if base <= 0:
        raise ValueError("Logarithm base cannot be less than or equal to zero")
    
    if(np.all(x._val > 0)):
        z = Rnode(np.log(x._val) / np.log(base))
        x._children.append((1/(x._val * np.log(base)), z))
        return z
    else:
        raise ValueError("Logarithm is not defined for negative numbers")


def sqrt(x):
    z = Rnode(x._val ** (1/2))
    x._children.append(((1/2) * (x._val ** (-1/2)), z))
    return z


def sin(x):
    z = Rnode(np.sin(x._val))
    x._children.append((np.cos(x._val), z))
    return z


def arcsin(x):
    z = Rnode(np.arcsin(x._val))
    x._children.append((1/(1 - x._val ** 2) ** 0.5, z))
    return z


def sinh(x):
    z = Rnode(np.sinh(x._val))
    x._children.append((np.cosh(x._val), z))
    return z


def cos(x):
    z = Rnode(np.cos(x._val))
    x._children.append((-np.sin(x._val), z))
    return z


def arccos(x):
    z = Rnode(np.arccos(x._val))
    x._children.append((-1 / (1 - x._val ** 2) ** 0.5, z))



def cosh(x):
    z = Rnode(np.cosh(x._val))
    x._children.append((np.sinh(x._val), z))
    return z


def exp(x):
    z = Rnode(np.exp(x._val))
    x._children.append((np.exp(x._val), z))
    return z
