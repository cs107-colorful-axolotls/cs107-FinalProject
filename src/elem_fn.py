"""Elementary functions"""

from src.fnode import Fnode
import numpy as np

def tan(x):
    try:
        return Fnode(np.tan(x._val), (1 / (np.cos(x._val)**2)) * x._deriv)
    except AttributeError:
        return np.tan(x)

def arctan(x):
    try:
        return Fnode(np.arctan(x._val), (1 / (1 + x._val**2)) * x._deriv)
    except AttributeError:
        return np.arctan(x)

def tanh(x):
    try:
        return Fnode(np.tanh(x._val), (1 - np.tanh(x._val)**2) * x._deriv)
    except AttributeError:
        return np.tanh(x)

def ln(x):
    try:
        if x._val <= 0:
            raise ValueError("The natural log is not defined for negative numbers")
        return Fnode(np.log(x._val), 1 / x._val * x._deriv)
    except AttributeError:  # This is not a Fnode
        return np.log(x)

def log(x, base):
    try:
        if x._val <= 0:
            raise ValueError("Log is not defined for negative numbers")
        return Fnode(
            np.log(x._val) / np.log(base), 1 / (x._val * np.log(base)) * x._deriv
        )
    except AttributeError:  # This is not a Fnode
        return np.log(x) / np.log(base)

def sqrt(x):
    try:
        if x._val < 0:
            raise ValueError("Square root is not real for negative numbers")
        return Fnode(x._val ** 0.5, 0.5 * x._val ** -0.5 * x._deriv)
    except AttributeError:  # This is not a Fnode
        return np.sqrt(x)

def sin(x):
    try:
        return Fnode(np.sin(x._val), np.cos(x._val))
    except AttributeError:
        return np.sin(x)

def arcsin(x):
    try:
        if x._val < -1 or x._val > 1: # range of valid values for arc_sin values is -1 ≤ x ≤ 1
            raise ValueError("The domain of arcsin is between -1 and 1 inclusive")
        return Fnode(np.arcsin(x._val), (1/((1 - x._val ** 2)) ** 0.5) * x._deriv)
    except AttributeError:  # This is not a Fnode
        return np.arcsin(x)

def sinh(x):
    try:
        return Fnode(np.sinh(x._val), np.cosh(x._val) * x._deriv)
    except AttributeError:
        return np.sinh(x)

def cos(x):
    try:
        return Fnode(np.cos(x._val), -1 * np.sin(x._val) * x._deriv)
    except AttributeError:  # This is not a Fnode
        return np.cos(x)

def arccos(x):
    try:
        if x._val > 1 or x._val < -1:
            raise ValueError("The domain of arccos is between -1 and 1 inclusive")
        return Fnode(np.arccos(x._val), (-1 / (1 - x._val ** 2) ** 0.5) * x._deriv)
    except AttributeError:  # This is not a Fnode
        return np.arccos(x)

def cosh(x):
    try:
        return Fnode(np.cosh(x._val), np.sinh(x._val) * x._deriv)
    except AttributeError:  # This is not a Fnode
        return np.cosh(x)
