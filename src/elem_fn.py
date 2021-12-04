from src.fnode import Fnode
import numpy as np

"""Elementary functions"""

def tan(x):
    """
    Elementary function tan

    Parameters:
    x: Value or Fnode at which the tan function is to be evaluated

    Returns:
    For Fnodes, a new Fnode object with tan computed for the value and derivative
    For values, the tan function evaluated at that value
    """
    try:
        return Fnode(np.tan(x._val), (1 / (np.cos(x._val)**2)) * x._deriv)
    except AttributeError:
        return np.tan(x)

def arctan(x):
    """
    Elementary function arctan

    Parameters:
    x: Value or Fnode at which the arctan function is to be evaluated

    Returns:
    For Fnodes, a new Fnode object with arctan computed for the value and derivative
    For values, the arctan function evaluated at that value
    """
    try:
        return Fnode(np.arctan(x._val), (1 / (1 + x._val**2)) * x._deriv)
    except AttributeError:
        return np.arctan(x)

def tanh(x):
    """
    Elementary function tanh

    Parameters:
    x: Value or Fnode at which the tanh function is to be evaluated

    Returns:
    For Fnodes, a new Fnode object with tanh computed for the value and derivative
    For values, the tanh function evaluated at that value
    """
    try:
        return Fnode(np.tanh(x._val), (1 - np.tanh(x._val)**2) * x._deriv)
    except AttributeError:
        return np.tanh(x)

def ln(x):
    """
    Elementary function ln (natural log)

    Parameters:
    x: Value or Fnode at which the natural log function is to be evaluated. Cannot be negative

    Returns:
    For Fnodes, a new Fnode object with natural log computed for the value and derivative
    For values, the natural log function evaluated at that value
    """
    try:
        if x._val <= 0:
            raise ValueError("The natural log is not defined for negative numbers")
        return Fnode(np.log(x._val), 1 / x._val * x._deriv)
    except AttributeError:  # This is not a Fnode
        return np.log(x)

def log(x, base):
    """
    Elementary function log

    Parameters:
    x: Value or Fnode at which the log function is to be evaluated
    base: The base of the log function. Cannot be negative or zero

    Returns:
    For Fnodes, a new Fnode object with log computed for the value and derivative
    For values, the log function evaluated at that value
    """
    try:
        if x._val <= 0:
            raise ValueError("Log is not defined for negative numbers or zero")
        return Fnode(
            np.log(x._val) / np.log(base), 1 / (x._val * np.log(base)) * x._deriv
        )
    except AttributeError:  # This is not a Fnode
        return np.log(x) / np.log(base)

def sqrt(x):
    """
    Elementary function square root

    Parameters:
    x: Value or Fnode at which the square root function is to be evaluated. Cannot be negative.

    Returns:
    For Fnodes, a new Fnode object with natural log computed for the value and derivative
    For values, the natural log function evaluated at that value
    """
    try:
        if x._val < 0:
            raise ValueError("Square root is not real for negative numbers")
        return Fnode(x._val ** 0.5, 0.5 * x._val ** -0.5 * x._deriv)
    except AttributeError:  # This is not a Fnode
        return np.sqrt(x)

def sin(x):
    """
    Elementary function sin

    Parameters:
    x: Value or Fnode at which the sin function is to be evaluated

    Returns:
    For Fnodes, a new Fnode object with sin computed for the value and derivative
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
    x: Value or Fnode at which the arcsin function is to be evaluated

    Returns:
    For Fnodes, a new Fnode object with arcsin computed for the value and derivative
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
    x: Value or Fnode at which the sinh function is to be evaluated

    Returns:
    For Fnodes, a new Fnode object with sinh computed for the value and derivative
    For values, the sinh function evaluated at that value
    """
    try:
        return Fnode(np.sinh(x._val), np.cosh(x._val) * x._deriv)
    except AttributeError:
        return np.sinh(x)

def cos(x):
    """
    Elementary function cos

    Parameters:
    x: Value or Fnode at which the cos function is to be evaluated

    Returns:
    For Fnodes, a new Fnode object with cos computed for the value and derivative
    For values, the cos function evaluated at that value
    """
    try:
        return Fnode(np.cos(x._val), -1 * np.sin(x._val) * x._deriv)
    except AttributeError:  # This is not a Fnode
        return np.cos(x)

def arccos(x):
    """
    Elementary function arccos

    Parameters:
    x: Value or Fnode at which the arccos function is to be evaluated

    Returns:
    For Fnodes, a new Fnode object with arccos computed for the value and derivative
    For values, the arccos function evaluated at that value
    """
    try:
        if x._val > 1 or x._val < -1:
            raise ValueError("The domain of arccos is between -1 and 1 inclusive")
        return Fnode(np.arccos(x._val), (-1 / (1 - x._val ** 2) ** 0.5) * x._deriv)
    except AttributeError:  # This is not a Fnode
        return np.arccos(x)

def cosh(x):
    """
    Elementary function cosh

    Parameters:
    x: Value or Fnode at which the cosh function is to be evaluated

    Returns:
    For Fnodes, a new Fnode object with cosh computed for the value and derivative
    For values, the cosh function evaluated at that value
    """
    try:
        return Fnode(np.cosh(x._val), np.sinh(x._val) * x._deriv)
    except AttributeError:  # This is not a Fnode
        return np.cosh(x)

def exp(x):
    """
        Elementary function exp

        Parameters:
        x: Value or Fnode at which the exp function is to be evaluated

        Returns:
        For Fnodes, a new Fnode object with exp computed for the value and derivative
        For values, the exp function evaluated at that value
        """
    try:
        return Fnode(np.exp(x._val), np.exp(x._val) * x._deriv)
    except AttributeError:  # This is not a Fnode
        return np.exp(x)