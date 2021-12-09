from src.forward_mode.fnode import Fnode
import numpy as np

"""Elementary functions for forward mode"""

def tan(x):
    """
    Elementary function tan

    Parameters:
    x: Value or Fnode at which the tan function is to be evaluated

    Returns:
    For Fnodes, a new Fnode object with tangent computed for the value and derivative
    For values, the tangent function evaluated at that value
    """
    if isinstance(x, (int, float)):
        return np.tan(x)

    value = np.tan(x.val)
    deriv_dict = {}
    for var in x.get_vars():
        deriv_dict[var] = x.deriv[var] / (np.cos(x.val) ** 2)
    return Fnode(value, deriv_dict, x.var_name)
    

def arctan(x):
    """
    Elementary function arctan

    Parameters:
    x: Value or Fnode at which the arctan function is to be evaluated

    Returns:
    For Fnodes, a new Fnode object with arctan computed for the value and derivative
    For values, the arctan function evaluated at that value
    """
    if isinstance(x, (int, float)):
        return np.arctan(x)

    value = np.arctan(x.val)
    deriv_dict = {}
    for var in x.get_vars():
        deriv_dict[var] = x.deriv[var] * (1 / (1 + x._val**2))
    return Fnode(value, deriv_dict, x.var_name)


def tanh(x):
    """
    Elementary function tanh

    Parameters:
    x: Value or Fnode at which the tanh function is to be evaluated

    Returns:
    For Fnodes, a new Fnode object with tanh computed for the value and derivative
    For values, the tanh function evaluated at that value
    """
    if isinstance(x, (int, float)):
        return np.tanh(x)

    value = np.tanh(x.val)
    deriv_dict = {}
    for var in x.get_vars():
        deriv_dict[var] = x.deriv[var] * (1 - np.tanh(x._val)**2)
    return Fnode(value, deriv_dict, x.var_name)

def ln(x):
    """
    Elementary function ln (natural log)

    Parameters:
    x: Value or Fnode at which the natural log function is to be evaluated. Cannot be negative

    Returns:
    For Fnodes, a new Fnode object with natural log computed for the value and derivative
    For values, the natural log function evaluated at that value
    """
    if isinstance(x, (int, float)):
        return np.log(x)

    if x.val <= 0:
        raise ValueError("The natural log is not defined for negative numbers")

    value = np.log(x.val)
    deriv_dict = {}
    for var in x.get_vars():
        deriv_dict[var] = x.deriv[var] * (1 / x.val)
    return Fnode(value, deriv_dict, x.var_name)


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
    if isinstance(x, (int, float)):
        return np.log(x) / np.log(base)

    if x.val <= 0:
        raise ValueError("Log is not defined for negative numbers or zero")

    value = np.log(x.val) / np.log(base)
    deriv_dict = {}
    for var in x.get_vars():
        deriv_dict[var] = x.deriv[var] * 1 / (x.val * np.log(base))
    return Fnode(value, deriv_dict, x.var_name)


def sqrt(x):
    """
    Elementary function square root

    Parameters:
    x: Value or Fnode at which the square root function is to be evaluated. Cannot be negative.

    Returns:
    For Fnodes, a new Fnode object with natural log computed for the value and derivative
    For values, the natural log function evaluated at that value
    """
    if isinstance(x, (int, float)):
        return np.sqrt(x)

    if x.val < 0:
        raise ValueError("Square root is not real for negative numbers")

    value = np.sqrt(x.val)
    deriv_dict = {}
    for var in x.get_vars():
        deriv_dict[var] = x.deriv[var] * (0.5 * x._val ** -0.5)
    return Fnode(value, deriv_dict, x.var_name)


def sin(x):
    """
    Elementary function sin

    Parameters:
    x: Value or Fnode at which the sin function is to be evaluated

    Returns:
    For Fnodes, a new Fnode object with sin computed for the value and derivative
    For values, the sin function evaluated at that value
    """
    if isinstance(x, (int, float)):
        return np.sin(x)

    value = np.sin(x.val)
    deriv_dict = {}
    for var in x.get_vars():
        deriv_dict[var] = x.deriv[var] * np.cos(x.val)
    return Fnode(value, deriv_dict, x.var_name)


def arcsin(x):
    """
    Elementary function arcsin

    Parameters:
    x: Value or Fnode at which the arcsin function is to be evaluated

    Returns:
    For Fnodes, a new Fnode object with arcsin computed for the value and derivative
    For values, the arcsin function evaluated at that value
    """
    if isinstance(x, (int, float)):
        return np.arcsin(x)

    if x.val < -1 or x.val > 1:
        raise ValueError("The domain of arcsin is between -1 and 1 inclusive")

    value = np.arcsin(x.val)
    deriv_dict = {}
    for var in x.get_vars():
        deriv_dict[var] = x.deriv[var] * (1/((1 - x.val ** 2)) ** 0.5)
    return Fnode(value, deriv_dict, x.var_name)


def sinh(x):
    """
    Elementary function sinh

    Parameters:
    x: Value or Fnode at which the sinh function is to be evaluated

    Returns:
    For Fnodes, a new Fnode object with sinh computed for the value and derivative
    For values, the sinh function evaluated at that value
    """
    if isinstance(x, (int, float)):
        return np.sinh(x)

    value = np.sinh(x.val)
    deriv_dict = {}
    for var in x.get_vars():
        deriv_dict[var] = x.deriv[var] * np.cosh(x.val)
    return Fnode(value, deriv_dict, x.var_name)


def cos(x):
    """
    Elementary function cos

    Parameters:
    x: Value or Fnode at which the cos function is to be evaluated

    Returns:
    For Fnodes, a new Fnode object with cos computed for the value and derivative
    For values, the cos function evaluated at that value
    """
    if isinstance(x, (int, float)):
        return np.cos(x)

    value = np.cos(x.val)
    deriv_dict = {}
    for var in x.get_vars():
        deriv_dict[var] = x.deriv[var] * -1 * np.sin(x.val)
    return Fnode(value, deriv_dict, x.var_name)


def arccos(x):
    """
    Elementary function arccos

    Parameters:
    x: Value or Fnode at which the arccos function is to be evaluated

    Returns:
    For Fnodes, a new Fnode object with arccos computed for the value and derivative
    For values, the arccos function evaluated at that value
    """
    if isinstance(x, (int, float)):
        return np.arccos(x)

    if x.val > 1 or x.val < -1:
        raise ValueError("The domain of arccos is between -1 and 1 inclusive")

    value = np.arccos(x.val)
    deriv_dict = {}
    for var in x.get_vars():
        deriv_dict[var] = x.deriv[var] * (-1 / (1 - x._val ** 2) ** 0.5)
    return Fnode(value, deriv_dict, x.var_name)
    

def cosh(x):
    """
    Elementary function cosh

    Parameters:
    x: Value or Fnode at which the cosh function is to be evaluated

    Returns:
    For Fnodes, a new Fnode object with cosh computed for the value and derivative
    For values, the cosh function evaluated at that value
    """
    if isinstance(x, (int, float)):
        return np.cosh(x)

    value = np.cosh(x.val)
    deriv_dict = {}
    for var in x.get_vars():
        deriv_dict[var] = x.deriv[var] * np.sinh(x.val)
    return Fnode(value, deriv_dict, x.var_name)


def exp(x):
    """
    Elementary function exp

    Parameters:
    x: Value or Fnode at which the exp function is to be evaluated

    Returns:
    For Fnodes, a new Fnode object with exp computed for the value and derivative
    For values, the exp function evaluated at that value
    """
    if isinstance(x, (int, float)):
        return np.exp(x)

    value = np.exp(x.val)
    deriv_dict = {}
    for var in x.get_vars():
        deriv_dict[var] = x.deriv[var] * np.exp(x.val)
    return Fnode(value, deriv_dict, x.var_name)
