from src.auto_diff.reverse_mode.rnode import Rnode
import numpy as np

"""Elementary functions for reverse mode"""


def tan(x):
    """
       Elementary function tan

       Parameters:
       x: Value or Rnode at which the tan function is to be evaluated

       Returns:
       For Rnodes, a new Rnode object with tangent computed for the value
       For values, the tangent function evaluated at that value
       """
    z = Rnode(np.tan(x._val))
    x._children.append((1 / (np.cos(x._val) ** 2), z))
    return z


def arctan(x):
    """
      Elementary function arctan

      Parameters:
      x: Value or Rnode at which the arctan function is to be evaluated
      Returns:

      For Rnodes, a new Rnode object with arctan computed for the value
      For values, the arctan function evaluated at that value
      """
    z = Rnode(np.arctan(x._val))
    x._children.append((1 / (1 + (x._val ** 2)), z))
    return z


def tanh(x):
    """
     Elementary function tanh

     Parameters:
     x: Value or Rnode at which the tanh function is to be evaluated

     Returns:
     For Rnodes, a new Rnode object with tanh computed for the value
     For values, the tanh function evaluated at that value
     """
    z = Rnode(np.tanh(x._val))
    x._children.append((1 / (np.cosh(x._val) ** 2), z))
    return z


def ln(x):
    """
       Elementary function ln (natural log)

       Parameters:
       x: Value or Rnode at which the natural log function is to be evaluated. Cannot be negative

       Returns:
       For Rnodes, a new Rnode object with natural log computed for the value
       For values, the natural log function evaluated at that value
       """
    if(np.all(x._val > 0)):
        z = Rnode(np.log(x._val))
        x._children.append((1/x._val, z))
        return z
    else:
        raise ValueError("The natural log is not defined for negative numbers")


def log(x, base):
    """
       Elementary function log

       Parameters:
       x: Value or Rnode at which the log function is to be evaluated
       base: The base of the log function. Cannot be negative or zero

       Returns:
       For Rnodes, a new Rnode object with log computed for the value
       For values, the log function evaluated at that value
       """
    if base <= 0:
        raise ValueError("Logarithm base cannot be less than or equal to zero")
    
    if(np.all(x._val > 0)):
        z = Rnode(np.log(x._val) / np.log(base))
        x._children.append((1/(x._val * np.log(base)), z))
        return z
    else:
        raise ValueError("Logarithm is not defined for negative numbers")


def sqrt(x):
    """
       Elementary function square root

       Parameters:
       x: Value or Rnode at which the square root function is to be evaluated. Cannot be negative.

       Returns:
       For Rnodes, a new Rnode object with natural log computed for the value
       For values, the natural log function evaluated at that value
       """
    z = Rnode(x._val ** (1/2))
    x._children.append(((1/2) * (x._val ** (-1/2)), z))
    return z


def sin(x):
    """
       Elementary function sin

       Parameters:
       x: Value or Rnode at which the sin function is to be evaluated

       Returns:
       For Rnodes, a new Rnode object with sin computed for the value
       For values, the sin function evaluated at that value
       """
    z = Rnode(np.sin(x._val))
    x._children.append((np.cos(x._val), z))
    return z


def arcsin(x):
    """
    Elementary function arcsin

    Parameters:
    x: Value or Rnode at which the arcsin function is to be evaluated

    Returns:
    For Rnodes, a new Rnode object with arcsin computed for the value
    For values, the arcsin function evaluated at that value
    """
    z = Rnode(np.arcsin(x._val))
    x._children.append((1/(1 - x._val ** 2) ** 0.5, z))
    return z


def sinh(x):
    """
        Elementary function sinh

        Parameters:
        x: Value or Rnode at which the sinh function is to be evaluated

        Returns:
        For Rnodes, a new Rnode object with sinh computed for the value
        For values, the sinh function evaluated at that value
        """
    z = Rnode(np.sinh(x._val))
    x._children.append((np.cosh(x._val), z))
    return z


def cos(x):
    """
      Elementary function cos

      Parameters:
      x: Value or Rnode at which the cos function is to be evaluated

      Returns:
      For Rnodes, a new Rnode object with cos computed for the value and derivative
      For values, the cos function evaluated at that value
      """
    z = Rnode(np.cos(x._val))
    x._children.append((-np.sin(x._val), z))
    return z


def arccos(x):
    """
       Elementary function arccos

       Parameters:
       x: Value or Rnode at which the arccos function is to be evaluated

       Returns:
       For Rnodes, a new Rnode object with arccos computed for the value
       For values, the arccos function evaluated at that value
       """
    z = Rnode(np.arccos(x._val))
    x._children.append((-1 / (1 - x._val ** 2) ** 0.5, z))
    return z



def cosh(x):
    """
       Elementary function cosh

       Parameters:
       x: Value or Rnode at which the cosh function is to be evaluated

       Returns:
       For Rnodes, a new Rnode object with cosh computed for the value and derivative
       For values, the cosh function evaluated at that value
       """
    z = Rnode(np.cosh(x._val))
    x._children.append((np.sinh(x._val), z))
    return z


def exp(x):
    """
       Elementary function exp

       Parameters:
       x: Value or Rnode at which the exp function is to be evaluated

       Returns:
       For Rnodes, a new Rnode object with exp computed for the value
       For values, the exp function evaluated at that value
       """
    z = Rnode(np.exp(x._val))
    x._children.append((np.exp(x._val), z))
    return z
