"""Elementary functions"""

from fnode import Fnode

def ln(x):
    try:
        if x._val <= 0:
            raise ValueError("The natural log is not defined for negative numbers")
        return Fnode(np.log(x._val), 1 / x_val * x_deriv
    except AttributeError: #This is not a Fnode 
        return np.log(x)
