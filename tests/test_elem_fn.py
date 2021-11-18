import pytest
from src.fnode import Fnode
import numpy as np
import src.elem_fn as elem

def test_ln():
    v_0 = Fnode(2)
    v_1 = elem.ln(v_0)
    try:
        assert v_1.val == np.log(2), "ln function gave wrong value"
        assert v_1.deriv == 0.5, "ln function gave wrong derivative"
        assert elem.ln(1) == 0, "ln function not working for non Fnodes"
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_log():
    v_0 = Fnode(100)
    v_1 = elem.log(v_0, 10)
    try:
        assert v_1.val == np.log10(100), "log function gave wrong value"
        assert v_1.deriv == 1/(100*np.log(10)), "log function gave wrong derivative"
        assert elem.log(100,10) == 2, "log function not working for non Fnodes"
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_sqrt():
    v_0 = Fnode(4)
    v_1 = elem.sqrt(v_0)
    try:
        assert v_1.val == 2, "sqrt function gave wrong value"
        assert v_1.deriv == 0.5 * 4**-0.5, "sqrt function gave wrong derivative"
        assert elem.sqrt(4) == 2, "sqrt function not working for non Fnodes"
    except AssertionError as e:
        print(e)
        raise AssertionError

test_ln()
test_log()
test_sqrt()
