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
        with pytest.raises(ValueError):
            elem.ln(Fnode(-1))
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_log():
    v_0 = Fnode(100)
    v_1 = elem.log(v_0, 10)
    try:
        assert v_1.val == np.log10(100), "log function gave wrong value"
        assert v_1.deriv == 1 / (100 * np.log(10)), "log function gave wrong derivative"
        assert elem.log(100, 10) == 2, "log function not working for non Fnodes"
        with pytest.raises(ValueError):
            elem.log(Fnode(-1), 10)
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_sqrt():
    v_0 = Fnode(4)
    v_1 = elem.sqrt(v_0)
    try:
        assert v_1.val == 2, "sqrt function gave wrong value"
        assert v_1.deriv == 0.5 * 4 ** -0.5, "sqrt function gave wrong derivative"
        assert elem.sqrt(4) == 2, "sqrt function not working for non Fnodes"
        with pytest.raises(ValueError):
            elem.sqrt(Fnode(-1))
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_cos():
    v_0 = Fnode(6.0, 1.0)
    v_1 = elem.cos(v_0)

    assert v_1.val == np.cos(v_0.val)
    assert v_1.deriv == -1 * np.sin(v_0.val)

def test_arccos():
    v_0 = Fnode(0.5, 1.0)
    v_1 = elem.arccos(v_0)

    assert v_1.val == np.arccos(v_0.val)
    assert v_1.deriv == (-1 / (1 - v_0.val ** 2) ** 0.5)

    with pytest.raises(ValueError):
        elem.arccos(Fnode(6.0, 1.0))

def test_cosh():
    v_0 = Fnode(6.0, 1.0)
    v_1 = elem.cosh(v_0)

    assert v_1.val == np.cosh(v_0.val)
    assert v_1.deriv == np.sinh(v_0.val)

test_ln()
test_log()
test_sqrt()

test_cos()
test_arccos()
test_cosh()
