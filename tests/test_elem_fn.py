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

def test_sin():
    v_0 = Fnode(np.pi)
    v_1 = elem.sin(v_0)
    try:
        assert v_1.val == 0, "sin function gave wrong value"
        assert v_1.deriv == -1, "sin function gave wrong derivative"
        assert elem.sin(np.pi) == 0, "sin function not working for non Fnodes"
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_arcsin():
    v_0 = Fnode(0.5)
    v_1 = elem.arc_sin(v_0)
    try:
        assert v_1.val == np.pi/6, "arc_sin function gave wrong value"
        assert v_1.deriv == 1/np.sqrt(1-0.5**2), "arc_sin function gave wrong derivative"
        assert elem.arc_sin(0.5) == np.pi/6, "arc_sin function not working for non Fnodes"
        with pytest.raises(ValueError):
            elem.arc_sin(Fnode(-3))
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_sinh():
    v_0 = Fnode(0)
    v_1 = elem.sinh(v_0)
    try:
        assert v_1.val == 0, "sinh function gave wrong value"
        assert v_1.deriv == 1, "sinh function gave wrong derivative"
        assert elem.sinh(0) == 0, "sinh function not working for non Fnodes"
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

def test_tan():
    v_0 = Fnode(7.0, 1.0)
    v_1 = elem.tan(v_0)

    assert v_1.val == np.tan(v_0.val)
    assert v_1.deriv == (1 / (np.cos(v_0.val)**2))

def test_arctan():
    v_0 = Fnode(7.0, 1.0)
    v_1 = elem.arctan(v_0)

    assert v_1.val == np.arctan(v_0.val)
    assert v_1.deriv == (1 / (1 + v_0.val**2))

def test_tanh():
    v_0 = Fnode(7.0, 1.0)
    v_1 = elem.tanh(v_0)

    assert v_1.val == np.tanh(v_0.val)
    assert v_1.deriv == 1 - np.tanh(v_0.val)**2

if __name__ == '__main__':
    test_cos()
    test_arccos()
    test_cosh()
    test_tan()
    test_arctan()
    test_tanh()
    test_sin()
    test_arcsin()
    test_sinh()
    test_ln()
    test_log()
    test_sqrt()
