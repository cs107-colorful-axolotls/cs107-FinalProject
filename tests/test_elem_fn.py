import pytest
from src.auto_diff.forward_mode.fnode import Fnode
import src.auto_diff.forward_mode.elem as elem
import numpy as np

def test_ln():
    v_0 = Fnode(2, 1, 'x')
    v_1 = elem.ln(v_0)
    try:
        assert v_1.val == np.log(2), "ln function gave wrong value"
        assert v_1.deriv['x'] == 0.5, "ln function gave wrong derivative"
        assert elem.ln(1) == 0, "ln function not working for non Fnodes"
        with pytest.raises(ValueError):
            elem.ln(Fnode(-1))
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_log():
    v_0 = Fnode(100, 1, 'x')
    v_1 = elem.log(v_0, 10)
    try:
        assert v_1.val == np.log10(100), "log function gave wrong value"
        assert v_1.deriv['x'] == 1 / (100 * np.log(10)), "log function gave wrong derivative"
        assert elem.log(100, 10) == 2, "log function not working for non Fnodes"
        with pytest.raises(ValueError):
            elem.log(Fnode(-1), 10)
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_sqrt():
    v_0 = Fnode(4, 1, 'x')
    v_1 = elem.sqrt(v_0)
    try:
        assert v_1.val == 2, "sqrt function gave wrong value"
        assert v_1.deriv['x'] == 0.5 * 4 ** -0.5, "sqrt function gave wrong derivative"
        assert elem.sqrt(4) == 2, "sqrt function not working for non Fnodes"
        with pytest.raises(ValueError):
            elem.sqrt(Fnode(-1))
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_sin():
    v_0 = Fnode(np.pi, 1, 'x')
    v_1 = elem.sin(v_0)
    try:
        assert v_1.val == np.sin(v_0.val), "sin function gave wrong value"
        assert v_1.deriv['x'] == -1, "sin function gave wrong derivative"
        assert elem.sin(np.pi) == np.sin(np.pi), "sin function not working for non Fnodes"
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_arcsin():
    v_0 = Fnode(0.5, 1, 'x')
    v_1 = elem.arcsin(v_0)
    try:
        assert v_1.val == np.arcsin(v_0.val), "arc_sin function gave wrong value"
        assert v_1.deriv['x'] == 1/((1 - v_0.val ** 2) ** 0.5), "arc_sin function gave wrong derivative"
        assert elem.arcsin(0.5) == np.arcsin(0.5), "arc_sin function not working for non Fnodes"
        with pytest.raises(ValueError):
            elem.arcsin(Fnode(-3))
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_sinh():
    v_0 = Fnode(0, 1, 'x')
    v_1 = elem.sinh(v_0)
    try:
        assert v_1.val == 0, "sinh function gave wrong value"
        assert v_1.deriv['x'] == 1, "sinh function gave wrong derivative"
        assert elem.sinh(0) == 0, "sinh function not working for non Fnodes"
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_cos():
    v_0 = Fnode(6.0, 1.0, 'x')
    v_1 = elem.cos(v_0)

    assert v_1.val == np.cos(v_0.val)
    assert v_1.deriv['x'] == -1 * np.sin(v_0.val)
    assert elem.cos(np.pi) == np.cos(np.pi)

def test_arccos():
    v_0 = Fnode(0.5, 1.0, 'x')
    v_1 = elem.arccos(v_0)

    assert v_1.val == np.arccos(v_0.val)
    assert v_1.deriv['x'] == (-1 / (1 - v_0.val ** 2) ** 0.5)
    assert elem.arccos(0) == np.arccos(0)

    with pytest.raises(ValueError):
        elem.arccos(Fnode(6.0, 1.0))

def test_cosh():
    v_0 = Fnode(6.0, 1.0, 'x')
    v_1 = elem.cosh(v_0)

    assert v_1.val == np.cosh(v_0.val)
    assert v_1.deriv['x'] == np.sinh(v_0.val)
    assert elem.cosh(0) == np.cosh(0)

def test_tan():
    v_0 = Fnode(7.0, 1.0, 'x')
    v_1 = elem.tan(v_0)

    assert v_1.val == np.tan(v_0.val)
    assert v_1.deriv['x'] == (1 / (np.cos(v_0.val)**2))
    assert elem.tan(np.pi) == np.tan(np.pi)

def test_arctan():
    v_0 = Fnode(7.0, 1.0, 'x')
    v_1 = elem.arctan(v_0)

    assert v_1.val == np.arctan(v_0.val)
    assert v_1.deriv['x'] == (1 / (1 + v_0.val**2))
    assert elem.arctan(np.pi) == np.arctan(np.pi)

def test_tanh():
    v_0 = Fnode(7.0, 1.0,'x')
    v_1 = elem.tanh(v_0)

    assert v_1.val == np.tanh(v_0.val)
    assert v_1.deriv['x'] == 1 - np.tanh(v_0.val)**2
    assert elem.tanh(np.pi) == np.tanh(np.pi)

def test_linear():
    v_0 = Fnode(3.0, 1.0, 'x')
    v_1 = elem.sin(3 * v_0 + 1)

    assert v_1.val == np.sin(10.0)
    assert v_1.deriv['x'] == 3 * np.cos(10.0)

def test_exp():
    v_0 = Fnode(3.0, 1.0, 'x')
    v_1 = elem.exp(v_0)

    assert v_1.val == np.exp(3)
    assert v_1.deriv['x'] == np.exp(3)
    assert elem.exp(3) == np.exp(3)

def test_logistic():
    v_0 = Fnode(3.0, 1.0, 'x')
    v_1 = elem.logistic_fn(v_0, 1, 1, 1)

    assert v_1.val == 1 / (1 + np.exp(-1*(3 -1)))
    assert v_1.deriv['x'] ==  1 * np.exp(3) / (1+np.exp(3))**2

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
    test_linear()
    test_exp()
    test_logistic()
