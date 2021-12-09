import pytest
from src.reverse_mode.rnode import Rnode
import src.reverse_mode.elem as elem
import numpy as np

def test_ln():
    v_0 = Rnode(10)
    v_1 = elem.ln(v_0)**5
    v_1.grad_value = 1
    assert v_0.grad() == 5*(np.log(10)**4)/10


def test_log():
    v_0 = Rnode(10)
    v_1 = elem.log(v_0, 2)**5
    v_1.grad_value = 1
    assert v_0.grad() == 5*((np.log(10)/np.log(2))**4)/(np.log(2)*10)


def test_sqrt():
    v_0 = Rnode(10)
    v_1 = elem.sqrt(v_0)**5
    v_1.grad_value = 1
    assert np.around(v_0.grad(), 6) == np.around(5*(np.sqrt(10)**4)/(2*np.sqrt(10)), 6)


def test_sin():
    v_0 = Rnode(10)
    v_1 = elem.sin(v_0)**5
    v_1.grad_value = 1
    assert v_0.grad() == 5*(np.sin(10)**4)*np.cos(10)


def test_arcsin():
    v_0 = Rnode(0.5)
    v_1 = elem.arcsin(v_0)**5
    v_1.grad_value = 1
    assert v_0.grad() == 5*(np.arcsin(0.5)**4)/np.sqrt(1 - 0.5 ** 2)


def test_sinh():
    v_0 = Rnode(10)
    v_1 = elem.sinh(v_0)**5
    v_1.grad_value = 1
    assert v_0.grad() == 5*(np.sinh(10)**4)*np.cosh(10)


def test_cos():
    v_0 = Rnode(10)
    v_1 = elem.cos(v_0)**5
    v_1.grad_value = 1
    assert v_0.grad() == -5*(np.cos(10)**4)*np.sin(10)


def test_arccos():
    v_0 = Rnode(0.5)
    v_1 = elem.arccos(v_0)**5
    v_1.grad_value = 1
    assert v_0.grad() == -5*(np.arccos(0.5)**4)/np.sqrt(1 - 0.5 ** 2)


def test_cosh():
    v_0 = Rnode(10)
    v_1 = elem.cosh(v_0)**5
    v_1.grad_value = 1
    assert v_0.grad() == 5*(np.cosh(10)**4)*np.sinh(10)


def test_tan():
    v_0 = Rnode(10)
    v_1 = elem.tan(v_0)**5
    v_1.grad_value = 1
    assert v_0.grad() == 5*(np.tan(10)**4)/(np.cos(10) ** 2)


def test_arctan():
    v_0 = Rnode(0.1)
    v_1 = elem.arctan(v_0)**5
    v_1.grad_value = 1
    assert v_0.grad() == 5*(np.arctan(0.1)**4)/(1 + 0.1 ** 2)


def test_tanh():
    v_0 = Rnode(10)
    v_1 = elem.tanh(v_0)**5
    v_1.grad_value = 1
    assert v_0.grad() == 5*(np.tanh(10)**4)/(np.cosh(10) ** 2)


def test_exp():
    v_0 = Rnode(10)
    v_1 = elem.exp(v_0)
    v_1.grad_value = 1
    assert v_0.grad() == np.exp(10)


def test_error():
    with pytest.raises(ValueError):
        v_0 = Rnode(-3)
        v_1 = elem.ln(v_0)

    with pytest.raises(ValueError):
        v_0 = Rnode(-3)
        v_1 = elem.log(v_0, 2)

    with pytest.raises(ValueError):
        v_0 = Rnode(3)
        v_1 = elem.log(v_0, -3)


if __name__ == '__main__':
    test_ln()
    test_log()
    test_sqrt()
    test_sin()
    test_arcsin()
    test_sinh()
    test_cos()
    test_arccos()
    test_cosh()
    test_tan()
    test_arctan()
    test_tanh()
    test_exp()
    test_error()
