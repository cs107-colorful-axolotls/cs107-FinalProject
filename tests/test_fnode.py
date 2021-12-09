from typing import Type
import pytest
from src.fnode import Fnode
import numpy as np

def test_mul_const():
    v_0 = Fnode(10, 20, 'x')
    v_1 = v_0 * 5
    assert v_1.val == 50
    assert v_1.deriv['x'] == 100

def test_mul_obj():
    v_0 = Fnode(2, 3, 'v_0')
    v_1 = Fnode(5, 10, 'v_1')
    res = v_0 * v_1
    assert res.val == 10
    assert res.deriv['v_0'] == 15
    assert res.deriv['v_1'] == 20

def test_rmul():
    x = Fnode(2.0, 1.0, 'x')
    f1 = 3 * x
    assert f1.val == 6
    assert f1.deriv['x'] == 3

def test_truediv_const():
    v_0 = Fnode(6, 5, 'x')
    v_1 = v_0 / 2.0

    assert v_1.val == 3.0
    assert v_1.deriv['x'] == 2.5

def test_rtruediv():
    v_0 = Fnode(5, 5, 'x')
    v_1 = 2.0 / v_0

    assert v_1.val == 0.4
    assert v_1.deriv['x'] == -0.4

def test_truediv_obj():
    x = Fnode(2, 1, 'x')
    y = Fnode(5, 1, 'y')
    res = x / y
    assert res.val == [0.4]
    assert res.deriv['x'] == [0.2]
    assert res.deriv['y'] == [-0.08]

    with pytest.raises(ZeroDivisionError):
        v_0 = Fnode(2, 3, 'x')
        v_1 = Fnode(0, 10, 'y')
        res = v_0 / v_1

def test_pow():
    v_0 = Fnode(3, 2, 'x')
    v_1 = v_0 ** 2
    try:
        assert v_1.val == 9, "__pow__ on fnode gave wrong value"
        assert v_1.deriv['x'] == 12, "__pow__ on fnode gave wrong derivative"
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_pow_obj():
    v_0 = Fnode(3, 1, 'x')
    v_1 = Fnode(3, 1, 'y')
    res1 = v_0 ** v_1

    assert res1.val == 27
    assert res1.deriv['x'] == 27
    assert np.around(res1.deriv['y'], 6) == np.around(27 * np.log(3), 6)

def test_rpow():
    v_0 = Fnode(3, 2, 'x')
    v_1 = 3 ** v_0
    try:
        assert v_1.val == 27, "__rpow__ on fnode gave wrong value"
        assert v_1.deriv['x'] == np.log(3) * 9 * 2, "__rpow__ on fnode gave wrong derivative"
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_neg():
    v_0 = Fnode(2, 3, 'x')
    v_1 = -v_0
    try:
        assert v_1.val == -2, "__neg__ on fnode gave wrong value"
        assert v_1.deriv['x'] == -3, "__neg__ on fnode gave wrong derivative"
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_add():
    v_0 = Fnode(3, 1, 'x')
    v_1 = v_0 + 4
    try:
        assert v_1.val == 7, "__add__ on fnode gave wrong value"
        assert v_1.deriv['x'] == 1, "__add__ on fnode gave wrong derivative"
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_radd():
    v_0 = Fnode(3, 1, 'x')
    v_1 = 4 + v_0
    try:
        assert v_1.val == 7, "__add__ on fnode gave wrong value"
        assert v_1.deriv['x'] == 1, "__add__ on fnode gave wrong derivative"
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_sub():
    v_0 = Fnode(6, 3, 'x')
    v_1 = v_0 - 2
    try:
        assert v_1.val == 4, "__sub__ on fnode gave wrong value"
        assert v_1.deriv['x'] == 3, "__sub__ on fnode gave wrong derivative"
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_sub_obj():
    v_0 = Fnode(6, 3, 'x')
    v_1 = Fnode(6, 3, 'y')
    res = v_0 - v_1

    assert res.val == 0
    assert res.deriv['x'] == 3
    assert res.deriv['y'] == -3

def test_rsub():
    v_0 = Fnode(6, 3, 'x')
    v_1 = 2 - v_0
    try:
        assert v_1.val == -4, "__sub__ on fnode gave wrong value"
        assert v_1.deriv['x'] == -3, "__sub__ on fnode gave wrong derivative"
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_linear():
    v_0 = Fnode([4, 5, 6], [7, 8, 9], 'x')
    v_1 = Fnode([1, 2, 3], [11, 22, 33], 'y')
    res = v_0 + v_1
    assert np.array_equal(res.val, np.array([5, 7, 9]))
    assert np.array_equal(res.deriv["x"], np.array([7, 8, 9]))
    assert np.array_equal(res.deriv["y"], np.array([11, 22, 33]))

def test_error():
    with pytest.raises(TypeError):
        v_0 = Fnode("Hello", 1, 'x')

    with pytest.raises(TypeError):
        v_0 = Fnode(1, Fnode(1, 1, 'x'), 'x')

    v_0 = Fnode(1, 1, 'x')
    with pytest.raises(TypeError):
        v_1 = v_0 + "Hello"

    with pytest.raises(TypeError):
        v_1 = v_0 - "Hello"

    with pytest.raises(TypeError):
        v_1 = v_0 * "Hello"

    with pytest.raises(TypeError):
        v_1 = v_0 / "Hello"

    with pytest.raises(TypeError):
        v_1 = v_0 ** "Hello"

    with pytest.raises(TypeError):
        v_1 = "Hello" ** v_0
        
if __name__ == '__main__':
    test_mul_const()
    test_mul_obj()
    test_rmul()
    test_truediv_const()
    test_rtruediv()
    test_truediv_obj()
    test_add()
    test_radd()
    test_sub()
    test_pow()
    test_rpow()
    test_neg()
    test_error()
    test_linear()
    test_sub_obj()
    test_pow_obj()
    test_rsub()
