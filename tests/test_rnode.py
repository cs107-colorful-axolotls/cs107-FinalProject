import pytest
from src.reverse_mode.rnode import Rnode
import numpy as np

def test_pow():
    v_0 = Rnode(3)
    v_1 = v_0**2
    v_1.grad_value = 1.0

    assert v_1.val == 9, "__pow__ on rnode gave wrong value"
    assert v_0.grad() == 6, "__pow__ on rnode gave wrong derivative"

    v_0 = Rnode(3)
    v_1 = Rnode(2)
    res = v_0 ** v_1
    res.grad_value = 1

    assert res.val == 9
    assert v_0.grad() == 6


def test_rpow():
    v_0 = Rnode(3)
    v_1 = 3 ** v_0
    v_1.grad_value = 1.0
    try:
        assert v_1.val == 27, "__rpow__ on rnode gave wrong value"
        assert v_0.grad() == np.log(3) * 3 ** 3, "__rpow__ on rnode gave wrong derivative"
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_neg():
    v_0 = Rnode(3)
    v_1 = -v_0
    v_1.grad_value = 1.0
    try:
        assert v_1.val == -3, "__neg__ on rnode gave wrong value"
        assert v_0.grad() == -1, "__neg__ on rnode gave wrong derivative"
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_add():
    v_0 = Rnode(1)
    v_1 = Rnode(2)
    res1 = v_0 + 4
    res1.grad_value = 1

    assert res1.val == 5
    assert v_0.grad() == 0

    v_0 = Rnode(1)
    v_1 = Rnode(2)
    res2 = v_0 + v_1
    res2.grad_value = 1

    assert res2.val == 3
    assert v_0.grad() == 1


def test_radd():
    v_0 = Rnode(1)
    v_1 = 4 + v_0
    v_1.grad_value = 1

    assert v_1.val == 5
    assert v_0.grad() == 0


def test_sub():
    v_0 = Rnode(1)
    v_1 = Rnode(2)
    res1 = v_0 - 4
    res1.grad_value = 1

    assert res1.val == -3
    assert v_0.grad() == 0

    v_0 = Rnode(1)
    v_1 = Rnode(2)
    res2 = v_0 - v_1
    res2.grad_value = 1

    assert res2.val == -1
    assert v_0.grad() == 1


def test_rsub():
    v_0 = Rnode(1)
    v_1 = 4 - v_0
    v_1.grad_value = 1

    assert v_1.val == 3
    assert v_0.grad() == 0


def test_mul():
    v_0 = Rnode(1)
    v_1 = Rnode(2)
    res1 = v_0 * 5
    res1.grad_value = 1

    assert res1.val == 5
    assert v_0.grad() == 0

    v_0 = Rnode(1)
    v_1 = Rnode(2)
    res2 = v_0 * v_1
    res2.grad_value = 1

    assert res2.val == 2
    assert v_0.grad() == 2


def test_rmul():
    v_0 = Rnode(3)
    v_1 = 3 * v_0
    v_1.grad_value = 1

    assert v_1.val == 9
    assert v_0.grad() == 0


def test_truediv():
    v_0 = Rnode(4)
    v_1 = Rnode(8)
    res1 = v_0 / 5
    res1.grad_value = 1

    assert res1.val == 0.8
    assert v_0.grad() == 0

    v_0 = Rnode(4)
    v_1 = Rnode(8)
    res2 = v_0 / v_1
    res2.grad_value = 1

    assert res2.val == 0.5
    assert v_0.grad() == 0.125


def test_rtruediv():
    v_0 = Rnode(4)
    v_1 = 4 / v_0
    v_1.grad_value = 1

    assert v_1.val == 1
    assert v_0.grad() == 0


if __name__ == '__main__':
    test_pow()
    test_rpow()
    test_neg()
    test_add()
    test_radd()
    test_sub()
    test_rsub()
    test_mul()
    test_rmul()
    test_truediv()
    test_rtruediv()
