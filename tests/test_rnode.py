import pytest
from src.rnode import Rnode
import numpy as np

def test_pow():
    v_0 = Rnode(3)
    v_1 = v_0**2
    v_1.grad_value = 1.0
    try:
        assert v_1.val == 9, "__pow__ on rnode gave wrong value"
        assert v_0.grad() == 6, "__pow__ on rnode gave wrong derivative"
    except AssertionError as e:
        print(e)
        raise AssertionError


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


if __name__ == '__main__':
    test_pow()
    test_rpow()
    test_neg()
