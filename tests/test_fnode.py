import pytest
from src.fnode import Fnode
import numpy as np

def test_pow():
    v_0 = Fnode(2)
    v_1 = v_0**2
    try:
        assert v_1.val == 4, "__pow__ on fnode gave wrong value"
        assert v_1.deriv == 4, "__pow__ on fnode gave wrong derivative"
    except AssertionError as e:
        print(e)
        raise AssertionError


def test_neg():
    v_0 = Fnode(1)
    v_1 = -v_0
    try:
        assert v_1.val == -1, "__neg__ on fnode gave wrong value"
        assert v_1.deriv == -1, "__neg__ on fnode gave wrong derivative"
    except AssertionError as e:
        print(e)
        raise AssertionError

test_pow()
test_neg()
