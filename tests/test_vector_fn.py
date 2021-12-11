import pytest
from src.auto_diff.forward_mode.fnode import Fnode
import src.auto_diff.forward_mode.elem as elem
from src.auto_diff.forward_mode.vector_fn import Vector_Fn
import numpy as np

def test_vector():
    x = Fnode(1, 1, 'x')
    y = Fnode(2, 1, 'y')
    f1 = 2 * x ** 2 + 3 * y ** 2
    f2 = elem.sin(x + y)
    res = Vector_Fn([f1, f2])

    assert np.array_equal(res.get_vals()[0], np.array([14, np.sin(3)]))
    assert np.array_equal(res.get_deriv()[1][0][:, res.get_deriv()[0].index('x')], np.array([4, np.cos(3)]))
    assert np.array_equal(res.get_deriv()[1][0][:, res.get_deriv()[0].index('y')], np.array([12, np.cos(3)]))

if __name__ == '__main__':
    test_vector()