import pytest
from src.fnode import Fnode

def test_mul_const():
    v_0 = Fnode(10, 20)
    v_1 = v_0 * 5
    assert v_1.val == 50
    assert v_1.deriv == 100

def test_mul_obj():
    v_0 = Fnode(2, 3)
    v_1 = Fnode(5, 10)
    res = v_0 * v_1
    assert res.val == 10
    assert res.deriv == 35

def test_rmul():
    x = Fnode(2.0, 1.0)
    f1 = 3 * x
    assert f1.val == 6
    assert f1.deriv == 3

def test_truediv_const():
    v_0 = Fnode(6, 5)
    v_1 = v_0 / 2.0

    assert v_1.val == 3.0
    assert v_1.deriv == 2.5

def test_truediv_obj():
    v_0 = Fnode(2, 3)
    v_1 = Fnode(5, 10)
    res = v_0 / v_1
    assert res.val == 0.4
    assert res.deriv == -0.2

    with pytest.raises(ZeroDivisionError):
        v_0 = Fnode(2, 3)
        v_1 = Fnode(0, 10)
        res = v_0 / v_1

def test_pow():
    v_0 = Fnode(3, 2)
    v_1 = v_0 ** 2
    try:
        assert v_1.val == 9, "__pow__ on fnode gave wrong value"
        assert v_1.deriv == 12, "__pow__ on fnode gave wrong derivative"
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_rpow():
    v_0 = Fnode(3, 2)
    v_1 = 3 ** v_0
    try:
        assert v_1.val == 27, "__rpow__ on fnode gave wrong value"
        assert v_1.deriv == 0, "__rpow__ on fnode gave wrong derivative"
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_neg():
    v_0 = Fnode(2, 3)
    v_1 = -v_0
    try:
        assert v_1.val == -2, "__neg__ on fnode gave wrong value"
        assert v_1.deriv == -3, "__neg__ on fnode gave wrong derivative"
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_add():
    v_0 = Fnode(3, 1)
    v_1 = v_0 + 4
    try:
        assert v_1.val == 7, "__add__ on fnode gave wrong value"
        assert v_1.deriv == 1, "__add__ on fnode gave wrong derivative"
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_radd():
    v_0 = Fnode(3, 1)
    v_1 = 4 + v_0
    try:
        assert v_1.val == 7, "__add__ on fnode gave wrong value"
        assert v_1.deriv == 1, "__add__ on fnode gave wrong derivative"
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_sub():
    v_0 = Fnode(6, 3)
    v_1 = v_0 - 2
    try:
        assert v_1.val == 4, "__sub__ on fnode gave wrong value"
        assert v_1.deriv == 3, "__sub__ on fnode gave wrong derivative"
    except AssertionError as e:
        print(e)
        raise AssertionError

def test_rsub():
    v_0 = Fnode(6, 3)
    v_1 = 2 - v_0
    try:
        assert v_1.val == -4, "__sub__ on fnode gave wrong value"
        assert v_1.deriv == -3, "__sub__ on fnode gave wrong derivative"
    except AssertionError as e:
        print(e)
        raise AssertionError
        
if __name__ == '__main__':
    test_mul_const()
    test_mul_obj()
    test_rmul()
    test_truediv_const()
    test_truediv_obj()
    test_add()
    test_radd()
    test_sub()
    test_pow()
    test_rpow()
    test_neg()
