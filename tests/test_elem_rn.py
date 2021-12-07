import pytest
from src.rnode import Rnode
import src.elem_rn as elem
import numpy as np

def test_sin():
    v_0 = Rnode(np.pi)
    v_1 = elem.sin(v_0)
    #try:
       # assert v_1.val == np.sin(v_0.val), "sin function gave wrong value"
       # assert v_1.deriv == -1, "sin function gave wrong derivative"
       # assert elem.sin(np.pi) == np.sin(np.pi), "sin function not working for non Fnodes"
    #except AssertionError as e:
    #    print(e)
    #    raise AssertionError

def test_arcsin():
    v_0 = Rnode(0.5)
    v_1 = elem.arcsin(v_0)
   # try:
   #     assert v_1.val == np.arcsin(v_0.val), "arc_sin function gave wrong value"
   #     assert v_1.deriv == 1/((1 - v_0.val ** 2) ** 0.5), "arc_sin function gave wrong derivative"
   #     assert elem.arcsin(0.5) == np.arcsin(0.5), "arc_sin function not working for non Fnodes"
   #     with pytest.raises(ValueError):
   #         elem.arcsin(Fnode(-3))
   # except AssertionError as e:
   #     print(e)
   #     raise AssertionError

def test_sinh():
    v_0 = Rnode(0)
    v_1 = elem.sinh(v_0)
  #  try:
  #      assert v_1.val == 0, "sinh function gave wrong value"
  #      assert v_1.deriv == 1, "sinh function gave wrong derivative"
  #      assert elem.sinh(0) == 0, "sinh function not working for non Fnodes"
  #  except AssertionError as e:
  #      print(e)
  #      raise AssertionError

if __name__ == '__main__':
    test_sin()
    test_arcsin()
    test_sinh()