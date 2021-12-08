import numpy as np
from numpy.lib.arraysetops import isin


class Rnode:
    def __init__(self, val):
        """Constructor for Node for Reverse Automatic Differentiaton.

        Parameters
        val : int/float
            The value of the node
        """

        self._val = val
        self.grad_value = None
        self._children = []


    def grad(self):
        """return the gradient of the function via reverse mode automatic differentation

        Parameters:
        self:
            Rnode object

        Returns:
            The float value of the functions gradient
        """
        if self.grad_value is None:
            self.grad_value = sum(weight * rnode.grad() for weight, rnode in self._children)
        return self.grad_value

    @property
    def val(self):
        return self._val


    def __pow__(self, other):
        """
        Overloads exponentiation (powers)

        Parameters:
        other: Value or Rnode that represents the power to raise the current Rnode or value by

        Returns:
        For Fnodes, a new Rnode object where the self and other Rnodes are multiplied according to the power rule
        For values, a new Rnode object where the self and other values are multiplied according to the power rule
        """
        try:
            z = Rnode(self._val ** other._val)
            self._children.append(( other._val * self._val ** (other._val - 1), z))
            other._children.append((self._val ** other._val * np.log(self._val), z))
        except AttributeError:
            z = Rnode(self._val ** other)
            self._children.append((other * self._val ** (other - 1), z))
        return z

    def __rpow__(self, other):
        z = Rnode(other ** self._val)
        self._children.append(( np.log(other) * other ** (self._val), z))
        return z

    def __neg__(self):
        """
        Overloads negation

        Parameters:
        x: Value or Rnode to negate

        Returns:
        A new Rnode object where the value is negated
        """
        z = Rnode(-self._val)
        self._children.append((-1, z))
        return z


    def transform(self, a, b):
        if isinstance(b, float) or isinstance(b, int):
            b = Rnode([b] * len(a._val))

        return b


    def __add__(self, other):
        other = self.transform(self, other)
        z = Rnode(self._val + other._val)
        self._children.append((np.ones(self._val.shape), z))
        other._children.append((np.ones(self._val.shape), z))
        return z


    def __radd__(self, other):
        return self + other


    def __sub__(self, other):
        other = self.transform(self, other)
        z = Rnode(self._val - other._val)
        self._children.append((np.ones(self._val.shape), z))
        other._children.append((np.ones(self._val.shape), z))
        return z


    def __rsub__(self, other):
        return self - other

        
    def __mul__(self, other):
        """
        Overloads multiplication

        Parameters:
        other: Value or Rnode that represents the amount to multiply the current Rnode or value by

        Returns:
        For Fnodes, a new Rnode object where the self and other Rnodes are multiplied according to the product rule
        For values, a new Rnode object where the self and other values are multiplied according to the product rule
        """
        self.transform(self, other)
        z = Rnode(self._val * other._val)
        self._children.append((other._val, z))
        other._children.append((self._val, z))

    
    def __rmul__(self, other):
        return self * other


    def __truediv__(self, other):
        """
        Overloads division

        Parameters:
        other: Value or Rnode that represents the amount to divide the current Rnode or value by

        Returns:
        For Fnodes, a new Rnode object where the self and other Rnodes are divided according to the quotient rule
        For values, a new Rnode object where the self and other values are divided according to the quotient rule
        """
        return self * (other ** (-1))


    def __rtruediv__(self, other):
        return other * (self ** (-1))
