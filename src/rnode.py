import numpy as np


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
