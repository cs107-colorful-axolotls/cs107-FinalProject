class Fnode:
    def __init__(self, val, deriv=1):
        """Constructor for Node for Forward Automatic differentiaton.

        Parameters
        ==========
        val : int/float
            The value of the node
        deriv : int/float
            The first derivative of the node
        """

        self._val = val
        self._deriv = deriv

    @property
    def deriv(self):
        return self._deriv

    @property
    def val(self):
        return self._val

    def __neg__(self):
        """
        Overloads negation

        Parameters:
        x: Value or Fnode to negate

        Returns:
        A new Fnode object that where the value and derivative are both negated
        """
        return Fnode(-self._val, -self._deriv)

    def __add__(self, other):
        """
        Overloads addition

        Parameters:
        other: Value or Fnode to add

        Returns:
        For Fnodes, a new Fnode object where the other is added to self for the value and derivative
        For values, a new Fnode object where the other is added to self for the value and derivative
        """
        try:
            return Fnode(self._val + other.val, self._deriv + other.deriv)
        except AttributeError:
            return Fnode(self._val + other, self._deriv)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """
        Overloads subtraction

        Parameters:
        other: Value or Fnode to subtract

        Returns:
        For Fnodes, a new Fnode object where the other is subtracted from self for the value and derivative
        For values, a new Fnode object where the other is subtracted from self for the value and derivative
        """
        try:
            return Fnode(self._val - other.val, self._deriv - other.deriv)
        except AttributeError:
            return Fnode(self._val - other, self._deriv)

    def __rsub__(self, other):
        try:
            return Fnode(self._val - other.val, other.deriv - self._deriv)
        except AttributeError:
            return Fnode(other - self._val, -self._deriv)
      
    def __mul__(self, other):
        """
        Overloads multiplication

        Parameters:
        other: Value or Fnode to multiply against

        Returns:
        For Fnodes, a new Fnode object where the self and other Fnodes are multiplied according to the product rule for the value and derivative
        For values, a new Fnode object where the self and other values are multiplied according to the product rule for the value and derivative
        """
        try:
            return Fnode(self._val * other.val, self._val * other.deriv + other.val * self._deriv)
        except AttributeError:
            other = Fnode(other, 0)
            return Fnode(self._val * other.val, self._val * other.deriv + other.val * self._deriv)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Overloads division

        Parameters:
        other: Value or Fnode to divide against

        Returns:
        For Fnodes, a new Fnode object where the self and other Fnodes are divided according to the quotient rule for the value and derivative
        For values, a new Fnode object where the self and other values are divided according to the quotient rule for the value and derivative
        """
        try:
            return Fnode(self._val / other.val, (self.deriv * other.val - self._val * other.deriv) / (other.val ** 2))
        except AttributeError:
            return Fnode(self._val / other, self._deriv / other)

    def __pow__(self, other):
        """
        Overloads exponentiation (powers)

        Parameters:
        other: Value or Fnode that represents the power to raise the current Fnode or value by

        Returns:
        For Fnodes, a new Fnode object where the self and other Fnodes are multiplied according to the power rule for the value and derivative
        For values, a new Fnode object where the self and other values are multiplied according to the power rule for the value and derivative
        """
        try:
            return Fnode(self._val ** other.val, other.val * self._val ** (other.val - 1) * self._deriv)
        except AttributeError:
            return Fnode(self._val ** other, other * self._val ** (other - 1) * self._deriv)
