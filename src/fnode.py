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

    def __pow__(self, other):
        try:
            return Fnode(
                self._val ** other._val,
                other._val * self._val ** (other._val - 1) * self._deriv,
            )
        except AttributeError:
            return Fnode(
                self._val ** other, other * self._val ** (other - 1) * self._deriv
            )

    def __neg__(self):
        return Fnode(-self._val, -self._deriv)

    def __division__(self, other):
        try:
            return Fnode(
                self._val / other._val,
                (self._deriv * other._val - self._val * other._deriv)
                / (other._val ** 2),
            )
        except AttributeError:
            return Fnode(self._val / other, self._deriv)
