class Fnode:
   

    def __init__(self, val, deriv):
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
            return Fnode(self._val**other.val, other_val*self._val**(other._val-1)*self._deriv
        except AttributeError:
            return Fnode(self._val**other, other*self._val**(other-1)*self._deriv