import numpy as np

class Fnode:
    def __init__(self, val, deriv=1, var_name="none"):
        """Constructor for Node for Forward Automatic differentiaton.

        Parameters
        ==========
        val : list/int/float
            The value of the node
        deriv : list/int/float
            The first derivative of the node
        var_name : string
            The name of the variable
        """

        if isinstance(val, (int, float)):
            self._val = np.array([val])
        elif isinstance(val, list):
            self._val = np.array(val)
        elif isinstance(val, np.ndarray):
            self._val = val
        else:
            raise TypeError("val must be either a number, list, or numpy array")
        
        if isinstance(deriv, (int, float)):
            self._deriv = {var_name: np.array([deriv] * len(self._val))}
        elif type(deriv) == list:
            self._deriv = {var_name: np.array(deriv)}
        elif type(deriv) == dict:
            self._deriv = deriv
        else:
            raise TypeError("deriv must be either a number, list, or dictionary")

        self._var_name = var_name


    @property
    def deriv(self):
        return self._deriv


    @property
    def val(self):
        return self._val


    @property
    def var_name(self):
        return self._var_name


    def get_vars(self):
        return set(self._deriv.keys())


    def __neg__(self):
        """
        Overloads negation

        Parameters:
        x: Value or Fnode to negate

        Returns:
        A new Fnode object that where the value and derivative are both negated
        """
        total_deriv = {}
        for var in self.get_vars():
            total_deriv[var] = -self.deriv.get(var, 0)
        return Fnode(-self._val, total_deriv)


    def __add__(self, other):
        """
        Overloads addition

        Parameters:
        other: Value or Fnode to add

        Returns:
        For Fnodes, a new Fnode object where the other is added to self for the value and derivative
        For values, a new Fnode object where the other is added to self for the value and derivative
        """
        if isinstance(other, Fnode):
            total_deriv = {}
            total_vars = self.get_vars().union(other.get_vars())
            for var in total_vars:
                total_deriv[var] = self._deriv.get(var, 0) + other.deriv.get(var, 0)
            return Fnode(self._val + other.val, total_deriv, self._var_name)
        elif isinstance(other, (int, float)):
            return Fnode(self._val + other, self._deriv.copy(), self._var_name)
        else:
            raise TypeError("Invalid input type: must add either Fnode, int, or float")
    

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
        if isinstance(other, Fnode):
            total_deriv = {}
            total_vars = self.get_vars().union(other.get_vars())
            for var in total_vars:
                total_deriv[var] = self._deriv.get(var, 0) - other.deriv.get(var, 0)
            return Fnode(self._val - other._val, total_deriv, self._var_name)
        elif isinstance(other, (int, float)):
            return Fnode(self._val - other, self._deriv.copy(), self._var_name)
        else:
            raise TypeError("Invalid input type: must subtract either Fnode, int, or float")


    def __rsub__(self, other):
        return -self + other
      

    def __mul__(self, other):
        """
        Overloads multiplication

        Parameters:
        other: Value or Fnode to multiply against

        Returns:
        For Fnodes, a new Fnode object where the self and other Fnodes are multiplied according to the product rule for the value and derivative
        For values, a new Fnode object where the self and other values are multiplied according to the product rule for the value and derivative
        """
        if isinstance(other, Fnode):
            total_deriv = {}
            total_vars = self.get_vars().union(other.get_vars())
            for var in total_vars:
                total_deriv[var] = self._val * other.deriv.get(var, 0) + other.val * self._deriv.get(var, 0)
            return Fnode(self._val * other.val, total_deriv, self._var_name)
        elif isinstance(other, (int, float)):
            total_deriv = {}
            for var in self.get_vars():
                total_deriv[var] = self._deriv[var] * other
            return Fnode(self._val * other, total_deriv, self._var_name)
        else:
            raise TypeError("Invalid input type: must multiply either Fnode, int, or float")


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
        return self * (other ** (-1))

    
    def __rtruediv__(self, other):
        return other * (self ** (-1))


    def __pow__(self, other):
        """
        Overloads exponentiation (powers)

        Parameters:
        other: Value or Fnode that represents the power to raise the current Fnode or value by

        Returns:
        For Fnodes, a new Fnode object where the self and other Fnodes are multiplied according to the power rule for the value and derivative
        For values, a new Fnode object where the self and other values are multiplied according to the power rule for the value and derivative
        """
        if isinstance(other, Fnode):
            if len(other.val) != len(self.val):
                raise ValueError("Two Fnodes must be of the same length")
            elif len(other.val) == 1:
                value_other = other.val * np.ones(self._val.shape)
            else:
                value_other = other.val[:]

            total_deriv = {}
            total_vars = self.get_vars().union(other.get_vars())
            total_value = np.array([float(v) ** v_other for v, v_other in zip(self._val, other.val)])

            for var in total_vars:
                current_value = np.array([v ** (v_other - 1) for v, v_other in zip(self._val, other.val)])
                total_deriv[var] = (value_other * self._deriv.get(var, 0) + self._val * np.log(self._val) * other._deriv.get(var, 0)) * current_value
            
            return Fnode(total_value, total_deriv, self._var_name)
        elif isinstance(other, (int, float)):
            total_value = np.array([float(v) ** other for v in self._val])
            total_deriv = {}
            
            for var in self.get_vars():
                current_value = np.array([float(v) ** (other - 1) for v in self._val])
                total_deriv[var] = other * current_value * self._deriv[var]
            
            return Fnode(total_value, total_deriv, self._var_name)
        else:
            raise TypeError("Invalid input type: must raise to the power of an Fnode, int, or float")


    def __rpow__(self, other):
        if isinstance(other, Fnode):
            return other ** self
        elif isinstance(other, (int, float)):
            total_value = np.array([other ** v for v in self._val])
            total_deriv = {}


            for var in self.get_vars():
                current_value = np.array([other ** (v - 1) for v in self._val])
                total_deriv[var] = np.log(other) * current_value * self._deriv[var]
            return Fnode(total_value, total_deriv, self._var_name)
        else:
            raise TypeError("Invalid input type: must raise to the power of an Fnode, int, or float")
