import numpy as np

class Vector_Fn:
    def __init__(self, function):
        """Constructor for the Vector_Fn class that enables vector valued forward mode automatic differentiation

        Parameters
        ==========
        function_list : list
            list of functions for each variable
        """
        self._function_list = function

    def val_by_var(self):
        '''
        Returns list of values
        '''
        return [function.val for function in self._function_list]

    def deriv_by_var(self):
        '''
        Returns list of derivatives
        '''
        return [function.deriv for function in self._function_list]

    def get_vals(self):
        '''
        Returns an np array of values
        '''
        return np.array(self.val_by_var()).T

    def get_deriv(self):
        '''
        Returns the jacobian matrix
        '''
        var_names = set()
        for func in self._function_list:
            var_names = var_names.union(func.get_vars())
        var_names = list(var_names)
        var_names.sort()

        jacobian_matrix = []
        func_length = len(self._function_list[0].val)
        num_func = len(self._function_list)
        num_vars = len(var_names)

        for _ in range(func_length):
            jacobian_matrix.append(np.zeros((num_func, num_vars)))

        for i in range(func_length):
            for r in range(num_func):
                func = self._function_list[r]
                for c in range(num_vars):
                    var = var_names[c]
                    if var in func.get_vars():
                        jacobian_matrix[i][r, c] = func.deriv[var][i]

        return var_names, jacobian_matrix