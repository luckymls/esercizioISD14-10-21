from abc import ABC
from copy import deepcopy

import numpy as np


class ConvKernel(ABC):

    def __init__(self, kernel_size=3):
        
        if kernel_size % 2 == 0:
            raise TypeError("The kernel size must be an odd number")
        self._kernel_size = kernel_size
        self._mask = None

    def kernel_mask(self):
        raise NotImplementedError("This method isn't implemented")

    def kernel(self, x, mask=None):
        
        if mask is None:
            mask = self._mask

        xp = deepcopy(x)
        kernel_size_half = int(self.kernel_size / 2)

        xp[int(self.kernel_size / 2) : xp.size - int(self.kernel_size / 2)] = \
            np.rint([np.dot(xp[(i - int(self.kernel_size / 2)) : (i + int(self.kernel_size / 2) + 1)], mask)
                     for i in range(int(self.kernel_size / 2), xp.size - int(self.kernel_size / 2))]).astype(int)

        return xp

    @property
    def mask(self):
        return self._mask

    @property
    def kernel_size(self):  
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, new_size):
        self._kernel_size = new_size
        self._mask = self.kernel_mask()
