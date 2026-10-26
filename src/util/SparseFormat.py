"""The various types of sparse matrices

Resources:
    - Enums:
        https://docs.python.org/3/library/enum.html
    - Frozen Sets:
        https://www.w3schools.com/python/ref_func_frozenset.asp
"""

from enum import Enum

import scipy

TYPES: frozenset = frozenset(
    {
        scipy.sparse.coo_array,
        scipy.sparse.csr_array,
        scipy.sparse.csc_array,
    }
)


# class syntax
class SparseFormat(Enum):
    COO = 1
    CSR = 2
    CSC = 3
