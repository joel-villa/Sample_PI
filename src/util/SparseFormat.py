"""The various types of sparse matrices

Resources:
    - Enums:
        https://docs.python.org/3/library/enum.html
"""
from enum import Enum

# class syntax
class SparseFormat(Enum):
    COO = 1
    CSR = 2
    CSC = 3
