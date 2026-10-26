"""A superclass which defines the API needed to be a 'MatGetter'

Resources:
    - Union typing in python:
        https://docs.python.org/3/library/typing.html
    - Abstract methods in python:
        https://docs.python.org/3/library/abc.html
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import scipy


class MatGetter(ABC):
    """For loading matrices"""

    preprocess_data: bool

    def __init__(
        self,
        preprocess_data: bool,
    ) -> None:
        """Initialize a Matrix Getter

        Args:
            preprocesss_data (bool): True -> apply scaling, False -> don't

        """
        super().__init__()
        self.preprocess_data = preprocess_data

    @abstractmethod
    def get(
        self,
        kwargs: dict[str, Any] | None = None,
    ) -> scipy.sparse.sparray | np.ndarray | None:
        """Get a matrix based on its key word arguments

        Args:
            kwargs (dict): The key word arguments for generating the matrix

        Returns:
            mat (scipy.sparse.sparray | np.ndarray | None): The matrix
            gotten
        """

    @abstractmethod
    def preprocess(
        self,
        matrix: scipy.sparse.sparray | np.ndarray,
    ) -> tuple[scipy.sparse.sparray | np.ndarray, float]:
        """Get A scaled by a factor of 1/max_i(||a_i||)

        Args:
            mat_name (str): the name of the suite-spasrse matrix

        Returns:
            matrix (scipy.sparse): the matrix in sparse format
            scale_factor (float): the scaling factor
        """
