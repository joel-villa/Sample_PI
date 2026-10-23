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

    @abstractmethod
    def get(
        self,
        kwargs: dict[str, Any],
    ) -> tuple[scipy.sparse.sparray | np.ndarray, dict] | None:
        """Get a matrix based on its key word arguments

        Args:
            kwargs (dict): The key word arguments for generating the matrix

        Returns:
             mat (Union[scipy.spare.sparray, np.ndarray]): The matrix gotten
        """
