"""Doing the expected number of nonzero calculations after the fact"""

import scipy

from src.acc_vs_work.AccWorkResult import AccWorkResult
from src.acc_vs_work.SSGetter import SSGetter
from src.util.SparseFormat import SparseFormat


class NnzCalculator:
    """For expected number of nonzero calculations"""

    _ss_getter: SSGetter

    def __init__(
        self,
    ) -> None:
        """Construct her...
	    """
        self._ss_getter = SSGetter(SparseFormat.CSR)


    def update_expected_nnzs(
        self,
        result: AccWorkResult,
    ) -> None:
        """Description

        Args:
            result (AccWorkResult): The result which will be updated to include
            the proper number of nonzeros
        """
        mat_name = result.mat_name
        matrix = self._ss_getter.get_preprocessed(mat_name=mat_name)
        assert matrix is not None, "Didn't get a matrix"

        #TODO finish this
