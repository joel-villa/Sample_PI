"""Doing the expected number of nonzero calculations after the fact"""

import numpy as np
import scipy
from scipy.stats import binom
from tqdm import tqdm

from src.acc_vs_work.AccWorkResult import AccWorkResult
from src.acc_vs_work.SSGetter import SSGetter
from src.acc_vs_work.util.Approximator import Approximator
from src.util.SparseFormat import SparseFormat


class NnzCalculator:
    """For expected number of nonzero calculations"""

    _ss_getter: SSGetter

    def __init__(
        self,
    ) -> None:
        """Construct her..."""
        self._ss_getter = SSGetter(SparseFormat.CSR)

    def __prob_at_least_one_success(self, n_trials: int, prob_success: float) -> float:
        """The probability of getting at least one success in n binomials trials
        with the given probability of success

        Args:
            n_trials (int): The number of binomial trails
            prob_success (float): The probability of success per trial

        Returns:
             prob_success (float): The probability of at least one success
             occurring

        Resources:
            - Binomial distribution stuff:
                https://medium.com/@ja_x/binomial-distribution-probability-mass-function-cumulative-distribution-function-calculation-5ae74fd3a340
        """
        probs = binom.pmf(
            k=np.arange(n_trials),
            n=n_trials,  # Number of trials
            p=prob_success,  # Probability of success
        )

        prob_failure = probs[0]

        return 1 - prob_failure

    def __calculate_old(
        self,
        matrix: scipy.sparse.sparray,
        n_trials: int,
    ) -> float:
        """Calculate the expected number of nonzeroes of the A^TA using the
        sampling scheme on A

        Args:
            matrix (scipy.sparse.sparray): A sparse matrix (CSR format)
            n_trials (int): Number of binomial trials to generate the matrix
            approximation

        Returns:
             expected_nnz (float): The expected number of nonzeroes of the
             approximation
        """

        # Probability matrix is (n+1)xnxn
        probability_matrix = np.zeros(
            (matrix.shape[0] + 1, matrix.shape[0], matrix.shape[0]), dtype=float
        )

        magnitudiest = Approximator(matrix=matrix, rng=None)
        row_magnitudes = magnitudiest.calc_row_norms()
        for i in tqdm(range(1, probability_matrix.shape[0])):
            start_idx = matrix.indptr[i - 1]
            end_idx = matrix.indptr[i]
            curr_col_idxs = matrix.indices[start_idx:end_idx]

            for j in range(probability_matrix.shape[1]):
                # Is i'th row's j'th entry nonzero?
                j_indicator = 0
                if j_indicator in curr_col_idxs:
                    j_indicator = 1

                for k in range(probability_matrix.shape[2]):
                    # Is i'th row's k'th entry nonzero?
                    k_indicator = 0
                    if k_indicator in curr_col_idxs:
                        k_indicator = 1

                    old_prob_nz = probability_matrix[i - 1, j, k]
                    curr_prob_nz = (
                        j_indicator
                        * k_indicator
                        * self.__prob_at_least_one_success(
                            n_trials=n_trials,
                            prob_success=row_magnitudes[i],
                        )
                    )

                    probability_matrix[i, j, k] = (
                        old_prob_nz + curr_prob_nz - (old_prob_nz * curr_prob_nz)
                    )

        # Expected number of nonzeroes is the sum of the probabilities (linearity of expectation!)
        return np.sum(probability_matrix[n_trials])


    def __calculate(
        self,
        matrix: scipy.sparse.sparray,
        n_trials: int,
    ) -> float:
        """Calculate the expected number of nonzeroes of the A^TA using the
        sampling scheme on A

        Args:
            matrix (scipy.sparse.sparray): A sparse matrix (CSR format)
            n_trials (int): Number of binomial trials to generate the matrix
            approximation

        Returns:
             expected_nnz (float): The expected number of nonzeroes of the
             approximation
        Resources:
            - Sparse array API:
                https://docs.scipy.org/doc/scipy-1.18.0/reference/generated/scipy.sparse.csr_array.html#scipy.sparse.csr_array
            - Sparse matrix element-wise multiplication:
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.multiply.html
            - Numpy all
                https://numpy.org/doc/2.2/reference/generated/numpy.all.html
        """

        # Probability matrix is same shape as A^TA
        old_probs = matrix.T @ matrix
        max_nnzs = old_probs.nnz
        old_probs.data[:] = 0

        magnitudiest = Approximator(matrix=matrix, rng=None)
        row_magnitudes = magnitudiest.calc_row_norms()
        for i in range(matrix.shape[0]):
            curr_prob_nz = self.__prob_at_least_one_success(n_trials=n_trials, prob_success=row_magnitudes[i])

            # Only the outer product of the i'th row gets their probabilities updated
            ith_probs = matrix[i].T @ matrix[i]
            ith_probs.eliminate_zeros()
            ith_probs.data[:] = curr_prob_nz

            assert ith_probs.nnz <= max_nnzs, f"{i}'th outer product has {ith_probs.nnz} nonzeros, max is {max_nnzs}"

            curr_probs = old_probs + ith_probs - old_probs.multiply(ith_probs)
            old_probs = curr_probs

        old_probs.eliminate_zeros()

        assert np.all(old_probs.data <= 1), f"huh? {np.max(old_probs.data)}"

        # Expected number of nonzeroes is the sum of the probabilities (linearity of expectation!)
        return np.sum(old_probs)

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
        print(mat_name)
        matrix = self._ss_getter.get_preprocessed(mat_name=mat_name)
        assert matrix is not None, "Didn't get a matrix"

        expected_nnzs = np.zeros((len(result.trial_list),), dtype=float)

        for i, n_binom_trials in enumerate(result.trial_list):
            expected_nnz = self.__calculate(
                matrix=matrix,
                n_trials=n_binom_trials,
            )
            expected_nnzs[i] = expected_nnz

        result.expected_nnzs = expected_nnzs
        result.save_results()


def main():
    """Nothing but a thang"""
    result_ids = range(2, 25)

    nnz_calculator = NnzCalculator()

    for id in result_ids:
        results = AccWorkResult(reduct_type=ReductionType.SND_MOMENT_BINOMIAL, file_id=id)
        nnz_calculator.update_expected_nnzs(results)
        print(f"({id}) results.expected_nnz: {results.expected_nnzs}")

def main_1():
    """For testing"""
    file_id = 2

    result = AccWorkResult(reduct_type=ReductionType.SND_MOMENT_BINOMIAL, file_id=file_id)
    nnz_calculator = NnzCalculator()
    nnz_calculator.update_expected_nnzs(result)

    result = AccWorkResult(reduct_type=ReductionType.SND_MOMENT_BINOMIAL, file_id=file_id)
    print(f"expected nnzs: {result.expected_nnzs}")


if __name__ == "__main__":
    """For testing purporses"""
    from src.acc_vs_work.util.ReductionType import ReductionType

    main()
    #main_1()
