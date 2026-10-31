"""Generates matrix approximations"""

import numpy as np
import scipy
from scipy.sparse import diags

from src.acc_vs_work.util.ReductionType import ReductionType

from src.util.constants import THIRTY_TWO_BIT_PRECISION


class Approximator:
    """For generating approximations of a matrix"""

    matrix: scipy.sparse.sparray
    rng: np.random.Generator

    def __init__(self, matrix: scipy.sparse.sparray, rng: np.random.Generator) -> None:
        """Initialize the approximator

        Args:
            matrix (scipy.sparse.sparray): Matrix to approximate
        """
        self.matrix = matrix
        self.rng = rng

    def __get_next_diagonal_sampler(
        self,
        row_norms: np.ndarray,
    ) -> np.ndarray:
        """Get the vector that represents a diagonal matrix which randomly
        samples those columns of A

        Args:
            row_norms (np.ndarray): The norms of the rows of the matrix to sample
            rng (np.random.Generator): A random number generator

        Returns:
            np.ndarray: The vector which represents a random diagonal sampling
            matrix
        """  # TODO
        binomial_vals = self.rng.binomial(n=1, p=row_norms)
        scaled_vals = binomial_vals / np.sqrt(row_norms)

        assert np.all(
            (scaled_vals >= 1) | (scaled_vals == 0)
        )  # TODO: can be deleted after testing once
        return scaled_vals

    def __get_diag_mat(
        self,
        row_norms: np.ndarray,
    ) -> scipy.sparse.sparray:
        """Get an independent copy of the random diagonal matrix which serves the
        purpose of matrix sampling

        Args:
            row_norms (np.ndarray): The norms of the rows of the matrix to sample
            rng (np.random.Generator): A random number generator

        Returns:
            scipy.sparse.sparray: The random diagonal sampling matrix
        """
        diag_array = self.__get_next_diagonal_sampler(
            row_norms=row_norms,
        )
        return diags(diag_array)

    def __round_row_norms(
        self,
        row_norms: np.ndarray,
    ) -> np.ndarray:
        """Check that the row_norms are not errorneous, if they have slight
        rounding error, round them to one or zero where appropriate

        Args:
            row_norms (np.ndarray): Array of the row magnitudes of some scaled
            matrix, s.t. its maximum row magnitude is one

        Returns:
            np.ndarray: The row_norms which should be used
        """
        # Ensure row norms are valid
        assert np.all(row_norms <= 1 + THIRTY_TWO_BIT_PRECISION), (
            f"max row norm: {np.max(row_norms)}"
        )
        assert np.all(row_norms >= 0 - THIRTY_TWO_BIT_PRECISION), (
            f"min row norm: {np.min(row_norms)}"
        )

        idx_too_large = [idx for idx, val in enumerate(row_norms) if val > 1]
        idx_too_small = [idx for idx, val in enumerate(row_norms) if val < 0]

        row_norms[idx_too_large] = 1
        row_norms[idx_too_small] = 0

        return row_norms

    def __get_row_norms(
        self,
    ) -> np.ndarray:
        """Get the row_norms of the given matrix

        Args:
            A (scipy.sparse.sparray): The matrix

        Returns:
            np.ndarray: The row magnitudes of every row in Af
        """

        row_mags = np.asarray(scipy.sparse.linalg.norm(self.matrix, ord=2, axis=1))
        row_mags = self.__round_row_norms(row_norms=row_mags)
        return row_mags

    def __get_next_A_tilde(
        self,
    ) -> scipy.sparse.sparray:
        """Generate the next reduced version of A

        Args:
            A (scipy.sparse.sparray): The original matrix
            rng (np.random.Generator): the random number generator

        Returns:
            scipy.sparse.sparray: The row-reduced version of A
        """
        row_norms = self.__get_row_norms()
        diag_mat = self.__get_diag_mat(
            row_norms=row_norms,
        )

        return diag_mat @ self.matrix

    def __init_A_tilde_sq(
        self,
    ) -> tuple[scipy.sparse.csr_array, np.ndarray]:
        """Get the intial info necessary for generating ~A^TA

        Args:
            A (scipy.sparse.sparray): The matrix to approximate

        Returns:
            tuple[scipy.sparse.csr_array, np.ndarray]:
                scipy.sparse.csr_array: a zeros array
                np.ndarray: the row_norms of A
        """
        zeros = scipy.sparse.csr_array(self.matrix.shape)  # zeros sparse array
        row_norms = self.__get_row_norms()

        return zeros, row_norms

    def __naive_A_tilde_sq(
        self,
        num_trials: int,
    ) -> scipy.sparse.sparray:
        """Generate ~A^TA in a naive (non-paralellizable) way
        NOTE: ~A^TA = 1/N sum_{i=1}^N A^T D_i A
        where D_i is the i'th diagonal sampler

        Args:
            A (scipy.sparse.sparray): Some matrix to approximate A^TA of
            num_trials (int): The number of approximations of A^TA to generate
            rng (np.random.Generator): For repeatable randomization

        Returns:
            scipy.sparse.sparray: The approximation of A^TA
        """
        approximation, row_norms = self.__init_A_tilde_sq()

        for i in range(num_trials):
            diag_i = self.__get_diag_mat(
                row_norms=row_norms,
            )
            approximation += (
                (1 / num_trials)
                * self.matrix.transpose()
                @ diag_i
                @ diag_i
                @ self.matrix
            )

        return approximation

    def __binomial_A_tilde_sq(
        self,
        num_trials: int,
    ) -> scipy.sparse.sparray:
        """Generate ~A^TA in a (hopefully) more parallelizable way

        Args:
            A (scipy.sparse.sparray): Some matrix to approximate A^TA of
            num_trials (int): The number of approximations of A^TA to generate
            rng (np.random.Generator): For repeatable randomization

        Returns:
            scipy.sparse.sparray: The approximation of A^TA
        """
        approximation, row_norms = self.__init_A_tilde_sq()

        for i in range(self.matrix.shape[0]):
            """Notice that everything that happens in this for loop is independent,
            i.e. parallizable"""
            row_i = self.matrix[i, :]  # i'th row of A
            norm_i = row_norms[i]  # magnitude of i'th row
            binom = self.rng.binomial(
                n=num_trials,
                p=norm_i,  # w.p. ||a_j||, X_ij is one
            )

            scalar = binom / (num_trials * norm_i)
            approximation += scalar * (row_i.transpose() @ row_i)
        return approximation

    def generate(
        self,
        reduct_type: ReductionType,
        n_trials: int,
    ) -> scipy.sparse.sparray:
        """The vibes are immaculate, this method is so deep

        Args:
            reduct_type (ReductionType): The strategy for generating the
            approximation
            n_trials (int): The number of trials to generate the approximation

        Returns:
             approx (scipy.sparse.sparray): The matrix approximation generated
        """
        match reduct_type:
            case ReductionType.SND_MOMENT_NAIVE:
                return self.__naive_A_tilde_sq(num_trials=n_trials)
            case ReductionType.SND_MOMENT_BINOMIAL:
                return self.__binomial_A_tilde_sq(num_trials=n_trials)
            case _:
                raise ValueError(f"Unexpected reduction type: {reduct_type.name}")


if __name__ == "__main__":
    """Yeahhh
    """
    from scipy.sparse.linalg import norm

    from src.acc_vs_work.SSGetter import SSGetter
    from src.util.SparseFormat import SparseFormat

    mats = [
        "1138_bus",
        "494_bus",
        "Harvard500",
        "bcspwr06",
        "bcsstk07",
        "bcsstk08",
        "bcsstk19",
        "bcsstk34",
        "bcsstm07",
        "blckhole",
        "cage7",
        "can_229",
        "dwt_193",
        "eris1176",
        "ex2",
        "fs_541_1",
        "gre_1107",
        "gre_343",
        "hor_131",
        "lshp1561",
        "msc00726",
        "nasa1824",
        "nos3",
        "tomography",
    ]

    mats = sorted(mats)  # Alphabetical order
    print("Starting approximation tests...")
    n_trials = 8
    num_avg = 16
    print(f"N = {n_trials}")
    print(f"num_avg = {num_avg}")
    ss_getter = SSGetter(
        format=SparseFormat.CSR,
        nzbounds=(1000, 100000),
    )
    for mat in mats:
        print(f"\n{mat}")
        A = ss_getter.get_preprocessed(mat)
        if A is None:
            print(f"No matrix gotten, skipping...")
            continue
        rng = np.random.default_rng(seed=5334)
        rand_vect = rng.normal(loc=0.0, scale=0.0625, size=A.shape[0])
        approximator = Approximator(A, rng)
        # Operator norm clarity check
        print(f"ACTUAL operator norm: {norm(A.transpose() @ A)}")
        for reduct in {
            ReductionType.SND_MOMENT_BINOMIAL,
            ReductionType.SND_MOMENT_NAIVE,
        }:
            print(f"reduct: {reduct.name}")
            avg_op_norm = 0
            for i in range(num_avg):
                snd_mom_tilde = approximator.generate(
                    reduct_type=reduct,
                    n_trials=n_trials,
                )  # TODO
                avg_op_norm += norm(snd_mom_tilde)
            avg_op_norm = avg_op_norm / num_avg
            print(f"{avg_op_norm}")
