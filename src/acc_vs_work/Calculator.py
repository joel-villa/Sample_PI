"""For some repeated calculations (ex: work)"""

import scipy
import numpy as np
import warnings

class Calculator:
    """For doing some matrix involved calculations"""
    matrix: scipy.sparse.sparray
    def __init__(
        self,
        matrix: scipy.sparse.sparray,
    ) -> None:
        """Initialize the calculator for the given matrix

        Args:
            matrix (scipy.sparse.sparray): Sparse matrix to do calculations with
        """
        self.matrix = matrix

    def power_work(
        self,
            num_iter: int,
    ) -> np.ndarray:
        """The amount of cummulative work done at every iteration of Power on the
        given matrix

        Args:
            matrix (scipy.sparse.sparray): The matrix that power-iteration was run on
            num_iter (int): number of iterations run

        Returns:
            np.ndarray: A linearly increasing one-dimensional array of length
            num_iter
        """
        nnzs = matrix.nnz
        iterations = np.arange(0, num_iter)
        total_work = iterations * nnzs # Doing SpMv multiplication per iteration
        return total_work

    def prep_work_baseline(
        self,
            is_distributed:bool,
    ) -> int:
        """The number of scalar mults to compute A^TA

        NOTE: A^TA can be rewritten as a sum of the outerproducts of the rows

        Work breakdown:
            The i'th row contributes w_i^2, where w_i is the nnz's in the i'th row

        Args:
            A (scipy.sparse.sparray): The matrix A
            is_distributed (bool): True -> max thread work, False -> total work

        Returns:
            int: Number of scalar multsf
        """
        if type(self.matrix) != scipy.sparse.csr_array and type(self.matrix) != scipy.sparse._csr.csr_matrix:
            warnings.warn(f"A must be CSR, given {type(self.matrix)}")

        row_nnzs = np.diff(self.matrix.indptr)
        row_work = row_nnzs * row_nnzs

        assert row_nnzs.shape[0] == self.matrix.shape[0] #TODO: delete this after sure it is workign

        if is_distributed: #TODO: have return both, get rid of is_distributed
            return np.max(row_work)

        return np.sum(row_work)

    def prep_work_naive(
        self,
        num_tirals:int,
        is_distributed:bool,
    ) -> int:
        """Calculate ammount of work (scalar mults) involved in computing this
        average in a naive way

        NOTE: actually computing this work involves expectation, and sparsity
        rate, this is a simple overestimate

        Work breakdown: Doing A.transpose() @ diag_i @ diag_i @ A at every iteration
            1. diag_i @ diag_i => at most, n scalar mults
            2. diag_i^2 @ A => at most, additional A.nnz scalar mults
            3. A^T @ (diag^2 @ A) => at most, addiotnal A.nnz * n scalar mults

        Args:
            A (scipy.sparse.sparray): The matrix to approximate
            num_tirals (int): The ammount of approximations to average over
            is_distributed (bool): True -> taking maxium, False -> summing up work

        Returns:
            int: amount of scalar mults
        """
        mults_per_iter = self.matrix.nnz + self.matrix.shape[0] + self.matrix.nnz * self.matrix.shape[0]
        if is_distributed:
            return mults_per_iter
        return num_tirals * mults_per_iter

    def prep_work_binomial(
        self,
        is_distributed:bool,
    ) -> int:
        """Calculate ammount of work required to do the parallelizable calculation
        of an averaged approximation of A^TA

        NOTE:
            (1) the total work is independent of the number of trials, just
            dependent upon the number of rows
            (2) same amount of work to compute approx A^TA as A^TA, assuming the
            cost of calling binom() is insignificant in comparison to the outer
            product computations

        Work breakdown: doing row_i.transpose() @ row_i at every iteration
            The i'th row contributes w_i^2, where w_i is the nnz's in the i'th row

        Args:
            A (scipy.sparse.csr_array): The matrix in question
            is_distributed (bool): True -> taking maxium, False -> summing up work
            per row

        Returns:
            int: Ammount of work per thread
        """
        return self.prep_work_baseline(
            is_distributed=is_distributed,
        )

if __name__ == '__main__':
    """Yeahhh
    """
    from .SSGetter import SSGetter
    from ..util.SparseFormat import SparseFormat

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

    mats = sorted(mats) #Alphabetical order
    ss_getter = SSGetter(
        format=SparseFormat.CSR,
        nzbounds=(1000, 100000),
    )
    print("Starting work calculations...")
    N=8
    for mat in mats:
        print(mat)
        matrix = ss_getter.get_preprocessed(mat)
        calculator = Calculator(matrix)

        work = calculator.prep_work_naive(
            num_tirals=N,
            is_distributed=False,
        )
        print(f"Work Naive (non distributed): {work}")
        work = calculator.prep_work_naive(
            num_tirals=N,
            is_distributed=True,
        )
        print(f"Work Naive (distributed): {work}")
        work = calculator.prep_work_binomial(
            is_distributed=False
        )
        print(f"Work Binomial (non distributed): {work}")
        work = calculator.prep_work_binomial(
            is_distributed=True
        )
        print(f"Work Binomial (distributed): {work}")
        work = calculator.prep_work_baseline(
            is_distributed=False
        )
        print(f"Work Binomial (non distributed): {work}")
        work = calculator.prep_work_baseline(
            is_distributed=True
        )
        print(f"Work Binomial (distributed): {work}")
