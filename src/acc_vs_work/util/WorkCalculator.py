"""For some repeated calculations (ex: work)"""

import numpy as np
import scipy

from src.util.ReductionType import ReductionType


class WorkCalculator:
    """For doing some matrix involved calculations"""

    shape: tuple[int, int]
    nnz: int
    row_nnzs: np.ndarray
    reduct_type: ReductionType
    is_distributed: bool
    n_trials: int

    def __init__(
        self,
        reduct_type: ReductionType,
        is_distributed: bool,
        matrix: scipy.sparse.sparray | None = None,
        shape: tuple[int, int] | None = None,
        nnz: int | None = None,
        row_nnzs: np.ndarray | None = None,
    ) -> None:
        """Intiliaze a matrix calculator

        Args:
            matrix (scipy.sparse.sparray): The matrix in question
        """
        self.reduct_type = reduct_type
        self.is_distributed = is_distributed
        self.n_trials = -1

        if matrix is not None:
            self.shape = matrix.shape
            self.nnz = matrix.nnz
            self.row_nnzs = np.diff(matrix.indptr)
        else:
            if shape is None or nnz is None or row_nnzs is None:
                raise ValueError("If no matrix provided, parameters must be specified")
            if np.sum(row_nnzs) != nnz:
                raise ValueError(f"sum(row_nnzs) = {np.sum(row_nnzs)} != {nnz} = nnz")
            self.shape = shape
            self.nnz = nnz
            self.row_nnzs = row_nnzs

    def __work_power(
        self,
        num_iter: int,
    ) -> np.ndarray:
        """The amount of cummulative work done at every iteration of Power on the
        given matrix

        Args:
            num_iter (int): number of iterations run

        Returns:
            np.ndarray: A linearly increasing one-dimensional array of length
            num_iter
        """
        work_per_iter = 0
        if self.is_distributed:
            work_per_iter = np.max(self.row_nnzs)
        else:
            work_per_iter = np.sum(self.row_nnzs)

        iterations = np.arange(0, num_iter)
        total_work = (
            iterations * work_per_iter
        )  # Doing SpMv multiplication per iteration
        return total_work

    def __work_prep_baseline(
        self,
    ) -> int:
        """The number of scalar mults to compute A^TA

        Returns:
            int: Number of scalar multsf

        NOTE:
            A^TA can be rewritten as a sum of the outerproducts of the rows
            Work breakdown:
                The i'th row contributes w_i^2, where w_i is the nnz's in the
                i'th row
        """
        row_work = self.row_nnzs * self.row_nnzs

        if self.is_distributed:  # TODO: have return both, get rid of is_distributed
            return np.max(row_work)

        return np.sum(row_work)

    def __work_prep_naive(
        self,
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
            num_tirals (int): The ammount of approximations to average over
            is_distributed (bool): True -> taking maxium, False -> summing up work

        Returns:
            int: amount of scalar mults
        """
        mults_per_iter = (
            self.nnz
            + self.shape[0]
            + self.nnz * self.shape[0]
        )
        if self.is_distributed:
            return mults_per_iter
        return self.n_trials * mults_per_iter

    def __work_prep_binomial(
        self,
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
        return self.__work_prep_baseline()

    def calculate(self, num_iter: int) -> np.ndarray:
        """Calculate the amouant of work done for power iteration

        Args:
            num_iter (int): The number of iterations of power (will be the
            length of the array returned)
            is_distributed (bool): Changes work done in setup stages

        Returns:
             work (np.ndarray): Work done up and to that point
        """
        init_work = 0

        match self.reduct_type:
            case ReductionType.SND_MOM_NO_REDUCTION:
                init_work = self.__work_prep_baseline()
            case ReductionType.SND_MOMENT_BINOMIAL:
                init_work = self.__work_prep_binomial()
            case ReductionType.SND_MOMENT_NAIVE:
                if self.n_trials < 0:
                    raise ValueError(
                        "set WorkCalculator.n_trails before call to calculate"
                    )
                init_work = self.__work_prep_naive()
            case ReductionType.NO_REDUCTION | ReductionType.AVG_BEFORE | ReductionType.AVG_AFTER:
                init_work = 0
                # TODO: I think reducing does have some overhead not currently tracked
            case _:
                raise NotImplementedError()

        work = self.__work_power(num_iter=num_iter)

        return work + init_work


if __name__ == "__main__":
    """Yeahhh
    """
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
    ss_getter = SSGetter(
        format=SparseFormat.CSR,
        nzbounds=(1000, 100000),
    )
    print("Starting work calculations...")
    N = 8
    for mat in mats:
        print(mat)
        matrix = ss_getter.get_preprocessed(mat)

        for reduct_type in {
            ReductionType.SND_MOM_NO_REDUCTION,
            ReductionType.SND_MOMENT_NAIVE,
            ReductionType.SND_MOMENT_BINOMIAL,
        }:
            print(f"reduct_type: {reduct_type.name}")
            for is_dist in (True, False):
                print(f"is_dist: {is_dist}")
                calculator = WorkCalculator(
                    matrix=matrix,
                    reduct_type=reduct_type,
                    is_distributed=is_dist,
                )
                calculator.n_trials = 10
                print(calculator.calculate(num_iter=5))
