"""For some repeated calculations (ex: work)"""

import warnings

import numpy as np
import scipy

from ..util.constants import THIRTY_TWO_BIT_PRECISION


class Calculator:
    """For doing some matrix involved calculations"""

    def rayleigh_quotient(
        self,
            x:np.ndarray,
            A:scipy.sparse.sparray,
    ) -> float:
        """Get the rayleigh quotient (score) of the given array-matrix combo

        Args:
            x (np.ndarray): array in question
            A (scipy.sparse.sparray): matrix in question

        Returns:
            float: The Rayleigh quotient
        """
        assert np.linalg.norm(x, ord=2) < 1 + THIRTY_TWO_BIT_PRECISION, f"vect not normalized: {np.linalg.norm(x, ord=2)}"
        assert np.linalg.norm(x, ord=2) > 1 - THIRTY_TWO_BIT_PRECISION, f"vect not normalized: {np.linalg.norm(x, ord=2)}"

        return (x.T @ (A @ x)).item()

    def check_vals(
        self,
        max:float,
        xs:np.ndarray,
        check_max:bool,
    ):
        """Adjust the x values s.t. they are on a range from 0 to 1, where if x_i
        equals max then set that value to one

        Args:
            max (float): The maximum possible value
            xs (np.ndarray): The list of values
            check_max (bool): If True -> throw error if any values in xs are more
            than max, do no such error handling otherwise
        """
        assert np.all(xs >= 0), f"Scores must be nonnegative"
        if check_max and (np.max(xs) > (max + THIRTY_TWO_BIT_PRECISION)):
            raise ValueError(f"np.max(xs) = {np.max(xs)} > {max}")


    def rel_value(
        self,
            max:float,
            xs:np.ndarray,
            check_max:bool,
    ) -> np.ndarray:
        """Adjust the x values s.t. they are on a range from 0 to 1, where if x_i
        equals max then set that value to one

        Args:
            max (float): The maximum possible value
            xs (np.ndarray): The list of values
            check_max (bool): If True -> throw error if any values in xs are more
            than max, do no such error handling otherwise

        Returns:
            np.ndarray: Those adjust values
        """
        self.check_vals(
            max,
            xs,
            check_max,
        )

        return xs / max

    def rel_residue(
        self,
        max: float,
        xs:np.ndarray,
        check_max:bool,
    ) -> np.ndarray:
        """Calculate relative residual

        Args:
            max (float): The maximum possible value in xs

        Returns:
            residue (np.ndarray): The relative residue
            """
        self.check_vals(
            max,
            xs,
            check_max,
        )
        return np.absolute(max - xs) / max

    def rel_error(
        self,
            approx: float,
            true: float,
    ) -> float:
        """Get the relative error between the two values

        Args:
            approx (float): The approximate value
            true (float): The true value

        Returns:
            float: The relative error
        """
        rel_error = abs(approx - true) / abs(true)
        # print(f"approx - true / approx = {abs(approx - true)} / {abs(true)} = {rel_error}")
        return rel_error

    def terminate_power(
        self,
        prev_iter:float,
        curr_iter:float,
        tolerance:float,
    ) -> bool:
        """Stop power iteration?

        Args:
            prev_iter (float): The 'score' of the previous vector approximation
            found via power iteration

        Returns:
            terminate (bool):
                True -> stop power iteration (converged)
                False -> continue power iteration
        """
        if curr_iter < 1e-14:
            print(f"rel_error(prev_iter, curr_iter)={self.rel_error(prev_iter, curr_iter)}")
            print(f"{self.rel_error(prev_iter, curr_iter)} <= {tolerance} = {self.rel_error(prev_iter, curr_iter) <= tolerance}")
        return (self.rel_error(prev_iter, curr_iter) <= tolerance) or prev_iter > curr_iter

    def online_avg(
            self,
            old_avg:np.ndarray,
            new_val:np.ndarray,
            new_total:int,
    ) -> np.ndarray:
        """Generate the new average given the old + the new + the new total count

        Args:
            old_avg (scipy.sparse): The current average
            new_val (scipy.sparse): The value which is updating the average
            new_total (int): The grand total ammount

        Returns:
            scipy.sparse: The new average
        """

        return old_avg + ((new_val - old_avg) / new_total)

    def init_test(
        self,
            A:scipy.sparse.sparray,
            v0:np.ndarray,
            max_iter:int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Initialize this power iteration test

        Args:
            A (scipy.sparse.sparray): The matrix in question
            v0 (np.ndarray): The initial guess for the eigenvector
            max_iter (int): the maximum number of iterations

        Returns:
            tuple[np.ndarray, np.ndarray, int]:
                np.ndarray: the array which will hold the top eigenvalues
                np.ndarray: the array which will hold the top eigenvectors
                int: the initial number of iterations
        """
        scores = np.zeros(shape=(max_iter,))
        vects = np.zeros(shape=(max_iter,v0.shape[0]))

        scores[0] = self.rayleigh_quotient(x=v0, A=A)
        vects[0] = v0
        iter = 1

        return scores, vects, iter

    def work_power(
        self,
        matrix: scipy.sparse.sparray,
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
        total_work = iterations * nnzs  # Doing SpMv multiplication per iteration
        return total_work

    def work_prep_baseline(
        self,
        matrix: scipy.sparse.sparray,
        is_distributed: bool,
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
        if (
            type(self.matrix) != scipy.sparse.csr_array
            and type(self.matrix) != scipy.sparse._csr.csr_matrix
        ):
            warnings.warn(f"A must be CSR, given {type(self.matrix)}")

        row_nnzs = np.diff(self.matrix.indptr)
        row_work = row_nnzs * row_nnzs

        assert (
            row_nnzs.shape[0] == self.matrix.shape[0]
        )  # TODO: delete this after sure it is workign

        if is_distributed:  # TODO: have return both, get rid of is_distributed
            return np.max(row_work)

        return np.sum(row_work)

    def work_prep_naive(
        self,
        matrix: scipy.sparse.sparray,
        num_tirals: int,
        is_distributed: bool,
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
        mults_per_iter = (
            self.matrix.nnz
            + self.matrix.shape[0]
            + self.matrix.nnz * self.matrix.shape[0]
        )
        if is_distributed:
            return mults_per_iter
        return num_tirals * mults_per_iter

    def work_prep_binomial(
        self,
        matrix: scipy.sparse.sparray,
        is_distributed: bool,
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
        return self.work_prep_baseline(
            matrix=matrix,
            is_distributed=is_distributed,
        )


if __name__ == "__main__":
    """Yeahhh
    """
    from ..util.SparseFormat import SparseFormat
    from .SSGetter import SSGetter

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
        calculator = Calculator()

        work = calculator.work_prep_naive(
            matrix=matrix,
            num_tirals=N,
            is_distributed=False,
        )
        print(f"Work Naive (non distributed): {work}")
        work = calculator.work_prep_naive(
            matrix=matrix,
            num_tirals=N,
            is_distributed=True,
        )
        print(f"Work Naive (distributed): {work}")
        work = calculator.work_prep_binomial(
            matrix=matrix,
            is_distributed=False
        )
        print(f"Work Binomial (non distributed): {work}")
        work = calculator.work_prep_binomial(
            matrix=matrix,
            is_distributed=True,
        )
        print(f"Work Binomial (distributed): {work}")
        work = calculator.work_prep_baseline(
            matrix=matrix,
            is_distributed=False,
        )
        print(f"Work Binomial (non distributed): {work}")
        work = calculator.work_prep_baseline(
            matrix=matrix,
            is_distributed=True,
        )
        print(f"Work Binomial (distributed): {work}")

if __name__ == '__main__':
    """Yeahhh
    """
    from ...bounds.preprocess import preprocess
    from .approx import get_next_A_tilde
    from ..non_proven_tests import test_A_tilde

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
    for mat in mats:
        print(mat)
        A, _ = preprocess(mat_name=mat)
        rng = np.random.default_rng(seed=5334)
        rand_vect = rng.normal(loc=0.0, scale=0.0625, size=A.shape[0])
        A_tilde = get_next_A_tilde(A, rng=rng)
        top_lambdas, top_vs, iter = test_A_tilde(
            A_tilde=A_tilde,
            v0=rand_vect,
            max_iter=20,
            tol=1/128,
        )
        top_vs = top_vs[0:iter]
        work = power_work(
            matrix=A,
            num_iter=top_lambdas.shape[0],
        )
        print(f"num_iter: {top_lambdas.shape[0]}")
        print(f"top_lambdas: {top_lambdas[-7:]}\ntop_vs: {top_vs[-7:]}\nwork: {work[-7:]}")
