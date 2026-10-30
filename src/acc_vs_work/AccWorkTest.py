from warnings import warn

import numpy as np
import scipy
from tqdm import tqdm

from src.acc_vs_work.AccWorkResult import AccWorkResult
from src.acc_vs_work.SSGetter import SSGetter
from src.acc_vs_work.util.Approximator import Approximator
from src.acc_vs_work.util.Power import Power
from src.acc_vs_work.util.ReductionType import ReductionType
from src.acc_vs_work.util.WorkCalculator import WorkCalculator
from src.Test import Test
from src.util.ResultType import ResultType


class AccWorkTest(Test):
    """For doing some tests"""

    result_type: ResultType
    mat_getter: SSGetter
    mats: list[str]
    num_trials: list[int]
    tol: float
    num_avg: int
    max_iter: int
    seed: int

    def __init__(
        self,
        result_type: ResultType,
        mat_getter: SSGetter,
        mats: list[str],
        num_trials: list[int],
        tol: float,
        num_avg: int,
        max_iter: int,
        seed: int,
    ) -> None:
        """Create a Result object

        Args:
            result_type (ResultType): The type of result for this test
            mat_getter (SSGetter): For generating matrices for this test
            num_avg (int): The number averaged in the result (to reduce noise) # TODO
            mats (list[str]): The matrices to test
            on_easley (bool): on_easley ->  don't display, not on_easly -> display!
            max_iter (int): max # of iterations of power
            tol (float): Stopping tolerance of power iteration
            seed (int): for repeated randomness
            num_trials (list[int]): Number of bernoulli trials
        """
        assert result_type is ResultType.ACC_VS_WORK, (
            f"result_type not ACC_VS_WORK: {result_type}"
        )
        # Call to Parent's __init__()
        super().__init__(
            result_type=result_type,
            mat_getter=mat_getter,
        )
        self.result_type = result_type
        self.mat_getter = mat_getter
        self.num_avg = num_avg
        self.mats = mats
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.num_trials = num_trials

    def __binomial_test(
        self,
        A: scipy.sparse.sparray,
        v0: np.ndarray,
        num_trials: int,
        is_distributed: bool,
    ) -> tuple[np.ndarray, np.ndarray, int, str]:
        """Run the parallelizable approach, return results

        Args:
            A (scipy.sparse.sparray): Matrix in question
            v0 (np.ndarray): Initial guess for top eigenvector
            max_iter (int): Max number of iterations for power
            tol (float): Tolerance of power iteration
            num_trials (int): Number of things to average
            seed (int): For repeatable randomization
            is_distributed (bool): Changes ammount of work that goes into computing
            an approximation for A^TA

        Returns: #TODO fix this comment
            xs (np.ndarray): work per iteration
            ys (np.ndarray): score of approximate eigenvector per iteration
            nnz (int): Number of nonzeroes in the approximation of the 2nd moment
            lbl (str): label of this test
        """
        rng = np.random.default_rng(seed=self.seed)
        reduct_type = ReductionType.SND_MOMENT_BINOMIAL

        approximator = Approximator(A, rng)
        tilde_A_sq = approximator.generate(
            reduct_type=reduct_type,
            n_trials=num_trials,
        )

        power = Power(
            matrix=tilde_A_sq,
        )
        scores = power.test(
            v0=v0,
            max_iter=self.max_iter,
            tol=self.tol,
            goal_mat=A.transpose @ A,
        )

        calculator = WorkCalculator(
            matrix=tilde_A_sq,
            reduct_type=reduct_type,
            is_distributed=is_distributed,
        )
        work = calculator.calculate(scores.shape[0])

        if num_trials == 1:
            lbl = f"{num_trials} trial"
        else:
            lbl = f"{num_trials} trials"

        return work, scores, tilde_A_sq.nnz, lbl

    def __baseline_pays(
        self,
        A: scipy.sparse.sparray,
        v0: np.ndarray,
        is_distributed: bool,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Run the default power iteration on this matrix, charging it for the
        initial cost of computing A^TA

        Args:
            A: the original matrix
            v0: initial guess for top eigenvector
            max_iter: maximum number of iterations to do power iteration
            tol (float): how much precision before terminating power
            is_distributed (bool): Changes ammount of work that goes into computing
            A^TA

        Returns: #TODO fix this comment
            tuple [np.ndarray, np.ndarray, str]:
                np.ndarray: the x-values (the ammount of work done),
                np.ndarray: the y-values (the score of the vector),
                str: the string representation of this test
        """
        matrix = A.transpose() @ A
        power = Power(
            matrix=matrix,
        )
        scores = power.test(
            v0=v0,
            max_iter=self.max_iter,
            tol=self.tol,
        )

        # Add initial work
        calculator = WorkCalculator(
            matrix=matrix,
            reduct_type=ReductionType.SND_MOM_NO_REDUCTION,
            is_distributed=is_distributed,
        )
        work = calculator.calculate(num_iter=scores.shape[0])
        return work, scores, "baseline"

    def run_test(
        self,
        disp_result: bool = False,
        log_y: bool = True,
        run_length_type: int = 2,
    ) -> None:
        """Run the test with some display parameters

        Args:
            disp_results (bool): Display results?
            log_y (bool): Y-axis of plot log scaled?
            run_length_type (int): When averaging variable length sequences,
            must choose to average over all or part of them:
                1 -> average over those that exist for all trials
                2 -> average over the median num trials
                3 -> average over all (the outlier longest will be weighted 100%)
        """
        for mat in self.mats:
            print(mat)
            matrix = self.mat_getter.get_preprocessed(mat_name=mat)
            if matrix is None:
                warn(f"skipping matrix: {mat}")
                continue
            snd_moment_A = matrix.transpose() @ matrix
            two_norm = scipy.sparse.linalg.norm(
                x=snd_moment_A,
                ord=2,
            )
            rng = np.random.default_rng(seed=5334)
            rand_vect = rng.normal(loc=0.0, scale=0.0625, size=matrix.shape[0])
            normalized_vect = rand_vect / np.linalg.norm(rand_vect, ord=2)

            work_baseline, acc_baseline, _ = self.__baseline_pays(  # TODO
                A=matrix,
                v0=normalized_vect,
                is_distributed=True,  # Always assuming distributed -> less work
            )

            lbls = []
            num_Ns_tested = len(self.num_trials)
            test_run_iters = np.zeros((num_Ns_tested, self.num_avg), int)
            nnzs = np.zeros((num_Ns_tested, self.num_avg))
            work = np.full((num_Ns_tested, self.max_iter), np.nan)
            accuracies = np.full(
                (num_Ns_tested, self.num_avg, self.max_iter), np.nan
            )  # NaNs for that good averaging

            for i, n_trials in tqdm(enumerate(self.num_trials)):
                lbl = ""
                for j in range(self.num_avg):
                    xs_temp, ys_temp, nnzs[i, j], lbl = self.__binomial_test(  # TODO
                        A=matrix,
                        v0=normalized_vect,
                        num_trials=n_trials,
                        is_distributed=True,  # Always distributed => less work
                    )
                    # Iteration tracking
                    curr_iter = ys_temp.shape[0]
                    accuracies[i, j, :curr_iter] = ys_temp
                    test_run_iters[i, j] = curr_iter
                    if test_run_iters[i, j] == np.max(test_run_iters[i]):
                        # Track longest!
                        work[i, : test_run_iters[i, j]] = xs_temp

                lbls.append(lbl)

            result = AccWorkResult(
                mat_name=mat,
                mat_shape=matrix.shape,
                is_sparse=True,
                snd_moment_norm=two_norm,
                snd_moment_nnz=snd_moment_A.nnz,
                max_iter=self.max_iter,
                tol=self.tol,
                seed=self.seed,
                num_avg=self.num_avg,
                n_binom_trials=self.num_trials,
                lbls=lbls,
                notes=None,
                test_run_iters=test_run_iters,
                nnzs=nnzs,
                baseline_work=work_baseline,
                baseline_accuracies=acc_baseline,
                work=work,
                accuracies=accuracies,
            )
            file_id = result.save_results()
            if disp_result:
                result = AccWorkResult(file_id=file_id)
                result.display_results(log_y=log_y, run_length_type=run_length_type)

    def run(self) -> None:
        """Run the test!"""
        self.run_test()


if __name__ == "__main__":
    """For testing purporses"""
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
    tol = 0  # tolerance of zero -> machine percision
    max_iter = 2**17  # 131, 072
    num_avg = 32
    seed = 7

    n_trials = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
    ]

    ss_getter = SSGetter(
        format=SparseFormat.CSR,
        nzbounds=(1000, 100000),
    )

    test = AccWorkTest(
        result_type=ResultType.ACC_VS_WORK,
        mat_getter=ss_getter,
        mats=mats,
        num_trials=n_trials,
        tol=tol,
        num_avg=num_avg,
        max_iter=max_iter,
        seed=seed,
    )
    test.run_test(
        disp_result=True,
        log_y=True,
        run_length_type=2,
    )
