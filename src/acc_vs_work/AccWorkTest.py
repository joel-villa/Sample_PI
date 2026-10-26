from warnings import warn

import numpy as np
import scipy
from tqdm import tqdm

from src.acc_vs_work.AccWorkResult import AccWorkResult
from src.acc_vs_work.SSGetter import SSGetter
from src.Test import Test
from src.util.ResultType import ResultType


def baseline(
    A: scipy.sparse.sparray,
    v0: np.ndarray,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Run the default power iteration on this matrix

    Args:
        A: the original matrix
        v0: initial guess for top eigenvector
        max_iter: maximum number of iterations to do power iteration
        tol (float): how much precision before terminating power

    Returns:
        tuple [np.ndarray, np.ndarray, str]:
        np.ndarray: the x-values (the ammount of work done),
        np.ndarray: the y-values (the score of the vector),
        str: the string representation of this test
    """
    v = v0
    scores, _, iter = comp.init_test(
        A=A,
        v0=v0,
        max_iter=max_iter,
    )
    while iter < max_iter:
        lam, v = power(
            A=A,
            v0=v,
            num_iter=1,
        )
        scores[iter] = lam
        if comp.terminate_power(scores[iter - 1], scores[iter], tol):
            iter += 1
            break
        iter += 1

    scores = scores[0:iter]
    power_wrk = work.power_work(matrix=A, num_iter=iter)
    lbl = "baseline"
    return power_wrk, scores, lbl


class AccWorkTest(Test):
    """For doing some tests"""

    result_type: ResultType
    mat_getter: SSGetter
    log_y: bool
    run_length_type: int
    num_avg: int
    mats: list[str]
    on_easley: bool
    max_iter: int
    tol: float
    seed: int
    num_trials: list[int]

    def __init__(
        self,
        result_type: ResultType,
        mat_getter: SSGetter,
        log_y: bool,
        run_length_type: int,
        num_avg: int,
        mats: list[str],
        on_easley: bool,
        max_iter: int,
        tol: float,
        seed: int,
        num_trials: list[int],
    ) -> None:
        """Create a Result object

        Args:
            result_type (ResultType): The type of result for this test
            mat_getter (SSGetter): For generating matrices for this test
            log_y (bool): Y-axis of plot log scaled?
            run_length_type (int): When averaging variable length sequences, must choose
            to average over all or part of them.
                1 -> average over those that exist for all trials
                2 -> average over the median num trials
                3 -> average over all (the outlier longest will be weighted 100%)
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
        self.log_y = log_y
        self.run_length_type = run_length_type
        self.num_avg = num_avg
        self.mats = mats
        self.on_easley = on_easley
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
        tilde_A_sq = binomial_A_tilde_sq(
            A=A,
            num_trials=num_trials,
            rng=rng,
        )
        init_work = work.work_binomial(  # TODO: work calculator
            A=A, is_distributed=is_distributed
        )

        xs, ys = test(
            A_sq=A.transpose() @ A,
            tilde_A_sq=tilde_A_sq,
            v0=v0,
            max_iter=self.max_iter,
            tol=self.tol,
            init_work=init_work,
        )

        #    lbl = f"binomial (distributed={is_distributed}), {num_trials} trials"
        if num_trials == 1:
            lbl = f"{num_trials} trial"
        else:
            lbl = f"{num_trials} trials"

        #    print(f"xs[:8]: {xs[:8]}")

        return xs, ys, tilde_A_sq.nnz, lbl

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
        xs, ys, lbl = baseline(
            A=A.transpose() @ A, v0=v0, max_iter=self.max_iter, tol=self.tol
        )

        # Add initial work
        init_work = work.work_baseline(
            A=A,
            is_distributed=is_distributed,
        )
        xs += init_work

        #    return xs, ys, f"{lbl} (is_distributed={is_distributed})"
        return xs, ys, lbl

    def run(self) -> None:
        """Run the test!"""
        # TODO: this guy

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
            if not self.on_easley:
                result = AccWorkResult(file_id=file_id)
                result.display_results(log_y=True, run_length_type=self.run_length_type)
