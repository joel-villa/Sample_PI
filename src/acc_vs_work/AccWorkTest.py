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
            mats (list[str]): The matrices to test
            num_trials (list[int]): Number of bernoulli trials
            tol (float): Stopping tolerance of power iteration
            num_avg (int): The number averaged in the result (to reduce noise) # TODO
            max_iter (int): max # of iterations of power
            seed (int): for repeated randomness
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

    def __generic_test(
        self,
        matrix: scipy.sparse.sparray,
        v0: np.ndarray,
        reduct_type: ReductionType,
        goal_mat: scipy.sparse.sparray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        power = Power(
            matrix=matrix,
        )
        scores = power.test(
            v0=v0, max_iter=self.max_iter, tol=self.tol, goal_mat=goal_mat
        )

        # Add initial work
        calculator = WorkCalculator(
            matrix=matrix,
            reduct_type=reduct_type,
            is_distributed=self.is_distributed,
        )
        work = calculator.calculate(num_iter=scores.shape[0])
        return work, scores

    def __nnz_counts(
        self,
        matrix: scipy.sparse.sparray,
    ) -> tuple[int, np.ndarray]:
        """Get the nonzero information of significance

        Args:
            matrix (scipy.sparse.sparray): The matrix in question

        Returns:
             nnz (int): The number of nonzeros in the matrix
             row_nnzs (np.ndarray): The number of nonzeros per row in the matrix
    	"""
        return matrix.nnz, np.diff(matrix.indptr)

    def __snd_mom_binomial_test(
        self,
        A: scipy.sparse.sparray,
        v0: np.ndarray,
        num_trials: int,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Run the parallelizable approach, return results

        Args:
            A (scipy.sparse.sparray): Matrix in question
            v0 (np.ndarray): Initial guess for top eigenvector
            num_trials (int): Number of things to average

        Returns: #TODO fix this comment
            nnz (int): Number of nonzeroes in the approximation of the 2nd moment
            row_nnzs (np.ndarray): The number of nonzeros per row of the
            approximation
            scores (np.ndarray): score of approximate eigenvector per iteration
        Resources:
            - Tuple unpacking:
                https://www.w3schools.com/python/python_tuples_unpack.asp
        """
        rng = np.random.default_rng(seed=self.seed)

        approximator = Approximator(A, rng)
        tilde_A_sq = approximator.generate(
            reduct_type=ReductionType.SND_MOMENT_BINOMIAL,
            n_trials=num_trials,
        )

        power = Power(
            matrix=tilde_A_sq,
        )
        scores = power.test(
            v0=v0,
            max_iter=self.max_iter,
            tol=self.tol,
            goal_mat=A.transpose() @ A,
        )
        return *self.__nnz_counts(tilde_A_sq), scores

    def __snd_mom_baseline(
        self,
        A: scipy.sparse.sparray,
        v0: np.ndarray,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Run the default power iteration on this matrix, charging it for the
        initial cost of computing A^TA

        Args:
            A: the original matrix
            v0: initial guess for top eigenvector

        Returns: #TODO fix this comment
            nnz (int): Number of nonzeroes in the 2nd moment
            row_nnzs (np.ndarray): The number of nonzeros per row of the
            second moment
            scores (np.ndarray): score of approximate eigenvector per iteration
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
        return *self.__nnz_counts(matrix), scores

    def __fst_mom_baseline(
        self,
        matrix: scipy.sparse.sparray,
        v0: np.ndarray,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Run the baseline for the first moment tests

        Args:
            matrix (scipy.sparse.sparray): The matrix in question
            v0 (np.ndarray): Initial guess for the top eigenvector
            num_trials (int): Number of things to average

        Returns:
            nnz (int): Number of nonzeroes
            row_nnzs (np.ndarray): The number of nonzeros per row
            scores (np.ndarray): score of approximate eigenvector per iteration
        """

        power = Power(
            matrix=matrix,
        )
        scores = power.test(
            v0=v0,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        return *self.__nnz_counts(matrix), scores

    def __fst_mom_pre_avg(
        self,
        matrix: scipy.sparse.sparray,
        v0: np.ndarray,
        num_trials: int,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Doing averaging prior to power iteration, track accuracies and nnzs

        Args:
            matrix (scipy.sparse.sparray): The matrix in question
            v0 (np.ndarray): Initial guess for the top eigenvector
            num_trials (int)

        Returns:
            nnz (int): Number of nonzeroes of approximation
            row_nnzs (np.ndarray): The number of nonzeros per row of approximation
            scores (np.ndarray): score of approximate eigenvector per iteration
        """

        rng = np.random.default_rng(seed=self.seed)
        reduct_type = ReductionType.AVG_BEFORE

        approximator = Approximator(matrix, rng)
        approx_mat = approximator.generate(
            reduct_type=reduct_type,
            n_trials=num_trials,
        )

#        work, scores = self.__generic_test(
#            matrix=approx_mat,
#            v0=v0,
#            reduct_type=reduct_type,
#            goal_mat=matrix,
#        )

        power = Power(
            matrix=approx_mat,
        )
        scores = power.test(
            v0=v0,
            max_iter=self.max_iter,
            tol=self.tol,
            goal_mat=matrix,
        )
        return *self.__nnz_counts(approx_mat), scores

    def __fst_mom_post(
        self,
        matrix: scipy.sparse.sparray,
        v0: np.ndarray,
        num_trials: int,
        avg_type: int,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Run the parallelizable approach, return results

        Args:
            matrix (scipy.sparse.sparray): Matrix in question
            v0 (np.ndarray): Initial guess for top eigenvector
            num_trials (int): Number of things to average
            avg_type (int): How to handle average of variable length arrays?

        Returns: #TODO fix this comment
            nnz (int): Number of nonzeroes of approximation
            row_nnzs (np.ndarray): The number of nonzeros per row of approximation
            scores (np.ndarray): score of approximate eigenvector per iteration

        Resources:
            - Numpy average of rows:
                https://numpy.org/doc/2.5/reference/generated/numpy.average.html
        """
        rng = np.random.default_rng(seed=self.seed)
        reduct_type = ReductionType.AVG_AFTER
        approximator = Approximator(matrix=matrix, rng=rng)
        nnzs = np.zeros((num_trials,))
        row_nnzs = np.zeros((num_trials, matrix.shape[0]))
        vects = np.full((num_trials, max_iter, v0.shape[0]), np.nan)
        n_iters = np.zeros((num_trials,), dtype=int)
        power = None

        for i in range(num_trials):
            approx_mat = approximator.generate(
                reduct_type=reduct_type,
            )
            nnzs[i] = approx_mat.nnz
            row_nnzs[i, :] = np.diff(approx_mat.indptr)

            power = Power(
                matrix=approx_mat,
            )
            vects_temp = power.get_vects(
                v0=v0,
                max_iter=self.max_iter,
                tol=self.tol,
                goal_mat=matrix,
            )
            n_iters[i] = vects_temp.shape[0]
            vects[i, : n_iters[i]] = vects_temp

        sig_iter = -1

        match avg_type:
            case 1:
                sig_iter = np.min(n_iters)
            case 2:
                sig_iter = np.median(n_iters)
            case 3:
                sig_iter = np.min(n_iters)
            case _:
                raise NotImplementedError()
        sig_iter = round(sig_iter)
        vects = vects[:, :sig_iter, :]

        # Axis = 0 => one average per row
        axis = 0
        avg_vects = np.nanmean(vects, axis=axis)
        avg_row_nnzs = np.average(row_nnzs, axis=axis)
        avg_nnzs = np.average(nnzs)

        if power is None:
            raise ValueError("Why work None type?")
        scores = np.zeros((sig_iter,))
        for i in range(sig_iter):
            avg_vects[i] = avg_vects[i] / np.linalg.norm(avg_vects[i])

            scores[i] = power.rayleigh_quotient(
                A=matrix,
                x=avg_vects[i],
            )

        return avg_nnzs, avg_row_nnzs, scores

    def run_test(
        self,
        disp_result: bool = False,
        log_y: bool = True,
        is_distributed: bool = False,
        run_length_type: int = 2,
        reduct_type: ReductionType = ReductionType.SND_MOMENT_BINOMIAL,
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

            rng = np.random.default_rng(seed=self.seed)
            rand_vect = rng.normal(loc=0.0, scale=0.0625, size=matrix.shape[0])
            normalized_vect = rand_vect / np.linalg.norm(rand_vect, ord=2)

            match reduct_type:
                case ReductionType.AVG_AFTER | ReductionType.AVG_BEFORE:
                    nnz_baseline, row_nnzs_baseline, acc_baseline = self.__fst_mom_baseline(  # TODO
                        matrix=matrix,
                        v0=normalized_vect,
                    )
                case ReductionType.SND_MOMENT_BINOMIAL:
                    nnz_baseline, row_nnzs_baseline, acc_baseline = self.__snd_mom_baseline(  # TODO
                        A=matrix,
                        v0=normalized_vect,
                    )
                case _:
                    raise ValueError(f"Unexptected type: {reduct_type}")

            num_Ns_tested = len(self.num_trials)
            test_run_iters = np.zeros((num_Ns_tested, self.num_avg), int)
            nnzs = np.zeros((num_Ns_tested, self.num_avg))
            accuracies = np.full(
                (num_Ns_tested, self.num_avg, self.max_iter), np.nan
            )  # NaNs for that good averaging
            test_row_nnzs = np.full(
                (num_Ns_tested, self.num_avg, matrix.shape[0]), np.nan
            )

            for i, n_trials in tqdm(enumerate(self.num_trials)):
                for j in range(self.num_avg):
                    match reduct_type:
                        case ReductionType.SND_MOMENT_BINOMIAL:
                            nnzs[i, j], test_row_nnzs[i,j,:], acc_temp = (
                                self.__snd_mom_binomial_test(
                                    A=matrix,
                                    v0=normalized_vect,
                                    num_trials=n_trials,
                                )
                            )
                        case ReductionType.AVG_BEFORE:
                            nnzs[i, j], test_row_nnzs[i,j,:], acc_temp = self.__fst_mom_pre_avg( #TODO save label on outside
                                matrix=matrix,
                                v0=normalized_vect,
                                num_trials=n_trials,
                            )
                        case ReductionType.AVG_AFTER:
                            nnzs[i, j], test_row_nnzs[i,j,:], acc_temp,  = self.__fst_mom_post(
                                matrix=matrix,
                                v0=normalized_vect,
                                num_trials=n_trials,
                                avg_type=2,
                            )
                        case _:
                            raise ValueError(
                                f"Unexpected reduction type: {reduct_type}"
                            )

                    # Iteration tracking
                    curr_iter = acc_temp.shape[0]
                    accuracies[i, j, :curr_iter] = acc_temp
                    test_run_iters[i, j] = curr_iter

            #result = AccWorkResult( #TODO: save stuff to make a Work Calculator inside of results
            #    mat_name=mat,
            #    mat_shape=matrix.shape,
            #    is_sparse=True,
            #    is_distributed=self.is_distributed,
            #    matrix_norm=two_norm,
            #    baseline_nnz=goal_mat.nnz,
            #    max_iter=self.max_iter,
            #    tol=self.tol,
            #    seed=self.seed,
            #    num_avg=self.num_avg,
            #    n_trials=self.num_trials,
            #    lbls=lbls,
            #    notes=None,
            #    test_run_iters=test_run_iters,
            #    test_nnzs=nnzs,
            #    baseline_work=work_baseline,
            #    baseline_accuracies=acc_baseline,
            #    work=work,
            #    accuracies=accuracies,
            #    reduct_type=reduct_type,
            #)

            two_norm = scipy.sparse.linalg.norm(
                x=matrix.transpose() @ matrix,  # TODO: this doesn't seem right
                ord=2,
            )
            if reduct_type is not ReductionType.SND_MOMENT_BINOMIAL:
                # square root the guy
                two_norm = np.sqrt(two_norm) #TODO rerun the guy
            result = AccWorkResult(
                reduct_type=reduct_type,
                mat_name=mat,
                is_sparse=True,
                baseline_nnz=nnz_baseline,
                max_iter=self.max_iter,
                num_avg=self.num_avg,
                seed=self.seed,
                mat_shape=matrix.shape,
                matrix_norm=two_norm,
                tol=self.tol,
                trial_list=self.num_trials,
                notes="Mon 9 Nov",
                test_run_iters=test_run_iters,
                test_nnzs=nnzs,
                test_row_nnzs=test_row_nnzs, #TODO
                baseline_row_nnzs=row_nnzs_baseline, #TODO
                baseline_accuracies=acc_baseline,
                accuracies=accuracies,
            )
            file_id = result.save_results()
            if disp_result:
                result = AccWorkResult(reduct_type=reduct_type, file_id=file_id)
                result.display_results(
                    log_y=log_y,
                    run_length_type=run_length_type,
                    is_distributed=is_distributed,
                )

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
    is_distributed = False

    n_trials = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
    ]
#    n_trials = [
#        1,
#        2,
#        4,
#    ]

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

    for reduction_type in [ReductionType.AVG_BEFORE, ReductionType.SND_MOMENT_BINOMIAL]:
        test.run_test(
                disp_result=False,
                log_y=True,
                run_length_type=2,
                reduct_type=reduction_type,
            )
