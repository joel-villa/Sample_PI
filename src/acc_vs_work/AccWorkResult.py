"""The results of testing have some data associated with  them

Resources:
    https://docs.python.org/3/tutorial/classes.html
    https://www.geeksforgeeks.org/python/method-and-constructor-overloading-in-python/
"""

import json

import matplotlib
import numpy as np

from src.acc_vs_work.util.ReductionType import ReductionType
from src.acc_vs_work.util.WorkCalculator import WorkCalculator

matplotlib.use("QtAgg")

import matplotlib.pyplot as plt

from src.Result import Result
from src.util.constants import THIRTY_TWO_BIT_PRECISION

# Some helper fucntions:


def check_vals(
    max: float,
    xs: np.ndarray,
    check_max: bool,
):
    """Adjust the x values s.t. they are on a range from 0 to 1, where if x_i
    equals max then set that value to one

    Args:
        max (float): The maximum possible value
        xs (np.ndarray): The list of values
        check_max (bool): If True -> throw error if any values in xs are more
        than max, do no such error handling otherwise
    """
    assert np.all(xs >= 0 - THIRTY_TWO_BIT_PRECISION), (
        f"Scores must be nonnegative: {np.min(xs)}"
    )
    if check_max and (np.max(xs) > (max + THIRTY_TWO_BIT_PRECISION)):
        raise ValueError(f"np.max(xs) = {np.max(xs)} > {max}")


def rel_value(  # TODO move these into class
    max: float,
    xs: np.ndarray,
    check_max: bool,
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
    check_vals(
        max,
        xs,
        check_max,
    )

    return xs / max


def rel_residue(
    max: float,
    xs: np.ndarray,
    check_max: bool,
) -> np.ndarray:
    """Calculate relative residual

    Args:
        max (float): The maximum possible value in xs

    Returns:
         residue (np.ndarray): The relative residue
    """
    check_vals(
        max,
        xs,
        check_max,
    )
    return np.absolute(max - xs) / max


class AccWorkResult(Result):
    """The metadata to track"""

    TEST_STRING: str = "acc_vs_work"

    _mat_name: str
    _is_sparse: bool
    _baseline_nnz: int
    _max_iter: int
    _num_avg: int
    _seed: int
    _mat_shape: tuple[int, int]
    _matrix_norm: float
    _tol: float
    _trial_list: list[int]
    _notes: str | None

    _test_run_iters: np.ndarray
    _test_nnzs: np.ndarray
    _test_row_nnzs: np.ndarray
    _baseline_row_nnzs: np.ndarray
    _baseline_accuracies: np.ndarray
    _accuracies: np.ndarray

    _reduct_type: ReductionType

    def __init__(
        self,
        reduct_type: ReductionType,
        file_id: int | None = None,
        mat_name: str | None = None,
        is_sparse: bool | None = None,
        baseline_nnz: int | None = None,
        max_iter: int | None = None,
        num_avg: int | None = None,
        seed: int | None = None,
        mat_shape: tuple[int, int] | None = None,
        matrix_norm: float | None = None,
        tol: float | None = None,
        trial_list: list[int] | None = None,
        notes: str | None = None,
        test_run_iters: np.ndarray | None = None,
        test_nnzs: np.ndarray | None = None,
        test_row_nnzs: np.ndarray | None = None,
        baseline_row_nnzs: np.ndarray | None = None,
        baseline_accuracies: np.ndarray | None = None,
        accuracies: np.ndarray | None = None,
    ) -> None:
        """Initialize the data tracking object

        Args:
            reduct_type (str): Type of test run
            file_id (int | None): The id of the given results
            mat_name (str | None): The matrix being tested
            is_sparse (bool | None): Is this matrix sparse? If not assumed dense
            is_distributed (bool | None): Is the work distributed?
            baseline_nnz (int | None): Number of nonzeroes in A^TA
            max_iter (int | None): Maximum number of iterations of Power
            iteration
            num_avg (int | None): Number of runs to average over
            seed (int | None): The seed of random events
            mat_shape (tuple[int, int] | None): The shape of the matrix
            matrix_norm (float | None): 2 norm of the second moment of A
            tol (float | None): The tolerance of power iteration (0 ->
            machine precision)
            trial_list (list[int] | None): The number of binomial trials
            tested
            notes (str | None): Anything to note about this test?
            test_run_iters (np.ndarray | None): The number of iterations for
            every run of power
            test_nnzs (np.ndarray | None): The number of nonzeros of the 2nd
            moment approximation per trial
            test_row_nnzs (np.ndarray | None): For work calculations, the
            number of nonzeroes per row is useful
            baseline_nnzs (np.ndarray | None): For baseline work calculations
            baseline_accuracies (np.ndarray | None): The accuracy of the
            baseline
            accuracies (np.ndarray | None): The accuracy per test run (3d matrix)
        Resources:
            - Super Class Instatiation:
                https://stackoverflow.com/questions/2399307/how-to-invoke-the-super-constructor-in-python
        """
        super().__init__(f"{self.TEST_STRING}/{reduct_type}")

        if file_id:
            self.__load_results(file_id)
        else:
            if (
                mat_name is None
                or is_sparse is None
                or baseline_nnz is None
                or max_iter is None
                or num_avg is None
                or seed is None
                or mat_shape is None
                or matrix_norm is None
                or tol is None
                or trial_list is None
                or test_run_iters is None
                or test_nnzs is None
                or test_row_nnzs is None
                or baseline_row_nnzs is None
                or baseline_accuracies is None
                or accuracies is None
            ):
                raise ValueError(
                    "if file_name is None, most other fields cannot be None"
                )
            self._mat_name = mat_name
            self._mat_shape = mat_shape
            self._is_sparse = is_sparse
            self._matrix_norm = matrix_norm
            self._baseline_nnz = baseline_nnz
            self._max_iter = max_iter
            self._tol = tol
            self._seed = seed
            self._num_avg = num_avg
            self._trial_list = trial_list
            self._notes = notes
            self._test_run_iters = test_run_iters
            self._test_nnzs = test_nnzs
            self._test_row_nnzs = test_row_nnzs
            self._baseline_row_nnzs = baseline_row_nnzs
            self._baseline_accuracies = baseline_accuracies
            self._accuracies = accuracies
            self._reduct_type = reduct_type

        # Ensuring correct usage
        assert self._reduct_type in {
            ReductionType.AVG_AFTER,
            ReductionType.AVG_BEFORE,
            ReductionType.SND_MOMENT_BINOMIAL,
        }, f"Unexpected type: {self._reduct_type.name}"
        #        assert self.baseline_accuracies.shape == self.baseline_work.shape, (
        #            f"Incorrect dimensions {self.baseline_accuracies.shape} != {self.baseline_work.shape}"
        #        )

        n_trials = len(self._trial_list)
        assert n_trials == self._test_run_iters.shape[0], (
            f"Incorrect dimensions {n_trials} != {self._test_run_iters.shape[0]}"
        )
        assert n_trials == self._test_nnzs.shape[0], (
            f"Incorrect dimensions {n_trials} != {self._test_nnzs.shape[0]}"
        )
        assert n_trials == self._test_row_nnzs.shape[0], (
            f"Incorrect dimensions {n_trials} != {self._test_row_nnzs.shape[0]}"
        )
        #        assert n_trials == self.work.shape[0], (
        #            f"Incorrect dimensions {n_trials} != {self.work.shape[0]}"
        #        )
        assert n_trials == self._accuracies.shape[0], (
            f"Incorrect dimensions {n_trials} != {self._accuracies.shape[0]}"
        )

        assert self._num_avg == self._test_run_iters.shape[1], (
            f"Incorrect dimensions {self._num_avg} != {self._accuracies.shape[1]}"
        )
        assert self._num_avg == self._test_nnzs.shape[1], (
            f"Incorrect dimensions {self._num_avg} != {self._test_nnzs.shape[1]}"
        )
        assert self._num_avg == self._test_row_nnzs.shape[1], (
            f"Incorrect dimensions {self._num_avg} != {self._test_row_nnzs.shape[1]}"
        )
        assert self._num_avg == self._accuracies.shape[1], (
            f"Incorrect dimensions {self._num_avg} != {self._accuracies.shape[1]}"
        )

        assert self._mat_shape[0] == self._test_row_nnzs.shape[2], (
            f"Incorrect dimsinos: {self._mat_shape[0]} != {self._test_row_nnzs.shape[2]}"
        )
        assert self._mat_shape[0] == self._baseline_row_nnzs.shape[0], (
            f"Incorrect dimsinos: {self._mat_shape[0]} != {self._baseline_row_nnzs.shape[0]}"
        )

    def __load_results(self, file_id: int) -> None:
        """Load the results from the provided file name

        Args:
            file_name (int): The file of significance

        Resources:
            https://www.geeksforgeeks.org/python/convert-json-to-dictionary-in-python/
        """
        metadata = {}
        path_json, path_npz = self.get_paths(file_id)

        with open(path_json) as json_file:
            metadata = json.load(json_file)
            self._mat_name = metadata["mat_name"]
            self._mat_shape = metadata["mat_shape"]
            self._is_sparse = metadata["is_sparse"]
            self._matrix_norm = metadata["matrix_norm"]
            self._baseline_nnz = metadata["baseline_nnz"]
            self._max_iter = metadata["max_iter"]
            self._tol = metadata["tol"]
            self._seed = metadata["seed"]
            self._num_avg = metadata["num_avg"]
            self._trial_list = metadata["trial_list"]
            self._notes = metadata["notes"]
            self._reduct_type = ReductionType(metadata["reduct_type"])

        npzfile = np.load(path_npz)

        self._test_run_iters = npzfile["test_run_iters"]
        self._test_nnzs = npzfile["test_nnzs"]
        self._test_row_nnzs = npzfile["test_row_nnzs"]
        self._baseline_row_nnzs = npzfile["baseline_row_nnzs"]
        self._baseline_accuracies = npzfile["baseline_accuracies"]
        self._accuracies = npzfile["accuracies"]

    def save_results(self) -> int:
        """Save results to npz file and metadata to a json file

        Returns:
            file_id (int): The file id associated with this test

        Resources:
            https://www.geeksforgeeks.org/python/python-list-files-in-a-directory/
            https://www.geeksforgeeks.org/python/convert-string-to-integer-in-python/
            https://www.w3schools.com/PYTHON/python_try_except.asp
            https://numpy.org/doc/stable/reference/generated/numpy.savez.html
            https://www.geeksforgeeks.org/python/how-to-convert-python-dictionary-to-json/
        """
        file_id = self.get_next_file_id()

        path_json, path_npz = self.get_paths(file_id)

        metadata = {
            "mat_name": self._mat_name,
            "mat_shape": self._mat_shape,
            "is_sparse": self._is_sparse,
            "matrix_norm": self._matrix_norm,
            "baseline_nnz": self._baseline_nnz,
            "max_iter": self._max_iter,
            "tol": self._tol,
            "seed": self._seed,
            "num_avg": self._num_avg,
            "trial_list": self._trial_list,
            "reduct_type": self._reduct_type,
            "notes": self._notes,
        }
        with open(path_json, "w") as json_file:
            json.dump(metadata, json_file, indent=4)

        np.savez(
            path_npz,
            test_run_iters=self._test_run_iters,
            test_nnzs=self._test_nnzs,
            test_row_nnzs=self._test_row_nnzs,
            baseline_row_nnzs=self._baseline_row_nnzs,
            baseline_accuracies=self._baseline_accuracies,
            accuracies=self._accuracies,
        )

        return file_id

    def display_results(
        self,
        run_length_type: int,
        is_distributed: bool,
        log_y: bool,
    ) -> None:
        """Display the results

        Args:
            run_length_type (int): determines how average of variable length
            arrays is calculated
            is_distributed (bool): Determines work calculations
            log_y (bool): take logarithm of y-axis when plotting?
        """
        fig, ax = plt.subplots()
        print("Backend:", matplotlib.get_backend())

        base_reduct_type: ReductionType

        match self._reduct_type:
            case ReductionType.AVG_AFTER | ReductionType.AVG_BEFORE:
                base_reduct_type = ReductionType.NO_REDUCTION
            case ReductionType.SND_MOMENT_BINOMIAL | ReductionType.SND_MOMENT_NAIVE:
                base_reduct_type = ReductionType.SND_MOM_NO_REDUCTION
            case _:
                raise ValueError(f"Unexpected reduction type: {self._reduct_type}")

        work_calculator = WorkCalculator(
            reduct_type=base_reduct_type,
            is_distributed=is_distributed,
            shape=self._mat_shape,
            nnz=self._baseline_nnz,
            row_nnzs=self._baseline_row_nnzs,
        )
        base_ys = self._baseline_accuracies
        base_xs = work_calculator.calculate(num_iter=base_ys.shape[0])

        #        work_offset = base_xs[0] #TODO: init work tracking?
        total_base_work = base_xs[-1]

        base_xs = rel_value(
            max=total_base_work,
            xs=base_xs,
            check_max=True,
        )
        try:
            base_ys = rel_residue(
                max=self._matrix_norm,
                xs=base_ys,
                check_max=True,
            )
        except AssertionError as e:
            print(e)
            plt.show()
            plt.close(fig)
            return

        ax.plot(base_xs, base_ys, label=f"baseline, nnz = {self._baseline_nnz}")

        num_avg = self._num_avg

        for i, n_trials in enumerate(self._trial_list):
            # Looping through Ns

            #            xs = self.work[i, :most_iter]
            ys_i = np.full(  # for averaging
                (num_avg, self._accuracies.shape[2]),
                np.nan,
            )
            xs_i = np.full(  # for averaging
                (num_avg, self._accuracies.shape[2]),
                np.nan,
            )

            for j in range(num_avg):
                # Looping through the runs (for averaging purposes)
                ys_temp = self._accuracies[i, j, : self._test_run_iters[i, j]]
                ys_i[j, : self._test_run_iters[i, j]] = rel_residue(
                    max=self._matrix_norm,
                    xs=ys_temp,
                    check_max=True,
                )

                work_calculator = WorkCalculator(
                    reduct_type=self._reduct_type,
                    is_distributed=is_distributed,
                    shape=self._mat_shape,
                    nnz=self._test_nnzs[i][j],
                    row_nnzs=self._test_row_nnzs[i][j],
                )
                xs_temp = work_calculator.calculate(num_iter=self._test_run_iters[i, j])
                # Scale to be between zero and one
                xs_i[j, : self._test_run_iters[i, j]] = rel_value(
                    max=total_base_work,
                    xs=xs_temp,
                    check_max=False,
                )

            ys = np.nanmean(ys_i, axis=0)
            xs = np.nanmean(xs_i, axis=0)
            nnz = round(np.mean(self._test_nnzs[i]))

            sig_iter: int
            match run_length_type:
                case 1:
                    sig_iter = np.min(self._test_run_iters[i])
                case 2:
                    sig_iter = round(np.median(self._test_run_iters[i]))
                case 3:
                    sig_iter = np.max(self._test_run_iters[i])
                case _:
                    raise ValueError(
                        f"Unexpected value for run_length_type: {run_length_type}"
                    )
            xs = xs[:sig_iter]
            ys = ys[:sig_iter]

            lbl: str
            if n_trials == 1:
                lbl = f"{n_trials} trial"
            else:
                lbl = f"{n_trials} trials"
            ax.plot(xs, ys, label=f"{lbl}, nnz = {nnz}")

        match self._reduct_type:
            case ReductionType.SND_MOMENT_BINOMIAL:
                ax.set_title(
                    f"Work vs. Accuracy of Top Eigenvector ({self._mat_name}:"
                    f"{self._mat_shape[0]}x{self._mat_shape[1]}, avg of {num_avg} runs, 2nd Moment)"
                )
            case ReductionType.AVG_AFTER:
                ax.set_title(
                    f"Work vs. Accuracy of Top Eigenvector ({self._mat_name}:"
                    f"{self._mat_shape[0]}x{self._mat_shape[1]}, avg of {num_avg} runs)"
                )
            case ReductionType.AVG_BEFORE:
                ax.set_title(
                    f"Work vs. Accuracy of Top Eigenvector ({self._mat_name}:"
                    f"{self._mat_shape[0]}x{self._mat_shape[1]}, avg of {num_avg} runs, 1st Moment)"
                )
        match self._reduct_type:
            case ReductionType.SND_MOMENT_BINOMIAL:
                ax.set_ylabel(r"$\frac{|A^TA| - |A^TA\tilde v_1|}{|A^TA|}$", rotation=0)
            case ReductionType.AVG_AFTER | ReductionType.AVG_BEFORE:
                ax.set_ylabel(r"$\frac{|A| - |A\tilde v_1|}{|A|}$", rotation=0)
        ax.set_xlabel("Approximate Proportion of Scalar Mults")
        if log_y:
            ax.set_yscale("log")
            # ax.set_ylim(1e-12, 1)
            # ax.yaxis.set_major_locator(
            #    plt.LogLocator(base=10)
            # )
            # ax.yaxis.set_major_formatter(
            #    plt.LogFormatterMathtext(base=10)
            # )

        print("Y-axis scale:", ax.get_yscale())
        ax.legend()
        fig.canvas.draw()
        plt.show()
        plt.close(fig)

    def disp_results(self) -> None:
        self.display_results(
            run_length_type=2,
            is_distributed=False,
            log_y=True,
        )


def main():
    """Nothing but a thang"""
    result_ids = range(1, 25)

    reduct_types = [
        ReductionType.SND_MOMENT_BINOMIAL,
        ReductionType.AVG_BEFORE,
    ]

    for id in result_ids:
        for reduct_type in reduct_types:
            results = AccWorkResult(reduct_type=reduct_type, file_id=id)
            print(f"reduct_type: {reduct_type}, id: {id}")
            results.display_results(run_length_type=2, is_distributed=False, log_y=True)


def main_2():
    reduct_type = ReductionType.AVG_BEFORE
    result = AccWorkResult(reduct_type=reduct_type, file_id=25)
    result.disp_results()


def main_3():
    file_id = 1

    result = AccWorkResult(reduct_type=ReductionType.AVG_BEFORE, file_id=file_id)
    result.disp_results()


if __name__ == "__main__":
    """For testing purporses"""
    main()
    # main_2()
    # main_3()
