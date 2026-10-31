"""The results of testing have some data associated with  them

Resources:
    https://docs.python.org/3/tutorial/classes.html
    https://www.geeksforgeeks.org/python/method-and-constructor-overloading-in-python/
"""

import json

import matplotlib
import numpy as np

matplotlib.use("QtAgg")

import matplotlib.pyplot as plt

from ..Result import Result
from ..util.constants import THIRTY_TWO_BIT_PRECISION

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
    # assert np.all(xs >= 0), "Scores must be nonnegative"
    if check_max and (np.max(xs) > (max + THIRTY_TWO_BIT_PRECISION)):
        raise ValueError(f"np.max(xs) = {np.max(xs)} > {max}")


def rel_value(  # TODO delete this and rel residue and check_vals
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

    FILE_ID_LEN: int = 3
    TEST_STRING: str = "acc_vs_work"

    mat_name: str
    is_sparse: bool
    mat_shape: tuple[int, int]  # TODO: add this
    snd_moment_norm: float
    snd_moment_nnz: int
    max_iter: int
    tol: float
    seed: int
    num_avg: int
    n_binom_trials: list[int]
    lbls: list[str]
    notes: str | None

    test_run_iters: np.ndarray
    nnzs: np.ndarray
    baseline_work: np.ndarray
    baseline_accuracies: np.ndarray
    work: np.ndarray
    accuracies: np.ndarray

    def __init__(
        self,
        file_id: int | None = None,
        mat_name: str | None = None,
        mat_shape: tuple[int, int] | None = None,
        is_sparse: bool | None = None,
        snd_moment_norm: float | None = None,
        snd_moment_nnz: int | None = None,
        max_iter: int | None = None,
        tol: float | None = None,
        seed: int | None = None,
        num_avg: int | None = None,
        n_binom_trials: list[int] | None = None,
        lbls: list[str] | None = None,
        notes: str | None = None,
        test_run_iters: np.ndarray | None = None,
        nnzs: np.ndarray | None = None,
        baseline_work: np.ndarray | None = None,
        baseline_accuracies: np.ndarray | None = None,
        work: np.ndarray | None = None,
        accuracies: np.ndarray | None = None,
    ) -> None:
        """Initialize the data tracking object

        Args:
            test_str (str): test name (for pathing)
            file_id (int | None): The id of the given results
            mat_name (str | None): The matrix being tested
            mat_shape (tuple[int, int] | None): The shape of the matrix
            is_sparse (bool | None): Is this matrix sparse? If not assumed dense
            snd_moment_norm (float | None): 2 norm of the second moment of A
            snd_moment_nnz (int | None): Number of nonzeroes in A^TA
            max_iter (int | None): Maximum number of iterations of Power
            iteration
            tol (float | None): The tolerance of power iteration (0 ->
            machine precision)
            seed (int | None): The seed of random events
            num_avg (int | None): Number of runs to average over
            sequences, must choose to average over all or part of them.
                1 -> average over those that exist for all trials
                2 -> average over the median num trials
                3 -> average over all (the outlier longest will be weighted 100%)
            n_binom_trials (list[int] | None): The number of binomial trials
            tested
            lbls (list[str] | None): The labels for plotting
            notes (str | None): Anything to note about this test?
            test_run_iters (np.ndarray | None): The number of iterations for
            every run of power
            nnzs (np.ndarray | None): The number of nonzeros of the 2nd moment
            approximation per trial
            baseline_work (np.ndarray | None): The work done by the baseline
            baseline_accuracies (np.ndarray | None): The accuracy of the
            baseline
            work (np.ndarray | None): The work done per test run (3d matrix)
            accuracies (np.ndarray | None): The accuracy per test run (3d matrix)
        Resources:
            - Super Class Instatiation:
                https://stackoverflow.com/questions/2399307/how-to-invoke-the-super-constructor-in-python
        """
        super().__init__(self.TEST_STRING)

        if file_id:
            self.load_results(file_id)
        else:
            if (
                mat_name is None
                or mat_shape is None
                or is_sparse is None
                or snd_moment_norm is None
                or snd_moment_nnz is None
                or max_iter is None
                or tol is None
                or seed is None
                or num_avg is None
                or n_binom_trials is None
                or lbls is None
                or test_run_iters is None
                or nnzs is None
                or baseline_work is None
                or baseline_accuracies is None
                or work is None
                or accuracies is None
            ):
                raise ValueError(
                    "if file_name is None, most other fields cannot be None"
                )
            self.mat_name = mat_name
            self.mat_shape = mat_shape
            self.is_sparse = is_sparse
            self.snd_moment_norm = snd_moment_norm
            self.snd_moment_nnz = snd_moment_nnz
            self.max_iter = max_iter
            self.tol = tol
            self.seed = seed
            self.num_avg = num_avg
            self.n_binom_trials = n_binom_trials
            self.lbls = lbls
            self.notes = notes
            self.test_run_iters = test_run_iters
            self.nnzs = nnzs
            self.baseline_work = baseline_work
            self.baseline_accuracies = baseline_accuracies
            self.work = work
            self.accuracies = accuracies

        # Ensuring correct usage
        assert self.baseline_accuracies.shape == self.baseline_work.shape, (
            f"Incorrect dimensions {self.baseline_accuracies.shape} != {self.baseline_work.shape}"
        )
        n_trials = len(self.n_binom_trials)
        assert n_trials == len(self.lbls), (
            f"Incorrect dimensions {n_trials} != {len(self.lbls)}"
        )
        assert n_trials == self.test_run_iters.shape[0], (
            f"Incorrect dimensions {n_trials} != {self.test_run_iters.shape[0]}"
        )
        assert n_trials == self.nnzs.shape[0], (
            f"Incorrect dimensions {n_trials} != {self.nnzs.shape[0]}"
        )
        assert n_trials == self.work.shape[0], (
            f"Incorrect dimensions {n_trials} != {self.work.shape[0]}"
        )
        assert n_trials == self.accuracies.shape[0], (
            f"Incorrect dimensions {n_trials} != {self.accuracies.shape[0]}"
        )
        assert self.num_avg == self.test_run_iters.shape[1], (
            f"Incorrect dimensions {self.num_avg} != {self.accuracies.shape[1]}"
        )
        assert self.num_avg == self.nnzs.shape[1], (
            f"Incorrect dimensions {self.num_avg} != {self.nnzs.shape[1]}"
        )
        assert self.num_avg == self.accuracies.shape[1], (
            f"Incorrect dimensions {self.num_avg} != {self.accuracies.shape[1]}"
        )

    def load_results(self, file_id: int) -> None:
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
            self.mat_name = metadata["mat_name"]
            self.mat_shape = metadata["mat_shape"]
            self.is_sparse = metadata["is_sparse"]
            self.snd_moment_norm = metadata["snd_moment_norm"]
            self.snd_moment_nnz = metadata["snd_moment_nnz"]
            self.max_iter = metadata["max_iter"]
            self.tol = metadata["tol"]
            self.seed = metadata["seed"]
            self.num_avg = metadata["num_avg"]
            self.n_binom_trials = metadata["n_binom_trials"]
            self.lbls = metadata["lbls"]
            self.notes = metadata["notes"]

        npzfile = np.load(path_npz)

        self.test_run_iters = npzfile["test_run_iters"]
        self.nnzs = npzfile["nnzs"]
        self.baseline_work = npzfile["baseline_work"]
        self.baseline_accuracies = npzfile["baseline_accuracies"]
        self.work = npzfile["work"]
        self.accuracies = npzfile["accuracies"]

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
            "mat_name": self.mat_name,
            "mat_shape": self.mat_shape,
            "is_sparse": self.is_sparse,
            "snd_moment_norm": self.snd_moment_norm,
            "snd_moment_nnz": self.snd_moment_nnz,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "seed": self.seed,
            "num_avg": self.num_avg,
            "n_binom_trials": self.n_binom_trials,
            "lbls": self.lbls,
            "notes": self.notes,
        }
        with open(path_json, "w") as json_file:
            json.dump(metadata, json_file, indent=4)

        np.savez(
            path_npz,
            test_run_iters=self.test_run_iters,
            nnzs=self.nnzs,
            baseline_work=self.baseline_work,
            baseline_accuracies=self.baseline_accuracies,
            work=self.work,
            accuracies=self.accuracies,
        )

        return file_id

    def display_results(
        self,
        run_length_type: int,
        log_y: bool,
    ) -> None:
        """Display the results

        Args:
            log_y (bool): take logarithm of y-axis when plotting?
        """
        fig, ax = plt.subplots()

        base_xs = self.baseline_work
        base_ys = self.baseline_accuracies

        work_offset = base_xs[0]
        total_base_work = base_xs[-1]

        # Start at zero
        base_xs = base_xs - work_offset

        base_xs = rel_value(
            max=total_base_work,
            xs=base_xs,
            check_max=True,
        )
        base_ys = rel_residue(
            max=self.snd_moment_norm,
            xs=base_ys,
            check_max=True,
        )
        ax.plot(base_xs, base_ys, label=f"baseline, nnz = {self.snd_moment_nnz}")

        num_trials = self.work.shape[0]
        num_avg = self.num_avg

        for i in range(num_trials):
            # Looping through Ns
            least_iter = np.min(self.test_run_iters[i])
            most_iter = np.max(self.test_run_iters[i])
            median_iter = round(np.median(self.test_run_iters[i]))

            lbl = self.lbls[i]
            xs = self.work[i, :most_iter]
            ys_i = np.full(  # for averaging
                (num_avg, self.accuracies.shape[2]),
                np.nan,
            )
            # Scale to be between zero and one
            xs = xs - work_offset
            xs = rel_value(
                max=total_base_work,
                xs=xs,
                check_max=False,
            )

            for j in range(num_avg):
                # Looping through the runs (for averaging purposes)
                ys_temp = self.accuracies[i, j, : self.test_run_iters[i, j]]
                ys_i[j, : self.test_run_iters[i, j]] = rel_residue(
                    max=self.snd_moment_norm,
                    xs=ys_temp,
                    check_max=True,
                )

            ys = np.nanmean(ys_i, axis=0)
            nnz = round(np.mean(self.nnzs[i]))

            if run_length_type == 1:
                xs = xs[:least_iter]
                ys = ys[:least_iter]
            elif run_length_type == 2:
                xs = xs[:median_iter]
                ys = ys[:median_iter]
            else:
                xs = xs[:most_iter]
                ys = ys[:most_iter]

            ax.plot(xs, ys, label=f"{lbl}, nnz = {nnz}")

        ax.set_title(
            f"Work vs. Accuracy of Top Eigenvector ({self.mat_name}:"
            f"{self.mat_shape[0]}x{self.mat_shape[1]}, avg of {num_avg}) runs"
        )
        ax.set_xlabel("Approximate Proportion of Scalar Mults")
        ax.set_ylabel(r"$\frac{|A^TA| - |A^TA\tilde v_1|}{|A^TA|}$", rotation=0)
        if log_y:
            ax.set_yscale("log")
        ax.legend()
        fig.canvas.draw()
        plt.show()
        plt.close(fig)

    def disp_results(self) -> None:
        self.display_results(
            run_length_type=2,
            log_y=True,
        )


def main():
    """Nothing but a thang"""
    result_ids = range(1, 25)

    for id in result_ids:
        results = AccWorkResult(file_id=id)
        results.disp_results()


def main_2():
    result = AccWorkResult(file_id=25)
    result.disp_results()

    result = AccWorkResult(file_id=1)
    result.disp_results()


if __name__ == "__main__":
    """For testing purporses"""
    #    main()
    main_2()
