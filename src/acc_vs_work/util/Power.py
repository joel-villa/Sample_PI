"""Power iteration on a matrix"""

import scipy
import numpy as np

from src.util.constants import THIRTY_TWO_BIT_PRECISION


class Power:
    """Ain't one man should have all that power"""
    matrix: scipy.sparse.sparray

    def __init__(self, matrix: scipy.sparse.sparray) -> None:
        """Initialize power iteration object

        Args:
            matrix (scipy.sparse.sparray): The matrix to do power iteration on
        """
        self.matrix = matrix

    def __rayleigh_quotient(
            self,
            x: np.ndarray,
            A: scipy.sparse.sparray,
    ) -> float:
        """Get the rayleigh quotient (score) of the given array-matrix combo

        Args:
            x (np.ndarray): array in question
            A (scipy.sparse.sparray): matrix in question

        Returns:
            ray_quot (float): The Rayleigh quotient
        """
        assert np.linalg.norm(x, ord=2) < 1 + THIRTY_TWO_BIT_PRECISION, f"vect not normalized: {np.linalg.norm(x, ord=2)}"
        assert np.linalg.norm(x, ord=2) > 1 - THIRTY_TWO_BIT_PRECISION, f"vect not normalized: {np.linalg.norm(x, ord=2)}"

        return (x.T @ (A @ x)).item()

    def test(
        self,
            v0: np.ndarray,
            max_iter: int,
            tol: float,
            goal_mat: scipy.sparse.sparray | None = None,
    ) -> np.ndarray:
        """A generic test of the accuracy of tilde_A_sq, given A_sq and other
        informaiton for power iteration

        Args:
            A_sq (scipy.sparse.sparray): A^TA
            tilde_A_sq (scipy.sparse.sparray): Approximation of
            v0 (np.ndarray): initial guess for top eigenvector
            max_iter (int): max number of iterations
            tol (float): tolerance of power iteration
            init_work (int): to be added to the work calculations

        Returns:
            np.ndarray: score of top eigenvector approxation per iteration of power
        """

        if goal_mat is None:
            goal_mat =  self.matrix

        scores = np.zeros(shape=(max_iter,))
        scores[0] = self.__rayleigh_quotient(x=v0, A=goal_mat)
        i = 1
        v=v0

        while i < max_iter:
            # Compute Ax
            v = self.matrix @ v

            # Normalize to get next eigenvector approximation
            v = v / np.linalg.norm(v)

            scores[i] = self.__rayleigh_quotient( # Score of approximation based on actual
                A=goal_mat,
                x=v,
            )
            if (abs(scores[i - 1] - scores[i]) / abs(scores[i - 1]) <= tol) or (scores[i-1] > scores[0]):
                i += 1
                break
            i += 1

        scores = scores[0:i]

        return scores
