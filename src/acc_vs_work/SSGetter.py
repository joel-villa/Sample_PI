"""
A file for interfacing with the suite sparse matrix collection

Resources:
    https://nbviewer.org/github/drdarshan/ssgetpy/blob/master/demo.ipynb
"""

import os
import tarfile
from queue import Queue
from typing import Any
from warnings import warn

import numpy as np
import scipy
import ssgetpy

from ..MatGetter import MatGetter
from ..util import SparseFormat


class SSGetter(MatGetter):
    """
    A wrapper for the ssgetpy library
    """

    format: SparseFormat.SparseFormat

    nzbounds: tuple[int, int]
    limit: int | None
    rowbounds: tuple[int, int] | None
    colbounds: tuple[int, int] | None
    isspd: bool | None
    dtype: str | None
    group: str | None
    kind: str | None

    mat_queue: Queue

    def __init__(
        self,
        preprocess_data: bool,
        format: SparseFormat.SparseFormat,
        nzbounds: tuple[int, int],
        limit: int | None = None,
        rowbounds: tuple[int, int] | None = None,
        colbounds: tuple[int, int] | None = None,
        isspd: bool | None = None,
        dtype: str | None = None,
        group: str | None = None,
        kind: str | None = None,
    ) -> None:
        """The initializer for the SSGetter

        Args:
            format (SparseFormat.SparseFormat): The sparse format (COO, CSR, or CSC)
            nzbounds (tuple[int, int]): The bounds on number of nonzeroes used,
            limit (int | None): The maximum number of matrices to load at once
            rowbounds (tuple[int, int] | None): The bounds on the row count
            colbounds (tuple[int, int] | None): The bounds on the column count
            isspd (bool | None): Semi-positive definite?
            dtype (str | None): Data type of the matrix
            group (str | None): The group of matrices
            kind (str | None): Kind of matrix
        """
        # Call to Parent's __init__()
        super().__init__(preprocess_data)

        if dtype not in {"real", "complex", "binary"}:
            warn(f"Likely an erroneous dtype: {dtype}")

        self.format = format
        self.limit = limit
        self.nzbounds = nzbounds
        self.rowbounds = rowbounds
        self.colbounds = colbounds
        self.isspd = isspd
        self.dtype = dtype
        self.dtype = dtype
        self.group = group
        self.kind = kind

        self.mat_queue = Queue()

    def load_mat(
        self,
        ss_matrix: ssgetpy.matrix.Matrix,
    ) -> scipy.sparse.sparray | None:

        ss_matrix.download()  # Download the matrix

        # Get the local path of the .tar.gz file
        path_tuple = ss_matrix.localpath()  # This returns a tuple (two identical paths)
        tar_path = path_tuple[0]  # Use the first path (both are the same)

        # Extract the .tar.gz file to get the matrix file inside
        if tar_path.endswith(".tar.gz"):
            with tarfile.open(tar_path, "r:gz") as tar:
                # Extract files in the same folder as the .tar.gz file
                extract_dir = os.path.dirname(tar_path)
                tar.extractall(path=extract_dir)  # Extract to the same folder

            # After extraction, find the matrix file (.mtx) in the extracted files
            matrix_file = None
            extract_dir = os.path.join(extract_dir, ss_matrix.name)
            for extracted_file in os.listdir(extract_dir):
                if extracted_file.endswith(".mtx"):
                    matrix_file = extracted_file
                    break

            if matrix_file is None:
                raise FileNotFoundError(f"No .mtx file found in {extract_dir}")

            # Full path to the extracted matrix file
            extracted_path = os.path.join(extract_dir, matrix_file)

            # Now, read the extracted matrix file using mmread and return it in coo
            return scipy.io.mmread(extracted_path)
        else:
            warn(f"Matrix {ss_matrix.name} is not in .tar.gz format.")
        return None

    def check_mat(
        self,
        matrix: scipy.sparse.sparray,
    ) -> None:
        """Checks the metadata requested matches the matrix given

        Args:
            matrix (scipy.sparse.sparray): The matrix to check
        """
        if matrix.nnz < self.nzbounds[0] | matrix.nnz > self.nzbounds[1]:
            warn(f"matrix.nnz={matrix.nnz}, out of bounds: {self.nzbounds}")
        # TODO: rest of error checking
        raise NotImplementedError

    def get(
        self,
        kwargs: dict[str, Any] | None = None,
    ) -> scipy.sparse.sparray | np.ndarray | None:
        """Get a matrix based on its key word arguments

        Args:
            kwargs (dict): The key word arguments for generating the matrix

        Returns:
             mat ([scipy.sparse.sparray | np.ndarray, dict] | None): The matrix
             gotten
        """
        matrix: scipy.sparse.sparray | None = None
        if kwargs is None:
            if self.mat_queue.empty():
                # empty queue -> get stuff using ssgetpy
                ss_mats = ssgetpy.search(
                    rowbounds=self.rowbounds,
                    colbounds=self.colbounds,
                    nzbounds=self.nzbounds,
                    isspd=self.isspd,
                    dtyp=self.dtype,
                    group=self.group,
                    kind=self.kind,
                    limit=self.limit,
                )

                for ss_mat in ss_mats:
                    self.mat_queue.put(
                        item=ss_mat,
                        block=False,
                    )
                ss_matrix = self.mat_queue.get(block=False)
                matrix = self.load_mat(ss_matrix=ss_matrix)
            else:
                # Have stuff in queue, get it
                ss_matrix = self.mat_queue.get(block=False)
                matrix = self.load_mat(ss_matrix=ss_matrix)
        else:
            # Searching by name
            mat_name = kwargs.get("name")
            if mat_name:
                ss_mat = ssgetpy.search(mat_name)[0]
                matrix = self.load_mat(ss_mat)

        if matrix is None:
            return None

        match self.format:
            case SparseFormat.SparseFormat.COO:
                matrix = matrix.tocoo()
            case SparseFormat.SparseFormat.CSC:
                matrix = matrix.tocsc()
            case SparseFormat.SparseFormat.CSR:
                matrix = matrix.tocsr()
            case _:
                warn(f"Unexpected sparse format: {self.format}")
                return None

        self.check_mat(matrix)

        return matrix
    def preprocess(
        self,
        matrix: scipy.sparse.sparray | np.ndarray,
    ) -> tuple[scipy.sparse.sparray | np.ndarray, float]:
        """Get A scaled by a factor of 1/max_i(||a_i||)

        Args:
            mat_name (str): the name of the suite-spasrse matrix

        Returns:
            matrix (scipy.sparse): the matrix in sparse format
            max_row_magnitude (float): the scaling factor
        """
        if type(matrix) not in SparseFormat.TYPES:
            raise TypeError(f"Incorrect type of matrix: {type(matrix)}")

        elif (matrix.data.dtype not in {np.float32, np.float64}):
            if (matrix.data.dtype not in {np.int32, np.int64}):
                raise TypeError(f"Unexpected Type: {matrix.data.dtype}")
            matrix.data = matrix.data.astype(float)

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Given non-square matrix: A.shape = {matrix.shape}")

        max_row_magnitude = get_sorted_row_norms(mat_name, rescale=False)[0] #TODO

        # Rescale A
        matrix = matrix / max_row_magnitude

        matrix.eliminate_zeros()

        return matrix, max_row_magnitude


def main() -> None:
    ss_getter = SSGetter(
        preprocess_data=True,
        format=SparseFormat.SparseFormat.CSR,
        nzbounds=(1000, 100000)
    )
    sparse_mat = ss_getter.get({"name": "494_bus"})
    print(sparse_mat)


if __name__ == "__main__":
    """
    Testing ssgetpy bug
    """
    main()
