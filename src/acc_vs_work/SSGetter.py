"""
A file for interfacing with the suite sparse matrix collection

Resources:
    https://nbviewer.org/github/drdarshan/ssgetpy/blob/master/demo.ipynb
"""

import json
import os
from pathlib import Path
import tarfile
from queue import Queue
from typing import Any
from warnings import warn

import numpy as np
import scipy
import ssgetpy

from src.util.constants import UTF_8_STR

from ..MatGetter import MatGetter
from ..util.SparseFormat import SparseFormat


class SSGetter(MatGetter):
    """
    A wrapper for the ssgetpy library
    """

    TOP_MAG_PATH = Path('data/top_row_mags.json')
    format: SparseFormat
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
        format: SparseFormat,
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
            format (SparseFormat): The sparse format (COO, CSR, or CSC)
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
            warn(
                f"Matrix {ss_matrix.name} is not in .tar.gz format."
            )  # TODO: this is how to get the name
        return None

    def check_mat(
        self,
        matrix: ssgetpy.matrix.Matrix,
    ) -> None:
        """Checks the metadata requested matches the matrix given

        Args:
            matrix (scipy.sparse.sparray): The matrix to check
        """

        if matrix.nnz < self.nzbounds[0] or matrix.nnz > self.nzbounds[1]:
            warn(f"matrix.nnz={matrix.nnz}, out of bounds: {self.nzbounds}")
        if (self.rowbounds is not None):
            if (
                matrix.rows < self.rowbounds[0]  or
                matrix.rows > self.rowbounds[1]
            ):
                warn(f"matrix.shape={matrix.shape}, out of bounds: {self.rowbounds}")
        if (self.colbounds is not None):
            if (
                matrix.cols < self.colbounds[0]  or
                matrix.cols > self.colbounds[1]
            ):
                warn(f"matrix.shape={matrix.shape}, out of bounds: {self.colbounds}")
        if self.isspd is not None:
            if (self.isspd != matrix.isspd):
                warn(f"Requested {self.isspd}, got {matrix.isspd}")
        if self.dtype is not None:
            if (self.dtype != matrix.dtype):
                warn(f"Requested {self.dtype}, got {matrix.dtype}")
        if self.group is not None:
            if (self.group != matrix.group):
                warn(f"Requested {self.group}, got {matrix.group}")
        if self.kind is not None:
            if (self.kind != matrix.kind):
                warn(f"Requested {self.kind}, got {matrix.kind}")

    def get_ss_mat(
        self,
        mat_name: str | None,
    ) -> ssgetpy.matrix.Matrix | None:
        """Get the suite sparse matrix

        Args:
            mat_name (str | None): The name of the matrix in question

        Returns:
             ss_matrix (ssgetpy.matrix.Matrix): The Matrix object
        """
        ss_matrix: scipy.sparse.sparray | None = None
        if mat_name is None:
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

            # Have stuff in queue, get it
            ss_matrix = self.mat_queue.get(block=False)
        else:
            # Searching by name
            ss_matrix = ssgetpy.search(mat_name)[0]

        return ss_matrix

    def ssget(
        self,
        mat_name: str | None,
    ) -> tuple[str, scipy.sparse.sparray] | None:
        """Get a matrix based on its key word arguments

        Args:
            kwargs (dict): The key word arguments for generating the matrix

        Returns:
            mat_name (str): The matrix name
            mat (scipy.sparse.sparray): The matrix
            gotten
        """
        ss_matrix = self.get_ss_mat(mat_name=mat_name)
        if ss_matrix is None:
            return None

        matrix = self.load_mat(ss_matrix=ss_matrix)
        if matrix is None:
            return None

        match self.format:
            case SparseFormat.COO:
                matrix = matrix.tocoo()
            case SparseFormat.CSC:
                matrix = matrix.tocsc()
            case SparseFormat.CSR:
                matrix = matrix.tocsr()
            case _:
                warn(f"Unexpected sparse format: {self.format}")
                return None

        self.check_mat(ss_matrix)
        return ss_matrix.name, matrix

    def get(
        self,
        kwargs: dict[str, Any] | None = None,
    ) -> scipy.sparse.sparray | np.ndarray | None:
        """Get a matrix based on its key word arguments

        Args:
            kwargs (dict): The key word arguments for generating the matrix

        Returns:
            mat (scipy.sparse.sparray | np.ndarray | None): The matrix
            gotten
        """
        mat_name = None
        if kwargs:
            mat_name = kwargs.get("mat_name")

        ss_result = self.ssget(mat_name=mat_name)

        if ss_result is None:
            return None

        return ss_result[1]

    def rescale_matrix(
        self,
        matrix:scipy.sparse.sparray,
        max_row_mag: float,
    ) -> scipy.sparse.sparray:
        matrix = matrix / max_row_mag

        matrix.eliminate_zeros()

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
            scale_factor (float): the scaling factor
        """
        if type(matrix) is not scipy.sparse.sparray:
            raise TypeError(f"Expecting sparse array, not {type(matrix)}")
        if matrix.data.dtype not in {np.float32, np.float64}:
            if matrix.data.dtype not in {np.int32, np.int64}:
                raise TypeError(f"Unexpected Type: {matrix.data.dtype}")
            matrix.data = matrix.data.astype(float)

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Given non-square matrix: A.shape = {matrix.shape}")

        row_mags = np.asarray(scipy.sparse.norm(matrix, ord=ord, axis=1))
        top_row_mag = max(row_mags)

        # Rescale A
        matrix = self.rescale_matrix(
            matrix=matrix,
            max_row_mag=top_row_mag,
        )

        return matrix, top_row_mag

    def get_preprocessed(
        self,
        mat_name: str | None,
    ) -> scipy.sparse.sparray | None:
        """Get A scaled by a factor of 1/max_i(||a_i||)

        Args:
            mat_name (str): the name of the suite-spasrse matrix

        Returns:
            matrix (scipy.sparse.sparray): the matrix in sparse format

        Resources:
            Json encoding API:
                https://stackoverflow.com/questions/46408051/python-json-load-set-encoding-to-utf-8
        """
        ss_result = self.ssget(mat_name=mat_name)
        if ss_result is None:
            return None
        mat_name, matrix = ss_result

        top_mag_dict = None
        with open(self.TOP_MAG_PATH, 'r', encoding=UTF_8_STR) as json_file:
            top_mag_dict = json.load(json_file)

        top_mag = top_mag_dict.get(mat_name)

        if top_mag is None:
            # Not yet calculated -> calculate + save
            matrix, top_mag = self.preprocess(matrix)
            top_mag_dict |= {mat_name: top_mag}
            with open(self.TOP_MAG_PATH, 'w', encoding=UTF_8_STR) as json_file:
                json.dump(top_mag_dict, json_file, indent=2)
        else:
            # Top row magnitude stored, do necessary calculations
            matrix = self.rescale_matrix(
                matrix=matrix,
                max_row_mag=top_mag
            )

        return matrix


def main() -> None:
    ss_getter = SSGetter(
        preprocess_data=True,
        format=SparseFormat.CSR,
        nzbounds=(1000, 100000),
    )
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
    for mat in mats:
        matrix = ss_getter.get_preprocessed('494_bus')

if __name__ == "__main__":
    """
    Testing ssgetpy bug
    """
    main()
