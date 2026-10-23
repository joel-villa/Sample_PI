"""A superclass which defines the API needed to be a 'Result'
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from warnings import warn


class Result(ABC):
    """For tracking the results of a test"""

    path: Path

    def __init__(
        self,
        test_str: str,
    ) -> None:
        """Create a Result object

        Args:
            test_str (str): The string representation of this test type
        """
        self.path = Path(f"data/{test_str}")

    def get_paths(self, file_id: int) -> tuple[Path,Path]:
        """Get the paths to the file name in test_results directory

        Args:
            self: This LatencyMetadata instance
            file_name (int): The name of the file of significance

        Returns:
            path_json (Path): The path to the json file
            path_npz (Path): The path to the npz file
        """
        path_json = self.path / f"{file_id}.json"
        path_npz = self.path / f"{file_id}.npz"
        return path_json, path_npz

    def get_next_file_id(self) -> int:
        """Get the current maximum filename under the results path

        Returns:
             maximum (int): The maximum file number currently
        """
        curr_max = 0
        for (root, dirs, files) in os.walk(self.path):
            for file in files:
                if file.endswith(".json"):
                    try:
                        curr_file = int(file[:-5])
                        curr_max = max(curr_max, curr_file)
                    except ValueError as e:
                        warn(f"{e}: {file} in {self.path} is not of correct format")

        return curr_max + 1

    @abstractmethod
    def load_results(self, file_id:int) -> None:
        """Load the results from the provided file name

        Args:
            file_name (int): The file of significance
        """

    @abstractmethod
    def save_results(self) -> int:
        """Save results to npz file and metadata to a json file

        Returns:
            file_id (int): The file id associated with this test
        """

    @abstractmethod
    def disp_results(
        self,
    ) -> None:
        """Display the results

        Args:
        """
