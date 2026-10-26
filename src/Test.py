"""The Test Object"""

from abc import ABC, abstractmethod

from .MatGetter import MatGetter
from .util.ResultType import ResultType


class Test(ABC):
    """For doing some tests"""

    result_type: ResultType
    mat_getter: MatGetter

    def __init__(
        self,
        result_type: ResultType,
        mat_getter: MatGetter,
    ) -> None:
        """Create a Result object

        Args:
            result_type (ResultType): The type of result for this test
            mat_getter (MatGetter): For generating matrices for this test
        """
        self.result_type = result_type
        self.mat_getter = mat_getter

    @abstractmethod
    def run(self) -> None:
        """Run the test!"""
