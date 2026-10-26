"""The Test Object

Resources:
    Generics with upper bounds:
        https://typing.python.org/en/latest/spec/generics.html
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from .MatGetter import MatGetter
from .util.ResultType import ResultType

MGT = TypeVar("MGT", bound=MatGetter)


class Test(ABC, Generic[MGT]):
    """For doing some tests"""

    result_type: ResultType
    mat_getter: MGT

    def __init__(
        self,
        result_type: ResultType,
        mat_getter: MGT,
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
