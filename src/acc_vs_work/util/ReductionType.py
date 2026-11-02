"""The Reduction strategies implemented

Resources:
    - String Enums:
        https://docs.python.org/3/library/enum.html#enum.StrEnum
        https://stackoverflow.com/questions/58608361/string-based-enum-in-python
"""

from enum import StrEnum, auto


# class syntax
class ReductionType(StrEnum):
    SND_MOM_NO_REDUCTION = auto()
    SND_MOMENT_NAIVE = auto()
    SND_MOMENT_BINOMIAL = auto()
    NO_REDUCTION = auto()
    AVG_BEFORE = auto()
    AVG_AFTER = auto()

    @classmethod
    def _missing_(cls, value):
        value = str(value)
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        return None
