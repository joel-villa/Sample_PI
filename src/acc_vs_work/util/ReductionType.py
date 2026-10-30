"""The Reduction strategies implemented"""

from enum import Enum

# class syntax
class ReductionType(Enum):
    SND_MOM_NO_REDUCTION = 1
    SND_MOMENT_NAIVE = 2
    SND_MOMENT_BINOMIAL = 3
