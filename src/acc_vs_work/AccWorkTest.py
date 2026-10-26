
from src.Test import Test
from src.acc_vs_work.SSGetter import SSGetter
from src.util.ResultType import ResultType


class AccWorkTest(Test):
    """For doing some tests"""


    def __init__(
        self,
        result_type: ResultType,
        mat_getter: SSGetter,
        log_y:bool,
        run_length_type:int,
        num_avg:int,
        mats:list[str],
        on_easley:bool,
        max_iter:int,
        tol:float,
        seed:int,
        Ns:list[int],
    ) -> None:
        """Create a Result object

        Args:
            result_type (ResultType): The type of result for this test
            mat_getter (SSGetter): For generating matrices for this test
            log_y (bool): Y-axis of plot log scaled?
            run_length_type (int): When averaging variable length sequences, must choose
            to average over all or part of them.
                1 -> average over those that exist for all trials
                2 -> average over the median num trials
                3 -> average over all (the outlier longest will be weighted 100%)
            num_avg (int): The number averaged in the result (to reduce noise) # TODO
            mats (list[str]): The matrices to test
            on_easley (bool): on_easley ->  don't display, not on_easly -> display!
            max_iter (int): max # of iterations of power
            tol (float): Stopping tolerance of power iteration
            seed (int): for repeated randomness
            Ns (list[int]): Number of bernoulli trials
        """
        assert result_type is ResultType.ACC_VS_WORK, f"result_type not ACC_VS_WORK: {result_type}"
        # Call to Parent's __init__()
        super().__init__(
            result_type=result_type,
            mat_getter=mat_getter,
        )
        self.mat_getter = mat_getter
        # TODO: this guy

    def run(self) -> None:
        """Run the test!"""
        # TODO: this guy
