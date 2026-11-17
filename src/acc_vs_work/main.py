"""Ooooh tests so gooood"""

from src.acc_vs_work.AccWorkResult import AccWorkResult
from src.acc_vs_work.AccWorkTest import AccWorkTest
from src.acc_vs_work.NnzCalculator import NnzCalculator
from src.acc_vs_work.SSGetter import SSGetter
from src.acc_vs_work.util.ReductionType import ReductionType
from src.util.ResultType import ResultType
from src.util.SparseFormat import SparseFormat


def main():
    """Run that dang thang"""
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

    mats = sorted(mats)  # Alphabetical order
    tol = 0  # tolerance of zero -> machine percision
    max_iter = 2**17  # 131, 072
    num_avg = 32
    seed = 7

    n_trials = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
    ]

    ss_getter = SSGetter(
        format=SparseFormat.CSR,
        nzbounds=(1000, 100000),
    )

    test = AccWorkTest(
        result_type=ResultType.ACC_VS_WORK,
        mat_getter=ss_getter,
        mats=mats,
        num_trials=n_trials,
        tol=tol,
        num_avg=num_avg,
        max_iter=max_iter,
        seed=seed,
    )

    test.run_test(
        disp_result=False,
        log_y=True,
        run_length_type=2,
    )


    result_ids = range(1, len(mats) + 1)

    nnz_calculator = NnzCalculator()

    for id in result_ids:
        results = AccWorkResult(file_id=id)
        nnz_calculator.update_expected_nnzs(results)
        print(f"({id}) results.expected_nnz: {results.expected_nnzs}")

if __name__ == "__main__":
    """For testing purporses"""
    main()
