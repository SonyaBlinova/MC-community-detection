from plotting import *
from helpers import *
from time import time
import seaborn as sns

def main():
    sns.set_theme()

    N = 100
    num_iter = 4000
    a = 6.9
    b = 0.1
    n0 = 6
    num_exp = 1

    start_time = time()

    """compare convergence of different algos:"""

    # hamiltonians_1, hamiltonians_2, hamiltonians_3, overlaps_1, overlaps_2, overlaps_3, times = plot_time_average_overlap_algorithms_compare(N, a, b, num_iter, n0, num_exp)
    # plot_algorithms_compare_helper(hamiltonians_1, hamiltonians_2, hamiltonians_3, overlaps_1, overlaps_2, overlaps_3, times, N, a, b, n0)

    """plot dependece of the conv time with different r and d"""

    # a_1 = 6.9
    # b_1 = 0.1
    # a_2 = 16.9
    # b_2 = 1.1
    # a_3 = 26.9
    # b_3 = 10.1
    # plot_time_overlap_metropolis_compare(N, a_1, b_1, a_2, b_2, a_3, b_3, num_iter)
    # plot_time_overlap_houdayer_compare(N, a_1, b_1, a_2, b_2, a_3, b_3, num_iter)

    """dependence on b/a ration"""

    # plot_ratio_overlap_metropolis(N, num_iter)
    # plot_ratio_overlap_houdayer(N, num_iter, n0 = 1)
    # plot_ratio_overlap_houdayer(N, num_iter, n0 = 10)

    """n0 and b/a"""

    # plot_ratio_n0_houdayer(N, num_iter)

    """dependence for mixed algo over n0"""

    # n0_1 = 2
    # n0_2 = 10
    # n0_3 = 100
    # plot_time_overlap_mixed_compare(N, a, b, num_iter, n0_1, n0_2, n0_3)

    end_time = time()
    print('Excecution time:', end_time - start_time)

if __name__ == "__main__":
    main()