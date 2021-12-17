from plotting import *
from helpers import *
from time import time

def main():
    N = 500
    num_iter = 5000
    a = 6.9
    b = 0.1
    n0 = 6
    num_exp = 100

    start_time = time()
    hamiltonians_1, hamiltonians_2, hamiltonians_3, overlaps_1, overlaps_2, overlaps_3, times = plot_time_average_overlap_algorithms_compare(N, a, b, num_iter, n0, num_exp)
    plot_algorithms_compare_helper(hamiltonians_1, hamiltonians_2, hamiltonians_3, overlaps_1, overlaps_2, overlaps_3, times, N, a, b, n0)

    end_time = time()
    print('Excecution time:', end_time - start_time)

if __name__ == "__main__":
    main()