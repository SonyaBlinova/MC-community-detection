from plotting import *
from helpers import *
from time import time
import seaborn as sns
import numpy as np
import networkx as nx

def txt_to_npy():
    print(np.load('output.npy'))

def main():
    # enter filename here
    matrix = np.load('A_test.npy')
    N = matrix.shape[0]
    # enter a and b here
    a = 5.9
    b = 0.1
    graph = nx.from_numpy_matrix(matrix)
    h = compute_h(graph, a, b, N)
    initial_state = np.random.choice([-1, 1], N)
    graph_1 = graph.copy()
    nx.set_node_attributes(graph_1, dict(zip(range(N), initial_state)), 'cl')
    graph_2 = graph.copy()
    nx.set_node_attributes(graph_2, dict(zip(range(N), initial_state)), 'cl')
    i = 0
    while True:
        i += 1
        graph_1 = metropolis_step(graph_1, h, N, 1/N, 1/N)
        graph_2 = metropolis_step(graph_2, h, N, 1/N, 1/N)
        if i % 5 == 0:
            graph_1, graph_2 = houdayer_step(graph_1, graph_2)
        if i % 10 == 0:
            output = list(nx.get_node_attributes(graph_1, 'cl').values())
            with open('output.npy', 'wb') as f:
                np.save(f, output)

if __name__ == "__main__":
    main()