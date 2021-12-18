import numpy as np
import networkx as nx
from tqdm import tqdm

"""Numpy part"""

def generate_graph_np(N, x_star, a, b):
  """
  Numpy version.
  Observation graph generation. 
  """
  graph = np.zeros((N, N))
  for i in range(N):
    for j in range(i+1, N):
      if x_star[i]*x_star[j] == 1:
        if np.random.uniform(0,N) < a:
          graph[i][j] = 1
          graph[j][i] = 1
      elif x_star[i]*x_star[j] == -1:
        if np.random.uniform(0,N) < b:
          graph[i][j] = 1
          graph[j][i] = 1
  return graph

def compute_h_np(graph, a, b, N):
  """
  Numpy version.
  Compute h for the acceptance probabilities.
  """
  ln1 = np.log(a/b)
  ln2 = np.log((1 - a/N)/(1 - b/N))
  h = np.zeros_like(graph)
  h = 1/2*(graph * ln1 + (1 - graph) * ln2)
  return h

def compute_accep_prob_np(x, h, N, psi_ij, psi_ji, i):
  """
  Numpy version.
  Compute acceptance probability.
  """
  expo = 0
  for j in range(N):
    if j != i:
      expo += h[i,j] * x[j]
  expo = -2 * x[i] * expo
  a = min(1, np.exp(expo) * (psi_ji/psi_ij))
  return a

def hamiltonian_np(x, h, N):
  """
  Numpy version.
  Hamiltonian.
  """
  p = 0
  for i in range(N):
    for j in range(i+1, N):
      p -= h[i, j] * x[i] * x[j]
  return p

def overlap_np(x_star, x_final, N):
  """
  Numpy version.
  Overlap.
  """
  over = 0
  for i in range(N):
    over += x_star[i]*x_final[i]
  return (1/N)*np.abs(over)

def metropolis_step_np(curr_state, h, N, psi_ij, psi_ji):
  """
  Numpy version.
  Metropolis step. It is used for the running metropolis algotith.
  """
  rand_ind = np.random.choice(N, 1)[0]
  a = compute_accep_prob_np(curr_state, h, N, psi_ij, psi_ji, rand_ind)
  rand_num = np.random.uniform(0,1)
  if rand_num <= a:
    curr_state[rand_ind] *= -1
  return curr_state

def metropolis_run(curr_state, N, h, x_star, num_iter):
  """
  Numpy version.
  Metroppolis algorithm.
  """
  for i in tqdm(range(num_iter)):
    curr_state = metropolis_step_np(curr_state, h, N, 1/N, 1/N)
  return overlap_np(x_star, curr_state, N)

"""Networkx part"""

def generate_graph(N, x_star, a, b):
  """
  Networkx version.
  Observation graph deneration.
  """
  graph = nx.Graph()
  for node in range(N):
    graph.add_node(node, cl=x_star[node])
  for i in range(N):
    for j in range(i+1, N):
      if x_star[i]*x_star[j] == 1:
        if np.random.uniform(0,N) < a:
          graph.add_edge(i,j)
      elif x_star[i]*x_star[j] == -1:
        if np.random.uniform(0,N) < b:
          graph.add_edge(i,j)
  return graph

def compute_h(graph, a, b, N):
  """
  Networkx version.
  Compute h for the acceptance probabilities.

  """
  graph = nx.to_numpy_matrix(graph)
  ln1 = np.log(a/b)
  ln2 = np.log((1 - a/N)/(1 - b/N))
  h = np.zeros_like(graph)
  h = 1/2*(graph * ln1 + (1 - graph) * ln2)
  return h

def compute_accep_prob(graph, h, N, psi_ij, psi_ji, i):
  """
  Networkx version.
  Compute acceptance probability.
  """
  expo = 0
  for j in range(N):
    if j != i:
      expo += h[i,j] * graph.nodes[j]['cl']
  expo = -2 * graph.nodes[i]['cl'] * expo
  a = min(1, np.exp(expo) * (psi_ji/psi_ij))
  return a

def hamiltonian(graph, h, N):
  """
  Networkx version.
  Hamiltonian.
  """
  p = 0
  for i in range(N):
    for j in range(i+1, N):
      p -= h[i, j] * graph.nodes[i]['cl'] * graph.nodes[j]['cl']
  return p

def overlap(x_star, graph, N):
  """
  Networkx version.
  Overlap.
  """
  over = 0
  for i in range(N):
    over += x_star[i]*graph.nodes[i]['cl']
  return (1/N)*np.abs(over)

def metropolis_step(graph, h, N, psi_1, psi_2):
  """
  Networkx version.
  Metropolis step. It is used for the Houdayer algorithms.
  """
  rand_ind = np.random.choice(N, 1)[0]
  a = compute_accep_prob(graph, h, N, psi_1, psi_2, rand_ind)
  rand_num = np.random.uniform(0,1)
  if rand_num <= a:
    graph.nodes[rand_ind]['cl'] = -1 * graph.nodes[rand_ind]['cl']
  return graph

def houdayer_step(G1, G2):
  """
  Networkx version.
  Houdayer step.
  """
  # create subgraph
  subgraph_nodes = [i[0] for i in set(nx.get_node_attributes(G1, "cl").items()) - set(nx.get_node_attributes(G2, "cl").items())]
  R = G1.subgraph(subgraph_nodes)

  if len(list(R.nodes)) != 0:
    # choosing randomly element
    r_st = np.random.choice(list(R.nodes))
    G1.nodes[r_st]['cl'] *= -1
    G2.nodes[r_st]['cl'] *= -1
    # finding all connections of the chosen element
    for i in R.neighbors(r_st):
      G1.nodes[i]['cl'] *= -1
      G2.nodes[i]['cl'] *= -1
  return G1, G2

def houdayer_run(graph_curr_1, graph_curr_2, N, h, x_star, num_iter, n0 = 1):
  """
  Networkx version.
  Houdayer and Mixed algorithms.
  """
  for i in tqdm(range(num_iter)):
    graph_curr_1 = metropolis_step(graph_curr_1, h, N, 1/N, 1/N)
    graph_curr_2 = metropolis_step(graph_curr_2, h, N, 1/N, 1/N)
    if i % n0 == 0:
      graph_curr_1, graph_curr_2 = houdayer_step(graph_curr_1, graph_curr_2)
  return overlap(x_star, graph_curr_1, N), overlap(x_star, graph_curr_2, N)