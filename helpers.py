import numpy as np
import networkx as nx
from tqdm import tqdm

"""Numpy part"""

def generate_graph_np(N, x_star, a, b):
  """
  Numpy version for original base chain.
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
  Numpy version for original base chain.
  Compute h for the acceptance probabilities.
  """
  ln1 = np.log(a/b)
  ln2 = np.log((1 - a/N)/(1 - b/N))
  h = np.zeros_like(graph)
  h = 1/2*(graph * ln1 + (1 - graph) * ln2)
  return h

def compute_accep_prob_np(x, h, N, psi_ij, psi_ji, i):
  """
  Numpy version for original base chain.
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
  Numpy version for original base chain.
  Hamiltonian.
  """
  p = 0
  for i in range(N):
    for j in range(i+1, N):
      p -= h[i, j] * x[i] * x[j]
  return p

def overlap_np(x_star, x_final, N):
  """
  Numpy version for original base chain.
  Overlap.
  """
  over = 0
  for i in range(N):
    over += x_star[i]*x_final[i]
  return (1/N)*np.abs(over)

def metropolis_step_np(curr_state, h, N, psi_ij, psi_ji):
  """
  Numpy version for original base chain.
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
  Numpy version for original base chain.
  Metroppolis algorithm.
  """
  for i in tqdm(range(num_iter)):
    curr_state = metropolis_step_np(curr_state, h, N, 1/N, 1/N)
  return overlap_np(x_star, curr_state, N)

"""Networkx part for original base chain."""

def generate_graph(N, x_star, a, b):
  """
  Networkx version for original base chain.
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

def compute_accep_prob(graph, h, N, psi_ij, psi_ji, i):
  """
  Networkx version for original base chain.
  Compute acceptance probability.
  """
  expo = 0
  for j in range(N):
    if j != i:
      expo += h[i,j] * graph.nodes[j]['cl']
  expo = -2 * graph.nodes[i]['cl'] * expo
  a = min(1, np.exp(expo) * (psi_ji/psi_ij))
  return a

def compute_h(graph, a, b, N):
  """
  Networkx version for original base chain.
  Compute h for the acceptance probabilities.

  """
  graph = nx.to_numpy_matrix(graph)
  ln1 = np.log(a/b)
  ln2 = np.log((1 - a/N)/(1 - b/N))
  h = np.zeros_like(graph)
  h = 1/2*(graph * ln1 + (1 - graph) * ln2)
  return h

def hamiltonian(graph, h, N):
  """
  Networkx version for original base chain.
  Hamiltonian.
  """
  p = 0
  for i in range(N):
    for j in range(i+1, N):
      p -= h[i, j] * graph.nodes[i]['cl'] * graph.nodes[j]['cl']
  return p

def overlap(x_star, graph, N):
  """
  Networkx version for original base chain.
  Overlap.
  """
  over = 0
  for i in range(N):
    over += x_star[i]*graph.nodes[i]['cl']
  return (1/N)*np.abs(over)

def metropolis_step(graph, h, N, psi_1, psi_2):
  """
  Networkx version for original base chain.
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
  Networkx version for original base chain.
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
  Networkx version for original base chain.
  Houdayer and Mixed algorithms.
  """
  for i in tqdm(range(num_iter)):
    graph_curr_1 = metropolis_step(graph_curr_1, h, N, 1/N, 1/N)
    graph_curr_2 = metropolis_step(graph_curr_2, h, N, 1/N, 1/N)
    if i % n0 == 0:
      graph_curr_1, graph_curr_2 = houdayer_step(graph_curr_1, graph_curr_2)
  return overlap(x_star, graph_curr_1, N), overlap(x_star, graph_curr_2, N)

"""Networkx part for binary base chain."""

def compute_accep_prob_binary(curr_graph, next_graph, h, N):
  """
  Networkx version for binary base chain.
  Compute acceptance probability.
  """  
  #print(nx.get_node_attributes(curr_graph, 'cl'))
  #print(nx.get_node_attributes(next_graph, 'cl'))
  #print(nx.get_node_attributes(curr_graph, 'cl'))
  #print(h)
  #print(h[2,2])
  pi_i = 1
  for i in range(N):
    for j in range(i+1, N):
      #print(curr_graph.nodes[j]['cl'])
      pi_i *= np.exp(h[i,j] * curr_graph.nodes[i]['cl'] * curr_graph.nodes[j]['cl'])
  pi_j = 1
  #print(nx.get_node_attributes(next_graph, 'cl'))
  for i in range(N):
    for j in range(i+1, N):
      pi_j *= np.exp(h[i,j] * next_graph.nodes[i]['cl'] * next_graph.nodes[j]['cl'])
  #print(pi_i)
  #print(pi_j)
  a = np.min([1, pi_j/pi_i])

  return a

def metropolis_step_binary(graph, h, N):  
  """
  Networkx version for binary base chain.
  Metropolis step. It is used for the Houdayer algorithms.
  """
  next_step = 0
  while next_step == 0:
    next_step = np.random.choice(range(round(-N/2), round(N/2) + 1))
  curr_state = [0] * N
  for node, cl in nx.get_node_attributes(graph, 'cl').items():
    if cl == -1:
      curr_state[node] = 0
    else:
      curr_state[node] = 1
  
  curr_decimal = binary_to_decimal(curr_state)
  #print(curr_state)
  #print(curr_decimal)
  
  next_decimal = curr_decimal + next_step
  if next_decimal < 0:
    next_decimal += np.power(2, N)
  next_state = decimal_to_binary(next_decimal)
  
  
  for i, binary in enumerate(next_state):
    if binary == 0:
      next_state[i] = -1
    else:
      next_state[i] = 1
  
  if len(next_state) < N:
    filler_list = [-1] * (N - len(next_state))
    next_state = filler_list + next_state
    
  if len(next_state) > N:
    del next_state[0]

  #print(next_state)
  next_graph = graph.copy()
  nx.set_node_attributes(next_graph, dict(zip(range(N), next_state)), 'cl')
  
  a = compute_accep_prob_binary(graph, next_graph, h, N)
  #print(next_step, ': ', a)
  rand_num = np.random.uniform(0,1)
  if rand_num <= a:
    graph = next_graph.copy()  
  
  return graph

def binary_to_decimal(binary_list):
  """
  Converts a binary list into a decimal.
  """
  decimal = 0
  for digit in binary_list:
    decimal = (decimal << 1) | digit
  
  return decimal

def decimal_to_binary(decimal):
  """
  Converts a decimal into a binary list.
  """
  binary_string = [int(i) for i in bin(decimal)[2:]]
  return binary_string

def metropolis_run_binary(graph, N, h, x_star, num_iter):
  """
  Networkx version for binary base chain.
  Metropolis algorithm.
  """
  for i in tqdm(range(num_iter)):
    graph = metropolis_step_binary(graph, h, N)
  return overlap(x_star, graph, N)

def houdayer_run_binary(graph_curr_1, graph_curr_2, N, h, x_star, num_iter, n0 = 1):
  """
  Networkx version for binary base chain.
  Houdayer and Mixed algorithms.
  """
  for i in tqdm(range(num_iter)):
    graph_curr_1 = metropolis_step_binary(graph_curr_1, h, N)
    graph_curr_2 = metropolis_step_binary(graph_curr_2, h, N)
    if i % n0 == 0:
      graph_curr_1, graph_curr_2 = houdayer_step(graph_curr_1, graph_curr_2)
  return overlap(x_star, graph_curr_1, N), overlap(x_star, graph_curr_2, N)