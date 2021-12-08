import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import networkx as nx

def generate_graph_np(N, x_star, a, b):
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
  ln1 = np.log(a/b)
  ln2 = np.log((1 - a/N)/(1 - b/N))
  h = np.zeros_like(graph)
  h = 1/2*(graph * ln1 + (1 - graph) * ln2)
  return h

def probability_np(x, h, a, b, N):
  p = 1
  ln1 = np.log(a/b)
  ln2 = np.log((1 - a/N)/(1 - b/N))
  for i in range(N):
    for j in range(i+1, N):
      p *= np.exp(h[i, j] * x[i] * x[j])
  return p

def acceptance_prob_np(x_1, x_2, h, a, b, N, psi_1, psi_2):
  p1 = probability_np(x_1, h, a, b, N)
  p2 = probability_np(x_2, h, a, b, N)
  return np.min([1, (p2*psi_2)/(p1*psi_1)])

def hamiltonian_np(x, h, a, b, N):
  p = 0
  ln1 = np.log(a/b)
  ln2 = np.log((1 - a/N)/(1 - b/N))
  for i in range(N):
    for j in range(i+1, N):
      p -= h[i, j] * x[i] * x[j]
  return p

def overlap_np(x_star, x_final, N):
  over = 0
  for i in range(N):
    over += x_star[i]*x_final[i]
  return (1/N)*np.abs(over)

def plot_np(ham, over):
  _, axes = plt.subplots(nrows=1, ncols=2, figsize=(32,8))
  axes[0].set_ylabel('Hamiltonian', fontsize=20)
  axes[0].set_xlabel('iter', fontsize=20)
  axes[1].set_ylabel('Overlap', fontsize=20)
  axes[1].set_xlabel('iter', fontsize=20)
  axes[0].plot(ham)
  axes[1].plot(over)
  axes[0].grid()
  axes[1].grid()
  plt.show()
  

#second algorithm

def generate_graph_nx(N, x_star, a, b):
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

def compute_h_nx(graph, a, b, N):
  graph = nx.to_numpy_matrix(graph)
  ln1 = np.log(a/b)
  ln2 = np.log((1 - a/N)/(1 - b/N))
  h = np.zeros_like(graph)
  h = 1/2*(graph * ln1 + (1 - graph) * ln2)
  return h

def probability_nx(graph_x, h, a, b, N):
  p = 1
  ln1 = np.log(a/b)
  ln2 = np.log((1 - a/N)/(1 - b/N))
  for i in range(N):
    for j in range(i+1, N):
      p *= np.exp(h[i, j] * graph_x.nodes[i]['cl'] * graph_x.nodes[j]['cl'])
  return p

def acceptance_prob_nx(x_1, x_2, h, a, b, N, psi_1, psi_2):
  p1 = probability_nx(x_1, h, a, b, N)
  p2 = probability_nx(x_2, h, a, b, N)
  return np.min([1, (p2*psi_2)/(p1*psi_1)])

def metropolis_step_nx(graph_curr, h, a, b, N, psi_1, psi_2):
  rand_ind = np.random.choice(N, 1)[0]
  graph_next = graph_curr.copy()
  graph_next.nodes[rand_ind]['cl'] = -1*graph_next.nodes[rand_ind]['cl']
  a = acceptance_prob_nx(graph_curr, graph_next, h, a, b, N, psi_1, psi_2)
  rand_num = np.random.uniform(0,1)
  if rand_num <= a:
    graph_curr = graph_next.copy()
  return graph_curr

def hamiltonian_nx(graph_x, h, a, b, N):
  p = 0
  ln1 = np.log(a/b)
  ln2 = np.log((1 - a/N)/(1 - b/N))
  for i in range(N):
    for j in range(i+1, N):
      p -= h[i, j] * graph_x.nodes[i]['cl'] * graph_x.nodes[j]['cl']
  return p

def houdayer_step_nx(G1, G2):
  #creating subset
  R = G1.copy()
  R.remove_nodes_from(n for n in G1 if G1.nodes[n]['cl'] == G2.nodes[n]['cl'])
  if len(list(R.nodes)) != 0:
    r_st = np.random.choice(list(R.nodes)) #choosing randomly element
    #finding all connections of the chosen element
    neighbors = [n for n in R.neighbors(r_st)]
    for i in neighbors:
      G1.nodes[i]['cl'] *= -1
      G2.nodes[i]['cl'] *= -1
  return G1, G2

def overlap_nx(x_star, graph_x_curr, N):
  over = 0
  for i in range(N):
    over += x_star[i]*graph_x_curr.nodes[i]['cl']
  return (1/N)*np.abs(over)

def plot_nx(ham1, ham2, over1, over2):
  _, axes = plt.subplots(nrows=1, ncols=2, figsize=(32,8))
  axes[0].set_ylabel('Hamiltonian', fontsize=20)
  axes[0].set_xlabel('iter', fontsize=20)
  axes[1].set_ylabel('Overlap', fontsize=20)
  axes[1].set_xlabel('iter', fontsize=20)
  axes[0].plot(ham1, label = 'x_1')
  axes[0].plot(ham2, label = 'x_2')
  axes[1].plot(over1, label = 'x_1')
  axes[1].plot(over2, label = 'x_2')
  axes[0].legend()
  axes[1].legend()
  axes[0].grid()
  axes[1].grid()
  plt.show()