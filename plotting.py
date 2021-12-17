from IPython.display import clear_output
import matplotlib.pyplot as plt
from tqdm import tqdm
from operator import add
from helpers import *

def metropolis_time_plot_helper(hamiltonians, overlaps, times, N, a, b):
  _, axes = plt.subplots(nrows=1, ncols=2, figsize=(32,8))
  axes[0].set_ylabel('Hamiltonian', fontsize=18)
  axes[0].set_xlabel('Iterations', fontsize=18)
  axes[1].set_ylabel('Overlap', fontsize=18)
  axes[1].set_xlabel('Iterations', fontsize=18)
  axes[0].grid()
  axes[1].grid()
  title1 = 'Hamiltonian over Time: Standard Metropolis Algorithm with N = ' + str(N) + ', a = ' + str(a) + ', b = ' + str(b)
  title2 = 'Overlap over Time: Standard Metropolis Algorithm with N = ' + str(N) + ', a = ' + str(a) + ', b = ' + str(b)
  axes[0].set_title(title1, fontsize=20)
  axes[1].set_title(title2, fontsize=20)
  axes[0].plot(times, hamiltonians)
  axes[1].plot(times, overlaps)  
  filename = 'plots/overlap_metropolis_N' + str(N) + 'a' + str(a) + 'b' + str(b) + '.png'
  plt.savefig(filename)
  #plt.show()

def metropolis_time_plot_compare_helper(hamiltonians_1, hamiltonians_2, hamiltonians_3, overlaps_1, overlaps_2, overlaps_3, times, N, a_1, b_1, a_2, b_2, a_3, b_3):
  _, axes = plt.subplots(nrows=1, ncols=2, figsize=(32,8))
  axes[0].set_ylabel('Hamiltonian', fontsize=18)
  axes[0].set_xlabel('Time', fontsize=18)
  axes[1].set_ylabel('Overlap', fontsize=18)
  axes[1].set_xlabel('Time', fontsize=18)
  axes[0].grid()
  axes[1].grid()
  title1 = 'Hamiltonian over Time: Standard Metropolis Algorithm with N = ' + str(N)
  title2 = 'Overlap over Time: Standard Metropolis Algorithm with N = ' + str(N)
  axes[0].set_title(title1, fontsize=20)
  axes[1].set_title(title2, fontsize=20)
  label_1 = "a = " + str(a_1) + ", b = " + str(b_1)
  label_2 = "a = " + str(a_2) + ", b = " + str(b_2)
  label_3 = "a = " + str(a_3) + ", b = " + str(b_3)
  axes[0].plot(times, hamiltonians_1, label = label_1)
  axes[0].plot(times, hamiltonians_2, label = label_2)
  axes[0].plot(times, hamiltonians_3, label = label_3)
  axes[1].plot(times, overlaps_1, label = label_1)
  axes[1].plot(times, overlaps_2, label = label_2)
  axes[1].plot(times, overlaps_3, label = label_3) 
  axes[0].legend()
  axes[1].legend()
  filename = 'plots/overlap_metropolis_compare_N' + str(N) + "b1" + str(b_1) + "b2" + str(b_2) + "b3" + str(b_3)
  plt.savefig(filename)
  plt.show()

def houdayer_time_plot_compare_helper(hamiltonians_1, hamiltonians_2, hamiltonians_3, overlaps_1, overlaps_2, overlaps_3, times, N, a_1, b_1, a_2, b_2, a_3, b_3):
  _, axes = plt.subplots(nrows=1, ncols=2, figsize=(32,8))
  axes[0].set_ylabel('Hamiltonian', fontsize=18)
  axes[0].set_xlabel('Time', fontsize=18)
  axes[1].set_ylabel('Overlap', fontsize=18)
  axes[1].set_xlabel('Time', fontsize=18)
  axes[0].grid()
  axes[1].grid()
  title1 = 'Hamiltonian over Time: Houdayer Algorithm with N = ' + str(N)
  title2 = 'Overlap over Time: Houdayer Algorithm with N = ' + str(N)
  axes[0].set_title(title1, fontsize=20)
  axes[1].set_title(title2, fontsize=20)
  label_1 = "a = " + str(a_1) + ", b = " + str(b_1)
  label_2 = "a = " + str(a_2) + ", b = " + str(b_2)
  label_3 = "a = " + str(a_3) + ", b = " + str(b_3)
  axes[0].plot(times, hamiltonians_1, label = label_1)
  axes[0].plot(times, hamiltonians_2, label = label_2)
  axes[0].plot(times, hamiltonians_3, label = label_3)
  axes[1].plot(times, overlaps_1, label = label_1)
  axes[1].plot(times, overlaps_2, label = label_2)
  axes[1].plot(times, overlaps_3, label = label_3) 
  axes[0].legend()
  axes[1].legend()
  filename = 'plots/overlap_houdayer_compare_N' + str(N) + "b1" + str(b_1) + "b2" + str(b_2) + "b3" + str(b_3)
  plt.savefig(filename)
  plt.show()

def houdayer_time_plot_compare_n0_helper(hamiltonians_1, hamiltonians_2, hamiltonians_3, overlaps_1, overlaps_2, overlaps_3, times, N, a, b, n0_1, n0_2, n0_3):
  _, axes = plt.subplots(nrows=1, ncols=2, figsize=(32,8))
  axes[0].set_ylabel('Hamiltonian', fontsize=18)
  axes[0].set_xlabel('Time', fontsize=18)
  axes[1].set_ylabel('Overlap', fontsize=18)
  axes[1].set_xlabel('Time', fontsize=18)
  axes[0].grid()
  axes[1].grid()
  title1 = 'Hamiltonian over Time: Mixed Metropolis-Houdayer Algorithm with N = ' + str(N) + ', a = ' + str(a) + ', b = ' + str(b)
  title2 = 'Overlap over Time: Mixed Metropolis-Houdayer Algorithm with N = ' + str(N) + ', a = ' + str(a) + ', b = ' + str(b)
  axes[0].set_title(title1, fontsize=20)
  axes[1].set_title(title2, fontsize=20)
  label_1 = "n0 = " + str(n0_1 - 1)
  label_2 = "n0 = " + str(n0_2 - 1)
  label_3 = "n0 = " + str(n0_3 - 1)
  axes[0].plot(times, hamiltonians_1, label = label_1)
  axes[0].plot(times, hamiltonians_2, label = label_2)
  axes[0].plot(times, hamiltonians_3, label = label_3)
  axes[1].plot(times, overlaps_1, label = label_1)
  axes[1].plot(times, overlaps_2, label = label_2)
  axes[1].plot(times, overlaps_3, label = label_3) 
  axes[0].legend()
  axes[1].legend()
  filename = 'plots/overlap_mixed_compare_n0_N' + str(N) + "a" + str(a) + "b" + str(b) + '.png'
  plt.savefig(filename)
  plt.show()

def plot_algorithms_compare_helper(hamiltonians_1, hamiltonians_2, hamiltonians_3, overlaps_1, overlaps_2, overlaps_3, times, N, a, b, n0):
  _, axes = plt.subplots(nrows=1, ncols=2, figsize=(32,8))
  axes[0].set_ylabel('Hamiltonian', fontsize=26)
  axes[0].set_xlabel('Time', fontsize=26)
  axes[1].set_ylabel('Overlap', fontsize=26)
  axes[1].set_xlabel('Time', fontsize=26)
  axes[0].grid()
  axes[1].grid()
  title1 = 'Hamiltonian over Time with N = ' + str(N) + ', a = ' + str(a) + ', b = ' + str(b)
  title2 = 'Overlap over Time with N = ' + str(N) + ', a = ' + str(a) + ', b = ' + str(b)
  axes[0].set_title(title1, fontsize=28, pad=20)
  axes[1].set_title(title2, fontsize=28, pad=20)
  label_1 = "Metropolis"
  label_2 = "Houdayer"
  label_3 = "Mixed n0 = " + str(n0 - 1)
  axes[0].plot(times, hamiltonians_1, label = label_1)
  axes[0].plot(times, hamiltonians_2, label = label_2)
  axes[0].plot(times, hamiltonians_3, label = label_3)
  axes[1].plot(times, overlaps_1, label = label_1)
  axes[1].plot(times, overlaps_2, label = label_2)
  axes[1].plot(times, overlaps_3, label = label_3) 
  axes[0].legend(prop={'size': 22})
  axes[1].legend(prop={'size': 22})
  filename_png = 'plots/overlap_algorithms_compare_N' + str(N) + "a" + str(a) + "b" + str(b) + '.png'
  plt.savefig(filename_png)

def houdayer_time_plot_helper(hamiltonians_1, hamiltonians_2, overlaps_1, overlaps_2):
  _, axes = plt.subplots(nrows=1, ncols=2, figsize=(32,8))
  axes[0].set_ylabel('Hamiltonian', fontsize=20)
  axes[0].set_xlabel('Iterations', fontsize=20)
  axes[1].set_ylabel('Overlap', fontsize=20)
  axes[1].set_xlabel('Iterations', fontsize=20)
  axes[0].plot(hamiltonians_1, label = 'x_1')
  axes[0].plot(hamiltonians_2, label = 'x_2')
  axes[1].plot(overlaps_1, label = 'x_1')
  axes[1].plot(overlaps_2, label = 'x_2')
  axes[0].legend()
  axes[1].legend()
  axes[0].grid()
  axes[1].grid()
  axes[0].title.set_text('Hamiltonian over Time (Houdayer Algorithm)')
  axes[1].title.set_text("Overlap over time (Houdayer Algorithm)")
  plt.show()

def plot_ratio_metropolis_helper(ratio, over):
  _, axes = plt.subplots(nrows=1, ncols=1, figsize=(16,8))
  axes.set_ylabel('overlap', fontsize=20)
  axes.set_xlabel('b/a ratio', fontsize=20)
  axes.plot(ratio, over)
  axes.grid()
  plt.show()

def plot_ratio_houdayer_helper(ratio, over1, over2):
  _, axes = plt.subplots(nrows=1, ncols=1, figsize=(16,8))
  axes.set_ylabel('overlap', fontsize=20)
  axes.set_xlabel('b/a ratio', fontsize=20)
  axes.plot(ratio, over1, label = 'x_1')
  axes.plot(ratio, over2, label = 'x_2')
  axes.legend()
  axes.grid()
  plt.show()

def plot_time_overlap_metropolis_compare(graph_1, graph_2, graph_3, N, a_1, b_1, a_2, b_2, a_3, b_3, x_star, num_iter):
  h_1 = compute_h_np(graph_1, a_1, b_1, N)
  h_2 = compute_h_np(graph_2, a_2, b_2, N)
  h_3 = compute_h_np(graph_3, a_3, b_3, N)

  hamiltonians_1 = []
  hamiltonians_2 = []
  hamiltonians_3 = []
  overlaps_1 = []
  overlaps_2 = []
  overlaps_3 = []
  times = []
  curr_state_1 = np.random.choice([-1, 1], N)
  curr_state_2 = curr_state_1.copy()
  curr_state_3 = curr_state_1.copy()
  for i in range(num_iter):
    curr_state_1 = metropolis_step_np(curr_state_1, h_1, N, 1/N, 1/N)
    curr_state_2 = metropolis_step_np(curr_state_2, h_2, N, 1/N, 1/N)
    curr_state_3 = metropolis_step_np(curr_state_3, h_3, N, 1/N, 1/N)
    if i % 30 == 0:
      hamiltonians_1.append(hamiltonian_np(curr_state_1, h_1, N))
      hamiltonians_2.append(hamiltonian_np(curr_state_2, h_2, N))
      hamiltonians_3.append(hamiltonian_np(curr_state_3, h_3, N))
      overlaps_1.append(overlap_np(x_star, curr_state_1, N))
      overlaps_2.append(overlap_np(x_star, curr_state_2, N))
      overlaps_3.append(overlap_np(x_star, curr_state_3, N))
      times.append(i)
      clear_output(wait=True)
      metropolis_time_plot_compare_helper(hamiltonians_1, hamiltonians_2, hamiltonians_3, overlaps_1, overlaps_2, overlaps_3, times, N, a_1, b_1, a_2, b_2, a_3, b_3)

def plot_time_overlap_metropolis(graph, N, a, b, x_star, num_iter):
  h = compute_h_np(graph, a, b, N)

  hamiltonians = []
  overlaps = []
  times = []
  curr_state = np.random.choice([-1, 1], N)
  for i in range(num_iter):
    curr_state = metropolis_step_np(curr_state, h, N, 1/N, 1/N)
    if i % 30 == 0:
      hamiltonians.append(hamiltonian_np(curr_state, h, N))
      overlaps.append(overlap_np(x_star, curr_state, N))
      times.append(i)
      clear_output(wait=True)
  metropolis_time_plot_helper(hamiltonians, overlaps, times, N, a, b)

def plot_time_overlap_houdayer_compare(graph_1, graph_2, graph_3, N, a_1, b_1, a_2, b_2, a_3, b_3, x_star, num_iter):
  h_1 = compute_h(graph_1, a_1, b_1, N)
  h_2 = compute_h(graph_2, a_2, b_2, N)
  h_3 = compute_h(graph_3, a_3, b_3, N)

  hamiltonians_1 = []
  hamiltonians_2 = []
  hamiltonians_3 = []
  
  overlaps_1 = []
  overlaps_2 = []
  overlaps_3 = []

  times = []
  
  initial_state = np.random.choice([-1, 1], N)
  
  graph_1_1 = graph_1.copy()
  nx.set_node_attributes(graph_1_1, dict(zip(range(N), initial_state)), 'cl')
  graph_1_2 = graph_1.copy()
  nx.set_node_attributes(graph_1_2, dict(zip(range(N), initial_state)), 'cl')
  
  graph_2_1 = graph_2.copy()
  nx.set_node_attributes(graph_2_1, dict(zip(range(N), initial_state)), 'cl')
  graph_2_2 = graph_2.copy()
  nx.set_node_attributes(graph_2_2, dict(zip(range(N), initial_state)), 'cl')
  
  graph_3_1 = graph_3.copy()
  nx.set_node_attributes(graph_3_1, dict(zip(range(N), initial_state)), 'cl')
  graph_3_2 = graph_3.copy()
  nx.set_node_attributes(graph_3_2, dict(zip(range(N), initial_state)), 'cl')

  for i in range(num_iter):
    graph_1_1 = metropolis_step(graph_1_1, h_1, N, 1/N, 1/N)
    graph_1_2 = metropolis_step(graph_1_2, h_1, N, 1/N, 1/N)
    graph_1_1, graph_1_2 = houdayer_step(graph_1_1, graph_1_2)

    graph_2_1 = metropolis_step(graph_2_1, h_2, N, 1/N, 1/N)
    graph_2_2 = metropolis_step(graph_2_2, h_2, N, 1/N, 1/N)
    graph_2_1, graph_2_2 = houdayer_step(graph_2_1, graph_2_2)

    graph_3_1 = metropolis_step(graph_3_1, h_3, N, 1/N, 1/N)
    graph_3_2 = metropolis_step(graph_3_2, h_3, N, 1/N, 1/N)
    graph_3_1, graph_3_2 = houdayer_step(graph_3_1, graph_3_2)

    if i % 15 == 0:
      hamiltonians_1.append(hamiltonian(graph_1_1, h_1, N))
      hamiltonians_2.append(hamiltonian(graph_2_1, h_2, N))
      hamiltonians_3.append(hamiltonian(graph_3_1, h_3, N))

      overlaps_1.append(overlap(x_star, graph_1_1, N))
      overlaps_2.append(overlap(x_star, graph_2_1, N))
      overlaps_3.append(overlap(x_star, graph_3_1, N))
      
      times.append(i * 2)
      clear_output(wait=True)
      
      houdayer_time_plot_compare_helper(hamiltonians_1, hamiltonians_2, hamiltonians_3, overlaps_1, overlaps_2, overlaps_3, times, N, a_1, b_1, a_2, b_2, a_3, b_3)

def plot_time_overlap_houdayer(graph, N, a, b, x_star, num_iter):
  h = compute_h(graph, a, b, N)

  hamiltonians_1 = []
  hamiltonians_2 = []
  overlaps_1 = []
  overlaps_2 = []
  initial_state = np.random.choice([-1, 1], N)
  graph_1 = graph.copy()
  nx.set_node_attributes(graph_1, dict(zip(range(N), initial_state)), 'cl')
  graph_2 = graph.copy()
  nx.set_node_attributes(graph_2, dict(zip(range(N), initial_state)), 'cl')
  for i in range(num_iter):
    graph_1 = metropolis_step(graph_1, h, N, 1/N, 1/N)
    graph_2 = metropolis_step(graph_2, h, N, 1/N, 1/N)
    graph_1, graph_2 = houdayer_step(graph_1, graph_2)
    if i % 10 == 0:
      hamiltonians_1.append(hamiltonian(graph_1, h, N))
      hamiltonians_2.append(hamiltonian(graph_2, h, N))
      overlaps_1.append(overlap(x_star, graph_1, N))
      overlaps_2.append(overlap(x_star, graph_2, N))
      clear_output(wait=True)
      houdayer_time_plot_helper(hamiltonians_1, hamiltonians_2, overlaps_1, overlaps_2)

def plot_time_overlap_mixed(graph, N, a, b, x_star, num_iter, n0):
  h = compute_h(graph, a, b, N)

  hams_1 = []
  hams_2 = []
  overlaps_1 = []
  overlaps_2 = []
  initial_state = np.random.choice([-1, 1], N)
  graph_curr_1 = generate_graph(N, initial_state, a, b)
  graph_curr_2 = generate_graph(N, initial_state, a, b)
  for i in range(num_iter):
    graph_curr_1 = metropolis_step(graph_curr_1, h, N, 1/N, 1/N)
    graph_curr_2 = metropolis_step(graph_curr_2, h, N, 1/N, 1/N)
    if i % n0 == 0:
      graph_curr_1, graph_curr_2 = houdayer_step(graph_curr_1, graph_curr_2)
    if i % 10 == 0:
      hams_1.append(hamiltonian(graph_curr_1, h, N))
      hams_2.append(hamiltonian(graph_curr_2, h, N))
      overlaps_1.append(overlap(x_star, graph_curr_1, N))
      overlaps_2.append(overlap(x_star, graph_curr_2, N))
      clear_output(wait=True)
      houdayer_plot_helper(hams_1, hams_2, overlaps_1, overlaps_2)

def plot_time_overlap_mixed_compare(graph, N, a, b, x_star, num_iter, n0_1, n0_2, n0_3):
  h = compute_h(graph, a, b, N)

  hamiltonians_1 = []
  hamiltonians_2 = []
  hamiltonians_3 = []

  overlaps_1 = []
  overlaps_2 = []
  overlaps_3 = []

  times = []

  initial_state = np.random.choice([-1, 1], N)

  graph_1_1 = graph.copy()
  nx.set_node_attributes(graph_1_1, dict(zip(range(N), initial_state)), 'cl')
  graph_1_2 = graph_1_1.copy()
  
  graph_2_1 = graph_1_1.copy()
  graph_2_2 = graph_1_1.copy()
  
  graph_3_1 = graph_1_1.copy()
  graph_3_2 = graph_1_1.copy()



  for i in range(num_iter):
    if i % n0_1 == 1:
      graph_1_1, graph_1_2 = houdayer_step(graph_1_1, graph_1_2)
    else:
      graph_1_1 = metropolis_step(graph_1_1, h, N, 1/N, 1/N)
      graph_1_2 = metropolis_step(graph_1_2, h, N, 1/N, 1/N)

    if i % n0_2 == 1:
      graph_2_1, graph_2_2 = houdayer_step(graph_2_1, graph_2_2)
    else:
      graph_2_1 = metropolis_step(graph_2_1, h, N, 1/N, 1/N)
      graph_2_2 = metropolis_step(graph_2_2, h, N, 1/N, 1/N)

    if i % n0_3 == 1:
      graph_3_1, graph_3_2 = houdayer_step(graph_3_1, graph_3_2)
    else:
      graph_3_1 = metropolis_step(graph_3_1, h, N, 1/N, 1/N)
      graph_3_2 = metropolis_step(graph_3_2, h, N, 1/N, 1/N)
    
    if i % 10 == 0:
      hamiltonians_1.append(hamiltonian(graph_1_1, h, N))
      hamiltonians_2.append(hamiltonian(graph_2_1, h, N))
      hamiltonians_3.append(hamiltonian(graph_3_1, h, N))

      overlaps_1.append(overlap(x_star, graph_1_1, N))
      overlaps_2.append(overlap(x_star, graph_2_1, N))
      overlaps_3.append(overlap(x_star, graph_3_1, N))

      times.append(i)

      clear_output(wait=True)
      
      houdayer_time_plot_compare_n0_helper(hamiltonians_1, hamiltonians_2, hamiltonians_3, overlaps_1, overlaps_2, overlaps_3, times, N, a, b, n0_1, n0_2, n0_3)

def plot_time_average_overlap_algorithms_compare(N, a, b, num_iter, n0, num_exp):
  
  hamiltonians_avg_1 = []
  hamiltonians_avg_2 = []
  hamiltonians_avg_3 = []
  
  overlaps_avg_1 = []
  overlaps_avg_2 = []
  overlaps_avg_3 = []

  times_avg = []

  for exp in tqdm(range(num_exp)):
    x_star = np.random.choice([-1, 1], N)
    graph = generate_graph(N, x_star, a, b)
    
    hamiltonians_1, hamiltonians_2, hamiltonians_3, overlaps_1, overlaps_2, overlaps_3, times = plot_time_overlap_algorithms_compare(graph, N, a, b, x_star, num_iter, n0)
    
    if exp == 0:
      hamiltonians_avg_1 = hamiltonians_1.copy()
      hamiltonians_avg_2 = hamiltonians_2.copy()
      hamiltonians_avg_3 = hamiltonians_3.copy()

      overlaps_avg_1 = overlaps_1.copy()
      overlaps_avg_2 = overlaps_2.copy()
      overlaps_avg_3 = overlaps_3.copy()

      times_avg = times.copy()

    else:
      hamiltonians_avg_1 = list(map(add, hamiltonians_avg_1, hamiltonians_1))
      hamiltonians_avg_2 = list(map(add, hamiltonians_avg_2, hamiltonians_2))
      hamiltonians_avg_3 = list(map(add, hamiltonians_avg_3, hamiltonians_3))

      overlaps_avg_1 = list(map(add, overlaps_avg_1, overlaps_1))
      overlaps_avg_2 = list(map(add, overlaps_avg_2, overlaps_2))
      overlaps_avg_3 = list(map(add, overlaps_avg_3, overlaps_3))

    filename_txt = 'cache/overlap_algorithms_compare_N' + str(N) + "a" + str(a) + "b" + str(b) + '.txt'
    with open(filename_txt, 'a') as f:
      if exp == 0:
        f.write('Number experiments: ' + str(num_exp) + '\n')
        f.write('times: ' + str(times) + '\n \n \n')
      f.write(str(exp) + ':\n')
      f.write('Sum Hamiltonians Metropolis' + str(hamiltonians_avg_1) + '\n')
      f.write('Sum Hamiltonians Houdayer' + str(hamiltonians_avg_2) + '\n')
      f.write('Sum Hamiltonians Mixed' + str(hamiltonians_avg_3) + '\n')

      f.write('Sum Overlaps Metropolis' + str(overlaps_avg_1) + '\n')
      f.write('Sum Overlaps Houdayer' + str(overlaps_avg_2) + '\n')
      f.write('Sum Overlaps Mixed' + str(overlaps_avg_3) + '\n\n\n')
  
  hamiltonians_avg_1 = [(1/num_exp) * i for i in hamiltonians_avg_1]
  hamiltonians_avg_2 = [(1/num_exp) * i for i in hamiltonians_avg_2]
  hamiltonians_avg_3 = [(1/num_exp) * i for i in hamiltonians_avg_3]

  overlaps_avg_1 = [(1/num_exp) * i for i in overlaps_avg_1]
  overlaps_avg_2 = [(1/num_exp) * i for i in overlaps_avg_2]
  overlaps_avg_3 = [(1/num_exp) * i for i in overlaps_avg_3]

  with open(filename_txt, 'a') as f:
    f.write('Final Average:\n')
    f.write('Average Hamiltonians Metropolis' + str(hamiltonians_avg_1) + '\n')
    f.write('Average Hamiltonians Houdayer' + str(hamiltonians_avg_2) + '\n')
    f.write('Average Hamiltonians Mixed' + str(hamiltonians_avg_3) + '\n')

    f.write('Average Overlaps Metropolis' + str(overlaps_avg_1) + '\n')
    f.write('Average Overlaps Houdayer' + str(overlaps_avg_2) + '\n')
    f.write('Average Overlaps Mixed' + str(overlaps_avg_3) + '\n')

  return hamiltonians_avg_1, hamiltonians_avg_2, hamiltonians_avg_3, overlaps_avg_1, overlaps_avg_2, overlaps_avg_3, times_avg


def plot_time_overlap_algorithms_compare(graph, N, a, b, x_star, num_iter, n0):
  h = compute_h(graph, a, b, N)

  hamiltonians_1 = []
  hamiltonians_2 = []
  hamiltonians_3 = []
  
  overlaps_1 = []
  overlaps_2 = []
  overlaps_3 = []

  times = []
  
  curr_state = np.random.choice([-1, 1], N)
  
  graph_2_1 = graph.copy()
  nx.set_node_attributes(graph_2_1, dict(zip(range(N), curr_state)), 'cl')
  graph_2_2 = graph.copy()
  nx.set_node_attributes(graph_2_2, dict(zip(range(N), curr_state)), 'cl')
  
  graph_3_1 = graph.copy()
  nx.set_node_attributes(graph_3_1, dict(zip(range(N), curr_state)), 'cl')
  graph_3_2 = graph.copy()
  nx.set_node_attributes(graph_3_2, dict(zip(range(N), curr_state)), 'cl')

  for i in range(num_iter):
    curr_state = metropolis_step_np(curr_state, h, N, 1/N, 1/N)
    if i % 2 == 0:
      graph_2_1 = metropolis_step(graph_2_1, h, N, 1/N, 1/N)
      graph_2_2 = metropolis_step(graph_2_2, h, N, 1/N, 1/N)
    else:
      graph_2_1, graph_2_2 = houdayer_step(graph_2_1, graph_2_2)

    if i % n0 == 1:
      graph_3_1, graph_3_2 = houdayer_step(graph_3_1, graph_3_2)
    else:
      graph_3_1 = metropolis_step(graph_3_1, h, N, 1/N, 1/N)
      graph_3_2 = metropolis_step(graph_3_2, h, N, 1/N, 1/N)
    

    if i % 15 == 0:
      hamiltonians_1.append(hamiltonian_np(curr_state, h, N))
      hamiltonians_2.append(hamiltonian(graph_2_1, h, N))
      hamiltonians_3.append(hamiltonian(graph_3_1, h, N))

      overlaps_1.append(overlap_np(x_star, curr_state, N))
      overlaps_2.append(overlap(x_star, graph_2_1, N))
      overlaps_3.append(overlap(x_star, graph_3_1, N))
      
      times.append(i)
  
  return hamiltonians_1, hamiltonians_2, hamiltonians_3, overlaps_1, overlaps_2, overlaps_3, times

def plot_time_overlap_metropolis(graph, N, a, b, x_star, num_iter):
  h = compute_h_np(graph, a, b, N)

  hamiltonians = []
  overlaps = []
  times = []
  curr_state = np.random.choice([-1, 1], N)
  for i in range(num_iter):
    curr_state = metropolis_step_np(curr_state, h, N, 1/N, 1/N)
    if i % 30 == 0:
      hamiltonians.append(hamiltonian_np(curr_state, h, N))
      overlaps.append(overlap_np(x_star, curr_state, N))
      times.append(i)
      clear_output(wait=True)
  metropolis_time_plot_helper(hamiltonians, overlaps, times, N, a, b)



def plot_ratio_overlap_metropolis(N, num_iter):
  ratios = []
  overlaps = []
  for ratio in range(1, 10):
    ratio = ratio/10
    ratios.append(ratio)
    b = 50
    a = b/ratio
    # should fix it
    np.random.seed(100)
    x_star = np.random.choice([-1, 1], N)
    graph = generate_graph_np(N, x_star, a, b)
    h = compute_h_np(graph, a, b, N)
    # should fix it
    initial_state = np.random.choice([-1, 1], N)
    over = metropolis_run(initial_state, N, h, x_star, num_iter)
    overlaps.append(over)
  plot_ratio_metropolis_helper(ratios, overlaps)


def plot_ratio_overlap_houdayer(N, num_iter):
  ratios = []
  overlaps_1 = []
  overlaps_2 = []
  for ratio in range(1, 10):
    print(ratio)
    ratio = ratio/10
    ratios.append(ratio)
    b = 50
    a = b/ratio
    # should fix it
    np.random.seed(100)
    x_star = np.random.choice([-1, 1], N)
    graph = generate_graph(N, x_star, a, b)
    h = compute_h(graph, a, b, N)
    # should fix it
    initial_state = np.random.choice([-1, 1], N)
    graph_curr_1 = generate_graph(N, initial_state, a, b)
    graph_curr_2 = generate_graph(N, initial_state, a, b)
    over1, over2 = houdayer_run(graph_curr_1, graph_curr_2, N, h, x_star, num_iter)
    overlaps_1.append(over1)
    overlaps_2.append(over2)
  plot_ratio_houdayer_helper(ratios, overlaps_1, overlaps_2)

def plot_ratio_overlap_mixed(N, num_iter, n0):
  ratios = []
  overlaps_1 = []
  overlaps_2 = []
  for ratio in range(1, 10):
    print(ratio)
    ratio = ratio/10
    ratios.append(ratio)
    b = 50
    a = b/ratio
    # should fix it
    np.random.seed(100)
    x_star = np.random.choice([-1, 1], N)
    graph = generate_graph(N, x_star, a, b)
    h = compute_h(graph, a, b, N)
    # should fix it
    initial_state = np.random.choice([-1, 1], N)
    graph_curr_1 = generate_graph(N, initial_state, a, b)
    graph_curr_2 = generate_graph(N, initial_state, a, b)
    over1, over2 = houdayer_run(graph_curr_1, graph_curr_2, N, h, x_star, num_iter, n0)
    overlaps_1.append(over1)
    overlaps_2.append(over2)
  plot_ratio_houdayer_helper(ratios, overlaps_1, overlaps_2)