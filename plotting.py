from IPython.display import clear_output
import matplotlib.pyplot as plt
from tqdm import tqdm
from operator import add
from helpers import *
import seaborn as sns
import networkx as nx


# base chain !!!!!!!!!!!

# convergence time - ham + over all at one
# plot dependece of the conv time with different r and d
# dependence on b/a ration - 1) try different fixed a values for each algo. 1 plot per algorithm (pay attension on d) 2) choose best fixed value for each algo, compare them on the 1 plot
# dependence over n0
# dependence on b/a for mixed algo for different n0

"""Convergence time"""

# convergence time - ham + over all at one

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

def plot_time_average_overlap_algorithms_compare(N, a, b, num_iter, n0, num_exp):

  for exp_num in tqdm(range(num_exp)):
    x_star = np.random.choice([-1, 1], N)
    graph = generate_graph(N, x_star, a, b)
  
    hamiltonians_1, hamiltonians_2, hamiltonians_3, overlaps_1, overlaps_2, overlaps_3, times = plot_time_overlap_algorithms_compare(graph, N, a, b, x_star, num_iter, n0)
    if exp_num == 0:
      hamiltonians_avg_1 = np.array(hamiltonians_1.copy())
      hamiltonians_avg_2 = np.array(hamiltonians_2.copy())
      hamiltonians_avg_3 = np.array(hamiltonians_3.copy())

      overlaps_avg_1 = np.array(overlaps_1.copy())
      overlaps_avg_2 = np.array(overlaps_2.copy())
      overlaps_avg_3 = np.array(overlaps_3.copy())

      times_avg = np.array(times.copy())

    else:
      hamiltonians_avg_1 = hamiltonians_avg_1 + np.array(hamiltonians_1)
      hamiltonians_avg_2 = hamiltonians_avg_2 + np.array(hamiltonians_2)
      hamiltonians_avg_3 = hamiltonians_avg_3 + np.array(hamiltonians_3)

      overlaps_avg_1 = overlaps_avg_1 + np.array(overlaps_1)
      overlaps_avg_2 = overlaps_avg_2 + np.array(overlaps_2)
      overlaps_avg_3 = overlaps_avg_3 + np.array(overlaps_3)

    # filename_txt = 'cache/overlap_algorithms_compare_N' + str(N) + "a" + str(a) + "b" + str(b) + '.txt'
    # with open(filename_txt, 'a') as f:
    #   if exp_num == 0:
    #     f.write('Number experiments: ' + str(num_exp) + '\n')
    #     f.write('times: ' + str(times) + '\n \n \n')
    #   f.write(str(exp_num) + ':\n')
    #   f.write('Sum Hamiltonians Metropolis' + str(hamiltonians_avg_1) + '\n')
    #   f.write('Sum Hamiltonians Houdayer' + str(hamiltonians_avg_2) + '\n')
    #   f.write('Sum Hamiltonians Mixed' + str(hamiltonians_avg_3) + '\n')

    #   f.write('Sum Overlaps Metropolis' + str(overlaps_avg_1) + '\n')
    #   f.write('Sum Overlaps Houdayer' + str(overlaps_avg_2) + '\n')
    #   f.write('Sum Overlaps Mixed' + str(overlaps_avg_3) + '\n\n\n')
  
  hamiltonians_avg_1 = hamiltonians_avg_1 / num_exp
  hamiltonians_avg_2 = hamiltonians_avg_2 / num_exp
  hamiltonians_avg_3 = hamiltonians_avg_3 / num_exp

  overlaps_avg_1 = overlaps_avg_1 / num_exp
  overlaps_avg_2 = overlaps_avg_2 / num_exp
  overlaps_avg_3 = overlaps_avg_3 / num_exp

  # with open(filename_txt, 'a') as f:
  #   f.write('Final Average:\n')
  #   f.write('Average Hamiltonians Metropolis' + str(hamiltonians_avg_1) + '\n')
  #   f.write('Average Hamiltonians Houdayer' + str(hamiltonians_avg_2) + '\n')
  #   f.write('Average Hamiltonians Mixed' + str(hamiltonians_avg_3) + '\n')

  #   f.write('Average Overlaps Metropolis' + str(overlaps_avg_1) + '\n')
  #   f.write('Average Overlaps Houdayer' + str(overlaps_avg_2) + '\n')
  #   f.write('Average Overlaps Mixed' + str(overlaps_avg_3) + '\n')

  return hamiltonians_avg_1, hamiltonians_avg_2, hamiltonians_avg_3, overlaps_avg_1, overlaps_avg_2, overlaps_avg_3, times_avg

def plot_algorithms_compare_helper(hamiltonians_1, hamiltonians_2, hamiltonians_3, overlaps_1, overlaps_2, overlaps_3, times, N, a, b, n0):
  _, axes = plt.subplots(nrows=1, ncols=2, figsize=(24,8))
  axes[0].set_ylabel('Hamiltonian', fontsize=20)
  axes[0].set_xlabel('Time', fontsize=20)
  axes[1].set_ylabel('Overlap', fontsize=20)
  axes[1].set_xlabel('Time', fontsize=20)
  title1 = 'Hamiltonian over Time with N = ' + str(N) + ', a = ' + str(a) + ', b = ' + str(b)
  title2 = 'Overlap over Time with N = ' + str(N) + ', a = ' + str(a) + ', b = ' + str(b)
  axes[0].set_title(title1, fontsize=22, pad=20)
  axes[1].set_title(title2, fontsize=22, pad=20)
  label_1 = "Metropolis"
  label_2 = "Houdayer"
  label_3 = "Mixed n0 = " + str(n0 - 1)
  axes[0].plot(times, hamiltonians_1, label = label_1, linewidth=3.0)
  axes[0].plot(times, hamiltonians_2, label = label_2, linewidth=3.0)
  axes[0].plot(times, hamiltonians_3, label = label_3, linewidth=3.0)
  axes[1].plot(times, overlaps_1, label = label_1, linewidth=3.0)
  axes[1].plot(times, overlaps_2, label = label_2, linewidth=3.0)
  axes[1].plot(times, overlaps_3, label = label_3, linewidth=3.0) 
  axes[0].legend(prop={'size': 20})
  axes[1].legend(prop={'size': 20})
  filename_png = 'plots/overlap_algorithms_compare_N' + str(N) + "a" + str(a) + "b" + str(b) + '.png'
  plt.savefig(filename_png)

# plot dependece of the conv time with different r and d

def time_plot_compare_helper(overlaps_1, overlaps_2, overlaps_3, times, N, a_1, b_1, a_2, b_2, a_3, b_3, algo):
  _, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
  label_1 = "a = " + str(a_1) + ", b = " + str(b_1)
  label_2 = "a = " + str(a_2) + ", b = " + str(b_2)
  label_3 = "a = " + str(a_3) + ", b = " + str(b_3)
  sns.lineplot(times, overlaps_1, label = label_1, ax = axes, linewidth=3.0)
  sns.lineplot(times, overlaps_2, label = label_2, ax = axes, linewidth=3.0)
  sns.lineplot(times, overlaps_3, label = label_3, ax = axes, linewidth=3.0)
  axes.set_ylabel('Overlap', fontsize=20)
  axes.set_xlabel('Time', fontsize=20)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  # axes.grid()
  axes.legend(prop={'size': 20})
  title = 'Overlap over Time: ' + algo + ' with N = ' + str(N)
  axes.set_title(title, fontsize=22, pad=20)
  if algo == 'Standard Metropolis Algorithm':
    filename = 'plots/overlap_metropolis_compare_N' + str(N) + "b1" + str(b_1) + "b2" + str(b_2) + "b3" + str(b_3) + '.png'
  else:
    filename = 'plots/overlap_houdayer_compare_N' + str(N) + "b1" + str(b_1) + "b2" + str(b_2) + "b3" + str(b_3) + '.png'
  plt.savefig(filename)

def plot_time_overlap_metropolis_compare(N, a_1, b_1, a_2, b_2, a_3, b_3, num_iter):
  """
  Compare 3 different combination of a and b parameters.
  Plot 3 hamiltonians and overlaps over time
  """
  x_star = np.random.choice([-1, 1], N)

  graph_1 = generate_graph_np(N, x_star, a_1, b_1)
  graph_2 = generate_graph_np(N, x_star, a_2, b_2)
  graph_3 = generate_graph_np(N, x_star, a_3, b_3)

  h_1 = compute_h_np(graph_1, a_1, b_1, N)
  h_2 = compute_h_np(graph_2, a_2, b_2, N)
  h_3 = compute_h_np(graph_3, a_3, b_3, N)

  overlaps_1 = []
  overlaps_2 = []
  overlaps_3 = []
  times = []
  curr_state_1 = np.random.choice([-1, 1], N)
  curr_state_2 = curr_state_1.copy()
  curr_state_3 = curr_state_1.copy()
  for i in tqdm(range(num_iter)):
    curr_state_1 = metropolis_step_np(curr_state_1, h_1, N, 1/N, 1/N)
    curr_state_2 = metropolis_step_np(curr_state_2, h_2, N, 1/N, 1/N)
    curr_state_3 = metropolis_step_np(curr_state_3, h_3, N, 1/N, 1/N)
    if i % 30 == 0:
      overlaps_1.append(overlap_np(x_star, curr_state_1, N))
      overlaps_2.append(overlap_np(x_star, curr_state_2, N))
      overlaps_3.append(overlap_np(x_star, curr_state_3, N))
      times.append(i)
      clear_output(wait=True)
  time_plot_compare_helper(overlaps_1, overlaps_2, overlaps_3, times, N, a_1, b_1, a_2, b_2, a_3, b_3, 'Standard Metropolis Algorithm')


def plot_time_overlap_houdayer_compare(N, a_1, b_1, a_2, b_2, a_3, b_3, num_iter, n0 = 1):
  """
  Compare 3 different a and b configurations for Houdayer algorithm.
  """
  x_star = np.random.choice([-1, 1], N)

  graph_1 = generate_graph(N, x_star, a_1, b_1)
  graph_2 = generate_graph(N, x_star, a_2, b_2)
  graph_3 = generate_graph(N, x_star, a_3, b_3)

  h_1 = compute_h(graph_1, a_1, b_1, N)
  h_2 = compute_h(graph_2, a_2, b_2, N)
  h_3 = compute_h(graph_3, a_3, b_3, N)
  
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

  for i in tqdm(range(num_iter)):
    graph_1_1 = metropolis_step(graph_1_1, h_1, N, 1/N, 1/N)
    graph_1_2 = metropolis_step(graph_1_2, h_1, N, 1/N, 1/N)
    if i % n0 == 0:
      graph_1_1, graph_1_2 = houdayer_step(graph_1_1, graph_1_2)

    graph_2_1 = metropolis_step(graph_2_1, h_2, N, 1/N, 1/N)
    graph_2_2 = metropolis_step(graph_2_2, h_2, N, 1/N, 1/N)
    if i % n0 == 0:
      graph_2_1, graph_2_2 = houdayer_step(graph_2_1, graph_2_2)

    graph_3_1 = metropolis_step(graph_3_1, h_3, N, 1/N, 1/N)
    graph_3_2 = metropolis_step(graph_3_2, h_3, N, 1/N, 1/N)
    if i % n0 == 0:
      graph_3_1, graph_3_2 = houdayer_step(graph_3_1, graph_3_2)

    if i % 15 == 0:
      overlaps_1.append(overlap(x_star, graph_1_1, N))
      overlaps_2.append(overlap(x_star, graph_2_1, N))
      overlaps_3.append(overlap(x_star, graph_3_1, N))
      
      times.append(i * 2)
      clear_output(wait=True)
      
  time_plot_compare_helper(overlaps_1, overlaps_2, overlaps_3, times, N, a_1, b_1, a_2, b_2, a_3, b_3, 'Houdayer Algorithm')


"""b/a ratio"""

# dependence on b/a ration - 1) try different fixed a values for each algo. 1 plot per algorithm (pay attension on d) 2) choose best fixed value for each algo, compare them on the 1 plot

def plot_ratio_overlap_metropolis(N, num_iter):
  ratios = []
  x_star = np.random.choice([-1, 1], N)
  _, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
  for b in tqdm([0.1, 1.0, 5.0]):
    overlaps = []
    ratios = []
    for ratio in range(1, 20):
      ratio = ratio/20
      ratios.append(ratio)
      a = b/ratio
      graph = generate_graph_np(N, x_star, a, b)
      h = compute_h_np(graph, a, b, N)
      over_exp = []
      for i in range(10):
        initial_state = np.random.choice([-1, 1], N)
        over = metropolis_run(initial_state, N, h, x_star, num_iter)
        over_exp.append(over)
      overlaps.append(np.mean(over_exp))

    
    label = 'b = ' + str(b)
    axes.plot(ratios, overlaps, label=label, linewidth=3.0)

  axes.set_ylabel('overlap', fontsize=20)
  axes.set_xlabel('b/a ratio', fontsize=20)
  title = 'Overlap over b/a ratio: Standard Metropolis algorithm'
  axes.set_title(title, fontsize=22, pad=20)
  axes.legend(prop={'size': 20})
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.savefig('plots/ratio_metropolis_list_b.png')

def plot_ratio_overlap_houdayer(N, num_iter, n0 = 1):
  np.random.seed(100)
  x_star = np.random.choice([-1, 1], N)
  _, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
  for b in tqdm([0.1, 1.0, 5.0]):
    ratios = []
    overlaps = []
    for ratio in tqdm(range(1, 20)):
      ratio = ratio/20
      ratios.append(ratio)
      a = b/ratio
      graph = generate_graph(N, x_star, a, b)
      h = compute_h(graph, a, b, N)
      over_exp = []
      for i in range(10):
        initial_state = np.random.choice([-1, 1], N)
        graph_1 = graph.copy()
        nx.set_node_attributes(graph_1, dict(zip(range(N), initial_state)), 'cl')
        graph_2 = graph.copy()
        nx.set_node_attributes(graph_2, dict(zip(range(N), initial_state)), 'cl')
        over1, over2 = houdayer_run(graph_1, graph_2, N, h, x_star, num_iter, n0)
        over_exp.append(np.mean([over1, over2]))
      overlaps.append(np.mean(over_exp))

    label = 'b = ' + str(b)
    axes.plot(ratios, overlaps, label = label, linewidth=3.0)

  axes.set_ylabel('overlap', fontsize=20)
  axes.set_xlabel('b/a ratio', fontsize=20)
  axes.legend(prop={'size': 20})
  if n0 == 1:
    title = 'Overlap over b/a ratio: Houdayer algorithm'
    savename = 'plots/ratio_houndayer_listb.png'
  else:
    title = 'Overlap over b/a ratio: Mixed algorithm, n0 = ' + str(n0)
    savename = 'plots/ratio_houndayer_listb_n0_' + str(n0) + '.png'
  axes.set_title(title, fontsize=22, pad=20)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  # axes.grid()
  plt.savefig(savename)

# dependence on b/a for mixed algo for different n0

"""n0"""

def plot_ratio_n0_houdayer(N, num_iter):
  np.random.seed(100)
  x_star = np.random.choice([-1, 1], N)
  _, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
  for n0 in tqdm([1, 5, 10]):
    ratios = []
    overlaps = []
    for ratio in tqdm(range(1, 20)):
      ratio = ratio/20
      ratios.append(ratio)
      b = 10
      a = b/ratio
      graph = generate_graph(N, x_star, a, b)
      h = compute_h(graph, a, b, N)
      initial_state = np.random.choice([-1, 1], N)
      graph_1 = graph.copy()
      nx.set_node_attributes(graph_1, dict(zip(range(N), initial_state)), 'cl')
      graph_2 = graph.copy()
      nx.set_node_attributes(graph_2, dict(zip(range(N), initial_state)), 'cl')
      over1, over2 = houdayer_run(graph_1, graph_2, N, h, x_star, num_iter, n0)
      overlaps.append(np.mean([over1, over2]))
      
    label = 'n0 = ' + str(n0)
    axes.plot(ratios, overlaps, label = label, linewidth=3.0)

  axes.set_ylabel('overlap', fontsize=20)
  axes.set_xlabel('b/a ratio', fontsize=20)
  axes.legend(prop={'size': 20})
  title = 'Overlap over b/a ratio for different n0'
  savename = 'plots/ratio_mixed_different_n0.png'
  axes.set_title(title, fontsize=22, pad=20)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.savefig(savename)

# dependence over n0

def houdayer_time_plot_compare_n0_helper(overlaps_1, overlaps_2, overlaps_3, times, N, a, b, n0_1, n0_2, n0_3):
  _, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
  axes.set_ylabel('Overlap', fontsize=20)
  axes.set_xlabel('Time', fontsize=20)
  # axes.grid()
  title = 'Overlap over Time: Mixed algorithm with N = ' + str(N) + ', a = ' + str(a) + ', b = ' + str(b)
  axes.set_title(title, fontsize=22, pad=20)
  label_1 = "n0 = " + str(n0_1 - 1)
  label_2 = "n0 = " + str(n0_2 - 1)
  label_3 = "n0 = " + str(n0_3 - 1)
  axes.plot(times, overlaps_1, label = label_1, linewidth=3.0)
  axes.plot(times, overlaps_2, label = label_2, linewidth=3.0)
  axes.plot(times, overlaps_3, label = label_3, linewidth=3.0)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  axes.legend(prop={'size': 20})
  filename = 'plots/overlap_mixed_compare_n0_N' + str(N) + "a" + str(a) + "b" + str(b) + '.png'
  plt.savefig(filename)

def plot_time_overlap_mixed_compare(N, a, b, num_iter, n0_1, n0_2, n0_3):

  x_star = np.random.choice([-1, 1], N)
  graph = generate_graph(N, x_star, a, b)
  h = compute_h(graph, a, b, N)

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

  for i in tqdm(range(num_iter)):
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
      overlaps_1.append(overlap(x_star, graph_1_1, N))
      overlaps_2.append(overlap(x_star, graph_2_1, N))
      overlaps_3.append(overlap(x_star, graph_3_1, N))

      times.append(i)
      
  houdayer_time_plot_compare_n0_helper(overlaps_1, overlaps_2, overlaps_3, times, N, a, b, n0_1, n0_2, n0_3)

"""Plot helpers"""
#---------------------------------------------------------------------------------------------------------------------------------------------------
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

def plot_time_overlap_metropolis(graph, N, a, b, x_star, num_iter):
  '''
  Plot hamiltonian and overlap for 1 metropolis experiment.
  '''
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

def plot_time_overlap_houdayer(graph, N, a, b, x_star, num_iter, n0):
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
    if i % n0 == 0:
      graph_1, graph_2 = houdayer_step(graph_1, graph_2)
    if i % 10 == 0:
      hamiltonians_1.append(hamiltonian(graph_1, h, N))
      hamiltonians_2.append(hamiltonian(graph_2, h, N))
      overlaps_1.append(overlap(x_star, graph_1, N))
      overlaps_2.append(overlap(x_star, graph_2, N))
      clear_output(wait=True)
      houdayer_time_plot_helper(hamiltonians_1, hamiltonians_2, overlaps_1, overlaps_2)

def plot_time_overlap_metropolis_binary(graph, N, a, b, x_star, num_iter):
  '''
  Plot hamiltonian and overlap for 1 metropolis experiment for binary base chain.
  '''
  h = compute_h(graph, a, b, N)

  hamiltonians = []
  overlaps = []
  times = []

  curr_state = np.random.choice([-1, 1], N)
  
  nx.set_node_attributes(graph, dict(zip(range(N), curr_state)), 'cl')

  for i in tqdm(range(num_iter)):
    graph = metropolis_step_binary(graph, h, N)
    #print(nx.get_node_attributes(graph, 'cl'))
    if i % 15 == 0:
      hamiltonians.append(hamiltonian(graph, h, N))
      overlaps.append(overlap(x_star, graph, N))
      times.append(i)
      clear_output(wait=True)
  metropolis_time_plot_helper(hamiltonians, overlaps, times, N, a, b)

def plot_time_overlap_houdayer_binary(graph, N, a, b, x_star, num_iter):
  '''
  Plot hamiltonian and overlap for 1 houdayer experiment for binary base chain.
  '''
  h = compute_h(graph, a, b, N)

  hamiltonians_1 = []
  hamiltonians_2 = []
  overlaps_1 = []
  overlaps_2 = []
  initial_state_1 = np.random.choice([-1, 1], N)
  graph_1 = graph.copy()
  nx.set_node_attributes(graph_1, dict(zip(range(N), initial_state_1)), 'cl')
  initial_state_2 = np.random.choice([-1, 1], N)
  graph_2 = graph.copy()
  nx.set_node_attributes(graph_2, dict(zip(range(N), initial_state_2)), 'cl')
  for i in tqdm(range(num_iter)):
    graph_1 = metropolis_step_binary(graph_1, h, N)
    graph_2 = metropolis_step_binary(graph_2, h, N)
    graph_1, graph_2 = houdayer_step(graph_1, graph_2)
    if i % 10 == 0:
      hamiltonians_1.append(hamiltonian(graph_1, h, N))
      hamiltonians_2.append(hamiltonian(graph_2, h, N))
      overlaps_1.append(overlap(x_star, graph_1, N))
      overlaps_2.append(overlap(x_star, graph_2, N))
      clear_output(wait=True)
      houdayer_time_plot_helper(hamiltonians_1, hamiltonians_2, overlaps_1, overlaps_2)