"""
Some helper functions useful for visualization
"""
import numpy as np
from matplotlib import pyplot as plt

def plot_1d_assignments(assignment_pts, bin_edges, orig_scalar_samps,
                        plot_bin_num, title='', line_plot=True):
  fig, ax = plt.subplots(1, 1)
  tab10colors = plt.get_cmap('tab10').colors
  counts, histogram_bin_edges = np.histogram(orig_scalar_samps, plot_bin_num)
  empirical_density = counts / np.sum(counts)
  histogram_bin_centers = (histogram_bin_edges[:-1] +
                           histogram_bin_edges[1:]) / 2
  if line_plot:
    ax.plot(histogram_bin_centers, empirical_density,
            color=tab10colors[0], linewidth=1.5)
  else:
    ax.bar(histogram_bin_centers, empirical_density, align='center',
           color=tab10colors[0],
           width=histogram_bin_centers[1]-histogram_bin_centers[0])
  for ap in assignment_pts:
    ax.axvline(x=ap, color=tab10colors[1], linestyle='--')
  for be in bin_edges:
    ax.axvline(x=be, color=tab10colors[1], linestyle='-')

  fig.suptitle(title)
  ax.set_ylabel('Normalized count')
  ax.set_xlabel('Scalar values')
  return fig
