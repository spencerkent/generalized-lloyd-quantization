"""
Some helper functions useful for visualization
"""
import numpy as np
from matplotlib import pyplot as plt

def plot_1d_and_2d_assignments(assignment_pts, orig_samps, assignment_inds,
                               quant_scheme, plot_bin_num=100,
                               title='', line_plot=True):
  """
  Visualize the assignment points along with the underlying data

  We can visualize the assignment points, the resulting empirical
  probability mass function, and the noise introduced by quantization

  Parameters
  ----------
  assignment_pts : ndarray (m, n) or (m,)
      The n-dimensional assignment points used to apply quantization
  orig_samps : ndarray (d, n) or (d,)
      The original n-dimensional data points
  assignment_inds : ndarray (d, )
      For each sample, the index of the codeword to which quantization
      assigns this datapoint.
  quant_scheme : str
      The scheme we use to allocate the assignment points. This will usually
      imply bin edges that are equidistant between assignment points, but will
      be a little different for the 'optimal' version of Lloyd's algorithm. One
      of {'uniform_scalar', 'lloyd_scalar', 'optimal_lloyd_scalar',
      'uniform_vector', 'lloyd_vector', 'optimal_lloyd_vector'}
  plot_bin_num : int, optional
      The number of histogram bins we use in the plot visualization. Default 100
  title : str, optional
      The title of the plot
  line_plot : bool, optional
      Only checked when data is 1d, otherwise ignored. If true, display the
      empirical density with a line plot. Otherwise display it with a bar plot

  Returns
  -------
  assignment_fig : pyplot.figure
      A figure reference we can use to save the figure in the calling scope
  """
  if assignment_pts.ndim == 2:
    assert assignment_pts.shape[1] == 2  # for now just visualize 2d RVs
    assert orig_samps.shape[1] == 2
  assert quant_scheme in ['uniform_scalar', 'lloyd_scalar',
                          'optimal_lloyd_scalar', 'uniform_vector',
                          'lloyd_vector', 'optimal_lloyd_vector']

  tab10colors = plt.get_cmap('tab10').colors
  blues = plt.get_cmap('Blues')
  fig, ax = plt.subplots(1, 3, figsize=(24, 7))
  fig.suptitle(title, fontsize=20)
  ###########################################################################
  # plot the probability density the data with assignment points superimposed
  if assignment_pts.ndim == 1:
    counts, histogram_bin_edges = np.histogram(orig_samps, plot_bin_num)
    empirical_density = counts / np.sum(counts)
    histogram_bin_centers = (histogram_bin_edges[:-1] +
                             histogram_bin_edges[1:]) / 2
    if line_plot:
      ax[0].plot(histogram_bin_centers, empirical_density,
                 color=tab10colors[0], linewidth=1.5, label='Estimated density')
    else:
      ax[0].bar(histogram_bin_centers, empirical_density, align='center',
                color=tab10colors[0], label='Estimated density',
                width=histogram_bin_centers[1]-histogram_bin_centers[0])
    for ap_idx in range(len(assignment_pts)):
      ax[0].axvline(x=assignment_pts[ap_idx], color=tab10colors[1],
          linestyle='-', label='Assignment points' if ap_idx == 0 else '')
    if quant_scheme in ['uniform_scalar', 'lloyd_scalar']:
      # the bin edges are equidistant between assignment points
      bin_edges = (assignment_pts[:-1] + assignment_pts[1:]) / 2
      for be_idx in range(len(bin_edges)):
        ax[0].axvline(x=bin_edges[be_idx], color=tab10colors[1], linestyle='--',
            label='Bin edges' if be_idx == 0 else '')
    else:
      # the bin edges are determined by the code-length-augmented distance.
      # I'm not sure if these are guaranteed to be non-overlapping for adjacent
      # bins, so I'm leaving this blank until I decide how to plot these
      pass
    ax[0].legend(fontsize=12)
    ax[0].set_title('Empirical data density w/ assignment points', fontsize=15)
    ax[0].set_ylabel('Histogram-based density estimate', fontsize=12)
    ax[0].set_xlabel('Scalar values', fontsize=12)
  else:
    # this is a 2D plot. We'll plot the log probabilities of histogram bins
    counts, hist_bin_edges = np.histogramdd(orig_samps, plot_bin_num)
    empirical_density = counts / np.sum(counts)
    log_density = np.copy(empirical_density)
    nonzero_inds = log_density != 0
    log_density[nonzero_inds] = np.log(log_density[nonzero_inds])
    log_density[nonzero_inds] -= np.min(log_density[nonzero_inds])
    log_density[nonzero_inds] /= np.max(log_density[nonzero_inds])
    hist_bin_centers = [(hist_bin_edges[x][:-1] + hist_bin_edges[x][1:]) / 2
                        for x in range(len(hist_bin_edges))]
    min_max_x = [hist_bin_centers[0][0], hist_bin_centers[0][-1]]
    min_max_y = [hist_bin_centers[1][0], hist_bin_centers[1][-1]]
    ax[0].imshow(np.flip(log_density.T, axis=0),
                 interpolation='nearest', cmap='Blues')

    def transform_coords_to_image_space(scalar_coords, min_max, which_axis):
      rescaled = (plot_bin_num *
          (scalar_coords - min_max[0]) / (min_max[1] - min_max[0]))
      if which_axis == 'x':
        return rescaled
      else:
        return plot_bin_num - rescaled

    apts_x_img_coordinates = transform_coords_to_image_space(
      assignment_pts[:, 0], min_max_x, 'x')
    apts_y_img_coordinates = transform_coords_to_image_space(
      assignment_pts[:, 1], min_max_y, 'y')
    ax[0].scatter(apts_x_img_coordinates, apts_y_img_coordinates,
                  color=tab10colors[1], s=5)
    ax[0].set_title('Empirical (log) density of data w/ assignment points',
                    fontsize=15)
    ax[0].set_ylabel('Values for coefficient 1', fontsize=12)
    ax[0].set_xlabel('Values for coefficient 0', fontsize=12)
    ax[0].set_xlim([0, plot_bin_num])
    ax[0].set_ylim([plot_bin_num, 0])
    ax[0].set_aspect('equal')

  ##############################################
  # plot the probability mass of the assignments
  empirical_code_PMF = calculate_assignment_probabilites(
      assignment_inds, assignment_pts.shape[0])
  assert np.isclose(np.sum(empirical_code_PMF), 1.0)
  nonzero_prob_pts = np.where(empirical_code_PMF != 0)  # avoid log2(0)
  shannon_entropy = -1 * np.sum(empirical_code_PMF[nonzero_prob_pts] *
                                np.log2(empirical_code_PMF[nonzero_prob_pts]))
  log_PMF = np.copy(empirical_code_PMF)
  nonzero_inds = log_PMF != 0
  log_PMF[nonzero_inds] = np.log(log_PMF[nonzero_inds])
  log_PMF[nonzero_inds] -= np.min(log_PMF[nonzero_inds])
  log_PMF[nonzero_inds] /= np.max(log_PMF[nonzero_inds])

  if assignment_pts.ndim == 1:
    mline, stemline, baseline = ax[1].stem(assignment_pts, empirical_code_PMF,
                                           linefmt='-', markerfmt='o')
    plt.setp(stemline, 'color', tab10colors[1])
    plt.setp(mline, 'color', tab10colors[1])
    ax[1].text(0.98, 0.98,
        'Entropy of code: {:.2f} bits'.format(shannon_entropy),
        horizontalalignment='right', verticalalignment='top',
        transform=ax[1].transAxes, color=tab10colors[4], fontsize=15)
    ax[1].set_title('Empirical PMF of the code', fontsize=15)
    ax[1].set_ylabel('Histogram-based probability mass estimate', fontsize=12)
    ax[1].set_xlabel('Scalar values', fontsize=12)
  else:
    for ap_idx in range(len(assignment_pts)):
      ax[1].scatter(assignment_pts[ap_idx, 0], assignment_pts[ap_idx, 1], s=50,
          color=blues(log_PMF[ap_idx]))
    ax[1].text(0.98, 0.98,
        'Entropy of code: {:.2f} bits'.format(shannon_entropy),
        horizontalalignment='right', verticalalignment='top',
        transform=ax[1].transAxes, color=tab10colors[4], fontsize=15)
    ax[1].set_xlim(min_max_x)
    ax[1].set_ylim(min_max_y)
    ax[1].set_title('Empirical (log) PMF of the code', fontsize=15)
    ax[1].set_ylabel('Values for coefficient 1', fontsize=12)
    ax[1].set_xlabel('Values for coefficient 0', fontsize=12)
    manual_aspect_ratio = ((min_max_x[1] - min_max_x[0]) /
                           (min_max_y[1] - min_max_y[0]))
    ax[1].set_aspect(manual_aspect_ratio)
    plot1_ticklabels_x = ax[1].get_xticks()[1:-1].astype('int')
    plot1_ticklabels_y = ax[1].get_yticks()[1:-1].astype('int')

    ax[0].set_xticks(transform_coords_to_image_space(plot1_ticklabels_x,
      min_max_x, 'x'))
    ax[0].set_yticks(transform_coords_to_image_space(plot1_ticklabels_y,
      min_max_y, 'y'))
    ax[0].set_xticklabels(plot1_ticklabels_x)
    ax[0].set_yticklabels(plot1_ticklabels_y)

  ###################################################################
  # plot the probability density of the noise induced by quantization
  quantized_values = assignment_pts[assignment_inds]
  quant_noise = quantized_values - orig_samps
  if assignment_pts.ndim == 1:
    counts, histogram_bin_edges = np.histogram(quant_noise, plot_bin_num)
    empirical_density = counts / np.sum(counts)
    histogram_bin_centers = (histogram_bin_edges[:-1] +
                             histogram_bin_edges[1:]) / 2
    if line_plot:
      ax[2].plot(histogram_bin_centers, empirical_density,
                 color=tab10colors[2], linewidth=1.5, label='Estimated density')
    else:
      ax[2].bar(histogram_bin_centers, empirical_density, align='center',
                color=tab10colors[2], label='Estimated density',
                width=histogram_bin_centers[1]-histogram_bin_centers[0])
    ax[2].legend(fontsize=12)
    ax[2].set_title('Empirical density of quantization noise', fontsize=15)
    ax[2].set_ylabel('Histogram-based density estimate', fontsize=12)
    ax[2].set_xlabel('Scalar values', fontsize=12)
  else:
    # this is a 2D plot. We'll plot the log probabilities of histogram bins
    counts, hist_bin_edges = np.histogramdd(quant_noise, plot_bin_num)
    empirical_density = counts / np.sum(counts)
    log_density = np.copy(empirical_density)
    nonzero_inds = log_density != 0
    log_density[nonzero_inds] = np.log(log_density[nonzero_inds])
    log_density[nonzero_inds] -= np.min(log_density[nonzero_inds])
    log_density[nonzero_inds] /= np.max(log_density[nonzero_inds])
    hist_bin_centers = [(hist_bin_edges[x][:-1] + hist_bin_edges[x][1:]) / 2
                        for x in range(len(hist_bin_edges))]
    min_max_x = [hist_bin_centers[0][0], hist_bin_centers[0][-1]]
    min_max_y = [hist_bin_centers[1][0], hist_bin_centers[1][-1]]
    ax[2].imshow(np.flip(log_density.T, axis=0),
                 interpolation='nearest', cmap='Blues')
    ax[2].set_xticks(np.linspace(0, plot_bin_num, 5))
    ax[2].set_yticks(np.linspace(0, plot_bin_num, 5))
    xaxis_labels = ['%.1f' % x for x in
                    np.linspace(min_max_x[0], min_max_x[1], 5)]
    yaxis_labels = ['%.1f' % x for x in
                    np.linspace(min_max_y[-1], min_max_y[0], 5)]
    ax[2].set_xticklabels(xaxis_labels)
    ax[2].set_yticklabels(yaxis_labels)
    ax[2].set_title('Empirical (log) density of quantization noise',
                    fontsize=15)
    ax[2].set_ylabel('Values for coefficient 1', fontsize=12)
    ax[2].set_xlabel('Values for coefficient 0', fontsize=12)
    ax[2].set_xlim([0, plot_bin_num])
    ax[2].set_ylim([plot_bin_num, 0])
    ax[2].set_aspect('equal')

  return fig

def calculate_assignment_probabilites(assignments, num_clusters):
  """
  Just counts the occurence of each assignment to get an empirical PMF estimate
  """
  temp = np.arange(num_clusters)
  hist_b_edges = np.hstack([-np.inf, (temp[:-1] + temp[1:]) / 2, np.inf])
  assignment_counts, _ = np.histogram(assignments, hist_b_edges)
  empirical_density = assignment_counts / np.sum(assignment_counts)
  return empirical_density

