import sys
sys.path.insert(0, '/Users/spencer.kent/Software_staging_area/quantizers/')

import time
from itertools import product as cartesian_product
import numpy as np
from matplotlib import pyplot as plt

from generalized_lloyd_LBG import compute_quantization as gl
from optimal_generalized_lloyd_LBG import compute_quantization as opt_gl
from utils import plot_1d_assignments as plot_1d

def main():

  ###################################################
  # first we'll visualize different solutions found
  # by the 4 variants that all have similar rates
  ###################################################
  random_laplacian_samps = np.random.laplace(scale=10, size=(50000, 2))
  dummy_data = np.copy(random_laplacian_samps)
  dummy_data[:, 1] = (np.abs(random_laplacian_samps[:, 0]) +
                      random_laplacian_samps[:, 1])
  plt.scatter(dummy_data[:, 0], dummy_data[:, 1], s=1, alpha=0.1)
  plt.title('Original multivariate data')
  plt.xlabel('Component 0')
  plt.ylabel('Component 1')

  # ####################################################
  # # Scalar quantization of each coefficient separately
  NUM_QUANTIZATION_BINS = 8
  init_assignments = np.linspace(
      np.min(dummy_data[:, 0]), np.max(dummy_data[:, 0]), NUM_QUANTIZATION_BINS)

  starttime = time.time()
  gl_assignments_s0, gl_quant_s0, gl_MSE_s0, gl_rate_s0 = gl(
      dummy_data[:, 0], init_assignments)

  init_assignments = np.linspace(
      np.min(dummy_data[:, 1]), np.max(dummy_data[:, 1]), NUM_QUANTIZATION_BINS)
  gl_assignments_s1, gl_quant_s1, gl_MSE_s1, gl_rate_s1 = gl(
      dummy_data[:, 1], init_assignments)
  print("Time to compute separate (suboptimal) scalar quantizations:",
        time.time() - starttime)

  gl_edges_s0 = (gl_assignments_s0[:-1] + gl_assignments_s0[1:]) / 2
  gl_fig_s0 = plot_1d(gl_assignments_s0, gl_edges_s0, dummy_data[:, 0],
                      NUM_QUANTIZATION_BINS*10,
                      title='Suboptimal scalar quantization, component 0')
  gl_edges_s1 = (gl_assignments_s1[:-1] + gl_assignments_s1[1:]) / 2
  gl_fig_s1 = plot_1d(gl_assignments_s1, gl_edges_s1, dummy_data[:, 1],
                      NUM_QUANTIZATION_BINS*10,
                      title='Suboptimal scalar quantization, component 1')

  print('The rate for the suboptimal scalar Lloyd quantizer is',
        (gl_rate_s0 + gl_rate_s1) / 2,
        'bits per component')
  print('The MSE for the suboptial scalar Lloyd quantizer is',
        (gl_MSE_s0 + gl_MSE_s1) / 2,
        'luminace units per component')
  print('===========================')

  #####################################
  # we'll try scalar quantization again
  # but this time with optimal Lloyd
  init_assignments = np.linspace(
      np.min(dummy_data[:, 0]), np.max(dummy_data[:, 0]),
      NUM_QUANTIZATION_BINS+3)
  #^ We'll give ourselves more clusters, but turn up the lambda and these will
  # be pruned down. After some trial and error these settings give us something
  # close to the rate of the non-optimal version
  init_cword_len = (-1. * np.log2(1. / len(init_assignments)) *
                    np.ones((len(init_assignments),)))
  starttime = time.time()
  opt_gl_assignments_s0, opt_gl_quant_s0, opt_gl_MSE_s0, opt_gl_rate_s0 = \
      opt_gl(dummy_data[:, 0], init_assignments,
             init_cword_len, lagrange_mult=0.55)

  init_assignments = np.linspace(
      np.min(dummy_data[:, 1]), np.max(dummy_data[:, 1]),
      NUM_QUANTIZATION_BINS+3)
  opt_gl_assignments_s1, opt_gl_quant_s1, opt_gl_MSE_s1, opt_gl_rate_s1 = \
      opt_gl(dummy_data[:, 1], init_assignments,
             init_cword_len, lagrange_mult=0.55)
  print("Time to compute separate (optimal) scalar quantizations:",
        time.time() - starttime)

  opt_gl_edges_s0 = (opt_gl_assignments_s0[:-1] + opt_gl_assignments_s0[1:]) / 2
  opt_gl_fig_s0 = plot_1d(
      opt_gl_assignments_s0, opt_gl_edges_s0, dummy_data[:, 0],
      NUM_QUANTIZATION_BINS*10,
      title='Optimal scalar quantization, component 0')
  opt_gl_edges_s1 = (opt_gl_assignments_s1[:-1] + opt_gl_assignments_s1[1:]) / 2
  opt_gl_fig_s1 = plot_1d(
      opt_gl_assignments_s1, opt_gl_edges_s1, dummy_data[:, 1],
      NUM_QUANTIZATION_BINS*10,
      title='Optimal scalar quantization, component 1')

  print('The rate for the optimal scalar Lloyd quantizer is',
        (opt_gl_rate_s0 + opt_gl_rate_s1) / 2,
        'bits per component')
  print('The MSE for the optimal scalar Lloyd quantizer is',
        (opt_gl_MSE_s0 + opt_gl_MSE_s1) / 2,
        'luminace units per component')
  print('===========================')

  ##########################################################################
  # Now we use the generalized Lloyd to do joint encoding of both components
  NUM_QUANTIZATION_BINS_EACH_DIM = 7
  # initialization can be important for getting a good tiling - mainly we just
  # don't want any of the initial assignment points to be too extreme. There
  # seems to be support of the pdf out to about 50 in each component so I'm
  # just going to set the extreme bins to something a little more conservative.
  # LBG (1980) discuss a coarse-to-fine splitting method for assigning the
  # initial bins which I'll use in the rate-distortion stuff below
  init_assignments = np.array(list(cartesian_product(
    np.linspace(-30., 30., NUM_QUANTIZATION_BINS_EACH_DIM),
    np.linspace(-30, 30., NUM_QUANTIZATION_BINS_EACH_DIM))))

  starttime = time.time()
  gl_2d_assignments, gl_2d_quant, gl_2d_MSE, gl_2d_rate = gl(
      dummy_data, init_assignments)
  print("Time to compute 2d (suboptimal) vector quantization:",
        time.time() - starttime)

  print('The rate for the suboptimal 2d Lloyd quantizer is',
        gl_2d_rate / 2,
        'bits per component')
  print('The MSE for the suboptimal 2d Lloyd quantizer is',
        gl_2d_MSE / 2,
        'luminace units per component')
  print('===========================')

  plt.figure()
  plt.scatter(dummy_data[:, 0], dummy_data[:, 1], s=1, alpha=0.1)
  plt.scatter(gl_2d_assignments[:, 0], gl_2d_assignments[:, 1], s=5)
  plt.title('2D samples with assignment points, suboptimal 2D Lloyd')

  #######################################################
  # We can compare this to the optimal generalized Lloyd
  NUM_QUANTIZATION_BINS_EACH_DIM = 8
  #^ Again we'll give ourselves more clusters, but turn up the lambda and
  #  these will be pruned down. After some trial and error these settings
  #  give us something close to the rate of the non-optimal version
  init_assignments = np.array(list(cartesian_product(
    np.linspace(-30., 30., NUM_QUANTIZATION_BINS_EACH_DIM),
    np.linspace(-30, 30., NUM_QUANTIZATION_BINS_EACH_DIM))))
  init_cword_len = (-1. * np.log2(1. / len(init_assignments)) *
                    np.ones((len(init_assignments),)))

  starttime = time.time()
  opt_gl_2d_assignments, opt_gl_2d_quant, opt_gl_2d_MSE, opt_gl_2d_rate = \
      opt_gl(dummy_data, init_assignments, init_cword_len, lagrange_mult=0.55)
  print("Time to compute 2d (optimal) vector quantization:",
        time.time() - starttime)

  print('The rate for the optimal 2d Lloyd quantizer is',
        opt_gl_2d_rate / 2,
        'bits per component')
  print('The MSE for the optimal 2d Lloyd quantizer is',
        opt_gl_2d_MSE / 2,
        'luminace units per component')

  plt.figure()
  plt.scatter(dummy_data[:, 0], dummy_data[:, 1], s=1, alpha=0.1)
  plt.scatter(opt_gl_2d_assignments[:, 0], opt_gl_2d_assignments[:, 1], s=5)
  plt.title('2D samples with assignment points, optimal 2D Lloyd')

  plt.show()


  ##########################################################################
  # Okay, now let's sweep out some rate-distortion curves using this dataset
  ##########################################################################
  # suboptimal scalar first
  gl_rates = []
  gl_MSEs = []
  for num_bins in range(16, 4, -2):
    init_assignments = np.linspace(
        np.min(dummy_data[:, 0]), np.max(dummy_data[:, 0]), num_bins)
    gl_assignments_s0, gl_quant_s0, gl_MSE_s0, gl_rate_s0 = gl(
        dummy_data[:, 0], init_assignments)
    init_assignments = np.linspace(
        np.min(dummy_data[:, 1]), np.max(dummy_data[:, 1]), num_bins)
    gl_assignments_s1, gl_quant_s1, gl_MSE_s1, gl_rate_s1 = gl(
        dummy_data[:, 1], init_assignments)
    gl_rates.append((gl_rate_s0 + gl_rate_s1) / 2)
    gl_MSEs.append((gl_MSE_s0 + gl_MSE_s1) / 2)

  # next optimal scalar
  opt_gl_rates = []
  opt_gl_MSEs = []
  num_bins = 20
  for lagrange_w in np.arange(0.2, 1.25, 0.2):
    init_assignments = np.linspace(
        np.min(dummy_data[:, 0]), np.max(dummy_data[:, 0]), num_bins)
    init_cword_len = (-1. * np.log2(1. / len(init_assignments)) *
                      np.ones((len(init_assignments),)))
    opt_gl_assignments_s0, opt_gl_quant_s0, opt_gl_MSE_s0, opt_gl_rate_s0 = \
        opt_gl(dummy_data[:, 0], init_assignments,
               init_cword_len, lagrange_mult=lagrange_w)
    init_assignments = np.linspace(
        np.min(dummy_data[:, 1]), np.max(dummy_data[:, 1]), num_bins)
    opt_gl_assignments_s1, opt_gl_quant_s1, opt_gl_MSE_s1, opt_gl_rate_s1 = \
        opt_gl(dummy_data[:, 1], init_assignments,
               init_cword_len, lagrange_mult=lagrange_w)
    opt_gl_rates.append((opt_gl_rate_s0 + opt_gl_rate_s1) / 2)
    opt_gl_MSEs.append((opt_gl_MSE_s0 + opt_gl_MSE_s1) / 2)

  # next suboptimal generalized Lloyd (2d)
  gl_2d_rates = []
  gl_2d_MSEs = []
  for num_bins in range(12, 4, -2):
    init_assignments = np.array(list(cartesian_product(
      np.linspace(-30., 30., num_bins),
      np.linspace(-30, 30., num_bins))))
    gl_2d_assignments, gl_2d_quant, gl_2d_MSE, gl_2d_rate = gl(
        dummy_data, init_assignments)
    gl_2d_rates.append(gl_2d_rate / 2)
    gl_2d_MSEs.append(gl_2d_MSE / 2)

  # finally, the optimal generalized Lloyd
  opt_gl_2d_rates = []
  opt_gl_2d_MSEs = []
  num_bins = 15
  for lagrange_w in np.arange(0.4, 2.5, 0.4):
    init_assignments = np.array(list(cartesian_product(
      np.linspace(-30., 30., num_bins),
      np.linspace(-30, 30., num_bins))))
    init_cword_len = (-1. * np.log2(1. / len(init_assignments)) *
                      np.ones((len(init_assignments),)))
    opt_gl_2d_assignments, opt_gl_2d_quant, opt_gl_2d_MSE, opt_gl_2d_rate = \
        opt_gl(dummy_data, init_assignments, init_cword_len,
               lagrange_mult=lagrange_w)
    opt_gl_2d_rates.append(opt_gl_2d_rate / 2)
    opt_gl_2d_MSEs.append(opt_gl_2d_MSE / 2)

  plt.plot(gl_MSEs, gl_rates, label='suboptimal scalar Lloyd')
  plt.plot(opt_gl_MSEs, opt_gl_rates, label='optimal scalar Lloyd')
  plt.plot(gl_2d_MSEs, gl_2d_rates, label='suboptimal generalized Lloyd')
  plt.plot(opt_gl_2d_MSEs, opt_gl_2d_rates, label='optimal generalized Lloyd')
  plt.legend()
  # plt.legend(['suboptimal scalar Lloyd', 'optimal scalar Lloyd'])
  plt.title('Rate-distortion performance of 4 variants ' +
            'of Lloyd/LBG quantization')
  plt.xlabel('Distortion (Mean squared error)')
  plt.ylabel('Rate (bits per component)')
  plt.show()

if __name__ == '__main__':
  main()
