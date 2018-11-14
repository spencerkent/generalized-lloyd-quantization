import sys
import os
demo_dir_fullpath = os.path.dirname(os.path.abspath(__file__))
toplevel_dir_fullpath = demo_dir_fullpath[:demo_dir_fullpath.rfind('/')+1]
sys.path.insert(0, toplevel_dir_fullpath)

import time
import numpy as np
from matplotlib import pyplot as plt

from null_uniform import compute_quantization as uni
from generalized_lloyd_LBG import compute_quantization as gl
from optimal_generalized_lloyd_LBG import compute_quantization as opt_gl

from utils.plotting import plot_1d_and_2d_assignments

def main():

  #############################################################################
  # first we'll visualize different solutions found uniform scalar quantization
  # as well as by the 4 variants of Lloyd that all have similar rates
  #############################################################################
  random_laplacian_samps = np.random.laplace(scale=10, size=(50000, 2))
  dummy_data = np.copy(random_laplacian_samps)
  dummy_data[:, 1] = (np.abs(random_laplacian_samps[:, 0]) +
                      random_laplacian_samps[:, 1])

  ############################################################
  # Uniform scalar quantization of each coefficient separately
  BINWIDTH_COMPONENT_0 = 11.
  BINWIDTH_COMPONENT_1 = 11.

  starttime = time.time()
  uni_apts_s0, uni_assignments_s0, uni_MSE_s0, uni_rate_s0 = uni(
      dummy_data[:, 0], BINWIDTH_COMPONENT_0, placement_scheme='on_mode')
  uni_apts_s1, uni_assignments_s1, uni_MSE_s1, uni_rate_s1 = uni(
      dummy_data[:, 1], BINWIDTH_COMPONENT_1, placement_scheme='on_mode')
  print("Time to compute uniform scalar quantizations:",
        time.time() - starttime)

  uni_fig_s0 = plot_1d_and_2d_assignments(
      uni_apts_s0, dummy_data[:, 0], uni_assignments_s0, 'uniform_scalar',
      100, title='Uniform scalar quantization, component 0')
  uni_fig_s1 = plot_1d_and_2d_assignments(
      uni_apts_s1, dummy_data[:, 1], uni_assignments_s1, 'uniform_scalar',
      100, title='Uniform scalar quantization, component 1')
  print('The rate for the uniform scalar quantizer is',
        (uni_rate_s0 + uni_rate_s1) / 2, 'bits per component')
  print('The MSE for the uniform scalar quantizer is',
        (uni_MSE_s0 + uni_MSE_s1) / 2, 'luminace units per component')
  print('===========================')

  def get_init_assignments_for_lloyd(data, the_binwidths):
    # Lloyd can run into trouble if the most extreme assignment points are
    # larger in magnitude than the most extreme datapoints, which can happen
    # with the uniform quantization, so we just get rid of those initial points.
    assgnmnts, _, _, _ = uni(data, the_binwidths, placement_scheme='on_mode')
    min_data = np.min(data, axis=0)
    max_data = np.max(data, axis=0)
    if data.ndim == 1:
      assgnmnts = np.delete(assgnmnts, np.where(assgnmnts < 0.9*min_data))
      assgnmnts = np.delete(assgnmnts, np.where(assgnmnts > 0.9*max_data))
      return assgnmnts
    else:
      assgnmnts = np.delete(assgnmnts,
          np.where(assgnmnts[:, 0] < 0.9*min_data[0]), axis=0)
      assgnmnts = np.delete(assgnmnts,
          np.where(assgnmnts[:, 0] > 0.9*max_data[0]), axis=0)
      assgnmnts = np.delete(assgnmnts,
          np.where(assgnmnts[:, 1] < 0.9*min_data[1]), axis=0)
      assgnmnts = np.delete(assgnmnts,
          np.where(assgnmnts[:, 1] > 0.9*max_data[1]), axis=0)
      return assgnmnts

  ##########################################################
  # Lloyd scalar quantization of each coefficient separately
  INIT_BW_C0 = 27  # we need way fewer in this case, bigger starting bins
  INIT_BW_C1 = 27
  starttime = time.time()
  init_assignments = get_init_assignments_for_lloyd(
      dummy_data[:, 0], INIT_BW_C0)
  gl_apts_s0, gl_assignments_s0, gl_MSE_s0, gl_rate_s0 = gl(
      dummy_data[:, 0], init_assignments)

  init_assignments = get_init_assignments_for_lloyd(
      dummy_data[:, 1], INIT_BW_C1)
  gl_apts_s1, gl_assignments_s1, gl_MSE_s1, gl_rate_s1 = gl(
      dummy_data[:, 1], init_assignments)
  print("Time to compute separate (suboptimal) scalar quantizations:",
        time.time() - starttime)

  gl_fig_s0 = plot_1d_and_2d_assignments(
      gl_apts_s0, dummy_data[:, 0], gl_assignments_s0, 'lloyd_scalar',
      100, title='Suboptimal Lloyd scalar quantization, component 0')
  gl_fig_s1 = plot_1d_and_2d_assignments(
      gl_apts_s1, dummy_data[:, 1], gl_assignments_s1, 'lloyd_scalar',
      100, title='Suboptimal Lloyd scalar quantization, component 1')
  print('The rate for the suboptimal scalar Lloyd quantizer is',
        (gl_rate_s0 + gl_rate_s1) / 2, 'bits per component')
  print('The MSE for the suboptial scalar Lloyd quantizer is',
        (gl_MSE_s0 + gl_MSE_s1) / 2, 'luminace units per component')
  print('===========================')


  #############################################################################
  # we'll try Lloyd scalar quantization again but this time the optimal version
  INIT_BW_C0 = 20
  INIT_BW_C1 = 20
  #^ We'll give ourselves more clusters, but turn up the lambda and these will
  # be pruned down. After some trial and error these settings give us something
  # close to the rate of the non-optimal version
  starttime = time.time()
  init_assignments = get_init_assignments_for_lloyd(
      dummy_data[:, 0], INIT_BW_C0)
  init_cword_len = (-1. * np.log2(1. / len(init_assignments)) *
                    np.ones((len(init_assignments),)))
  opt_gl_apts_s0, opt_gl_assignments_s0, opt_gl_MSE_s0, opt_gl_rate_s0 = \
      opt_gl(dummy_data[:, 0], init_assignments,
             init_cword_len, lagrange_mult=0.6)

  init_assignments = get_init_assignments_for_lloyd(
      dummy_data[:, 1], INIT_BW_C1)
  init_cword_len = (-1. * np.log2(1. / len(init_assignments)) *
                    np.ones((len(init_assignments),)))
  opt_gl_apts_s1, opt_gl_assignments_s1, opt_gl_MSE_s1, opt_gl_rate_s1 = \
      opt_gl(dummy_data[:, 1], init_assignments,
             init_cword_len, lagrange_mult=0.6)
  print("Time to compute separate (optimal) scalar quantizations:",
        time.time() - starttime)

  opt_gl_fig_s0 = plot_1d_and_2d_assignments(
      opt_gl_apts_s0, dummy_data[:, 0], opt_gl_assignments_s0,
      'optimal_lloyd_scalar', 100,
      title='Optimal Lloyd scalar quantization, component 0')
  opt_gl_fig_s1 = plot_1d_and_2d_assignments(
      opt_gl_apts_s1, dummy_data[:, 1], opt_gl_assignments_s1,
      'optimal_lloyd_scalar', 100,
      title='Optimal Lloyd scalar quantization, component 1')
  print('The rate for the optimal scalar Lloyd quantizer is',
        (opt_gl_rate_s0 + opt_gl_rate_s1) / 2, 'bits per component')
  print('The MSE for the optimal scalar Lloyd quantizer is',
        (opt_gl_MSE_s0 + opt_gl_MSE_s1) / 2, 'luminace units per component')
  print('===========================')


  # ##########################################
  # Now we can try Uniform VECTOR quantization
  BINWIDTHS = np.array([10., 10.])
  starttime = time.time()
  uni_2d_apts, uni_2d_assignments, uni_2d_MSE, uni_2d_rate = uni(
      dummy_data, BINWIDTHS, placement_scheme='on_mode')
  print("Time to compute uniform vector quantizations:",
        time.time() - starttime)

  uni_2d_fig_s0 = plot_1d_and_2d_assignments(
      uni_2d_apts, dummy_data, uni_2d_assignments, 'uniform_vector',
      100, title='Uniform vector quantization')
  print('The rate for the uniform vector quantizer is',
        uni_2d_rate / 2, 'bits per component')
  print('The MSE for the uniform vector quantizer is',
        uni_2d_MSE / 2, 'luminace units per component')
  print('===========================')


  ##########################################################################
  # Now we use the generalized Lloyd to do joint encoding of both components
  BINWIDTHS = np.array([29., 30.])
  starttime = time.time()
  init_assignments = get_init_assignments_for_lloyd(dummy_data, BINWIDTHS)
  gl_2d_apts, gl_2d_assignments, gl_2d_MSE, gl_2d_rate = gl(
      dummy_data, init_assignments)
  print("Time to compute 2d (suboptimal) vector quantization:",
        time.time() - starttime)

  gl_2d_fig = plot_1d_and_2d_assignments(
      gl_2d_apts, dummy_data, gl_2d_assignments,
      'lloyd_vector', 100,
      title='Generalized (vector) Lloyd quantization')
  print('The rate for the suboptimal 2d Lloyd quantizer is',
        gl_2d_rate / 2, 'bits per component')
  print('The MSE for the suboptimal 2d Lloyd quantizer is',
        gl_2d_MSE / 2, 'luminace units per component')
  print('===========================')


  #######################################################
  # We can compare this to the optimal generalized Lloyd
  BINWIDTHS = np.array([20, 20])
  starttime = time.time()
  init_assignments = get_init_assignments_for_lloyd(dummy_data, BINWIDTHS)
  init_cword_len = (-1. * np.log2(1. / len(init_assignments)) *
                    np.ones((len(init_assignments),)))

  opt_gl_2d_apts, opt_gl_2d_assignments, opt_gl_2d_MSE, opt_gl_2d_rate = \
      opt_gl(dummy_data, init_assignments, init_cword_len, lagrange_mult=0.6)
  print("Time to compute 2d (optimal) vector quantization:",
        time.time() - starttime)

  opt_gl_2d_fig = plot_1d_and_2d_assignments(
      opt_gl_2d_apts, dummy_data, opt_gl_2d_assignments,
      'optimal_lloyd_vector', 100,
      title='Optimal generalized (vector) Lloyd quantization')
  print('The rate for the optimal 2d Lloyd quantizer is',
        opt_gl_2d_rate / 2, 'bits per component')
  print('The MSE for the optimal 2d Lloyd quantizer is',
        opt_gl_2d_MSE / 2, 'luminace units per component')

  plt.show()

  ##########################################################################
  # Okay, now let's sweep out some rate-distortion curves using this dataset
  ##########################################################################
  # uniform scalar first
  uni_rates = []
  uni_MSEs = []
  for binwidth in np.linspace(8, 16, 10):
    _, _, uni_MSE_s0, uni_rate_s0 = uni(
         dummy_data[:, 0], binwidth, placement_scheme='on_mode')
    _, _, uni_MSE_s1, uni_rate_s1 = uni(
         dummy_data[:, 1], binwidth, placement_scheme='on_mode')
    uni_rates.append((uni_rate_s0 + uni_rate_s1) / 2)
    uni_MSEs.append((uni_MSE_s0 + uni_MSE_s1) / 2)

  # suboptimal scalar
  gl_rates = []
  gl_MSEs = []
  for binwidth in np.linspace(20, 35, 10):
    init_assignments = get_init_assignments_for_lloyd(
        dummy_data[:, 0], binwidth)
    _, _, gl_MSE_s0, gl_rate_s0 = gl(dummy_data[:, 0], init_assignments)
    init_assignments = get_init_assignments_for_lloyd(
        dummy_data[:, 1], binwidth)
    _, _, gl_MSE_s1, gl_rate_s1 = gl(dummy_data[:, 1], init_assignments)
    gl_rates.append((gl_rate_s0 + gl_rate_s1) / 2)
    gl_MSEs.append((gl_MSE_s0 + gl_MSE_s1) / 2)

  # next optimal scalar
  opt_gl_rates = []
  opt_gl_MSEs = []
  binwidth = 20
  for lagrange_w in np.linspace(0.4, 1.75, 10):
    print(lagrange_w)
    init_assignments = get_init_assignments_for_lloyd(
        dummy_data[:, 0], binwidth)
    init_cword_len = (-1. * np.log2(1. / len(init_assignments)) *
                      np.ones((len(init_assignments),)))
    _, _, opt_gl_MSE_s0, opt_gl_rate_s0 = opt_gl(dummy_data[:, 0],
        init_assignments, init_cword_len, lagrange_mult=lagrange_w)
    init_assignments = get_init_assignments_for_lloyd(
        dummy_data[:, 1], binwidth)
    init_cword_len = (-1. * np.log2(1. / len(init_assignments)) *
                      np.ones((len(init_assignments),)))
    _, _, opt_gl_MSE_s1, opt_gl_rate_s1 = opt_gl(dummy_data[:, 1],
        init_assignments, init_cword_len, lagrange_mult=lagrange_w)
    opt_gl_rates.append((opt_gl_rate_s0 + opt_gl_rate_s1) / 2)
    opt_gl_MSEs.append((opt_gl_MSE_s0 + opt_gl_MSE_s1) / 2)

  # next uniform vector (2d)
  uni_2d_rates = []
  uni_2d_MSEs = []
  for binwidth in np.linspace(8, 16, 10):
    _, _, uni_2d_MSE, uni_2d_rate = uni(
         dummy_data, np.array([binwidth, binwidth]), placement_scheme='on_mode')
    uni_2d_rates.append(uni_2d_rate / 2)
    uni_2d_MSEs.append(uni_2d_MSE / 2)

  # next suboptimal generalized Lloyd (2d)
  gl_2d_rates = []
  gl_2d_MSEs = []
  for binwidth in np.linspace(27, 48, 5):
    print('Lloyd 2d')
    init_assignments = get_init_assignments_for_lloyd(
        dummy_data, np.array([binwidth, binwidth]))
    _, _, gl_2d_MSE, gl_2d_rate = gl(dummy_data, init_assignments)
    gl_2d_rates.append(gl_2d_rate / 2)
    gl_2d_MSEs.append(gl_2d_MSE / 2)

  # finally, the optimal generalized Lloyd
  opt_gl_2d_rates = []
  opt_gl_2d_MSEs = []
  binwidth = 22
  for lagrange_w in np.linspace(0.4, 2.5, 5):
    print(lagrange_w)
    init_assignments = get_init_assignments_for_lloyd(
        dummy_data, np.array([binwidth, binwidth]))
    init_cword_len = (-1. * np.log2(1. / len(init_assignments)) *
                      np.ones((len(init_assignments),)))
    _, _, opt_gl_2d_MSE, opt_gl_2d_rate = opt_gl(
        dummy_data, init_assignments, init_cword_len, lagrange_mult=lagrange_w)
    opt_gl_2d_rates.append(opt_gl_2d_rate / 2)
    opt_gl_2d_MSEs.append(opt_gl_2d_MSE / 2)

  plt.figure(figsize=(20, 20))
  plt.plot(uni_MSEs, uni_rates, label='Uniform Scalar', linewidth=4)
  plt.plot(gl_MSEs, gl_rates, label='Suboptimal Scalar Lloyd', linewidth=4)
  plt.plot(opt_gl_MSEs, opt_gl_rates, label='Optimal Scalar Lloyd', linewidth=4)
  plt.plot(uni_2d_MSEs, uni_2d_rates, label='Uniform 2D', linewidth=4)
  plt.plot(gl_2d_MSEs, gl_2d_rates, label='Suboptimal 2D Lloyd', linewidth=4)
  plt.plot(opt_gl_2d_MSEs, opt_gl_2d_rates, label='Optimal 2D Lloyd',
           linewidth=4)
  plt.legend(fontsize=15)
  plt.title('Rate-distortion performance of 4 variants ' +
            'of Lloyd/LBG\nplus 2 variants of uniform quantization', fontsize=20)
  plt.xlabel('Distortion (Mean squared error)', fontsize=15)
  plt.ylabel('Rate (bits per component)', fontsize=15)

  plt.show()

if __name__ == '__main__':
  main()
