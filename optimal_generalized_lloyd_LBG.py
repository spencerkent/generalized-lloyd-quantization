"""
An implementation of the 'optimal' generalized Lloyd alg

This is sometimes called 'Entropy Constrained Vector Quantization'. I think
it's a bit of a misnomer because the nonoptimal LBG alg still has an implicit
entropy constraint - the number of bins. Instead, here our fitting procedure
takes into account the codeword lengths, which is what makes it 'optimal' and
the way this is done a lagrange multiplier which can be interpreted as
enforcing an entropy constraint on the solution.

This is currently implemented only for the 2-norm-squared error distortion
metric but is fully general for p-norms and the general quadratic form
e^T R e where e is the error vector. The computational or analytical details
of using these other distortion measures may be less desirable than for the
canonical 2-norm-squared distortion metric.

.. [1] Berger, T. (1982). Minimum entropy quantizers and permutation codes.
       IEEE transactions on information theory, 28(2), 149-157.

.. [2] Chou, P. A., Lookabaugh, T., & Gray, R. M. (1989). Entropy-constrained
       vector quantization. IEEE transactions on acoustics, speech, and signal
       processing, 37(1), 31-42.
"""
import copy
import numpy as np
from scipy.spatial.distance import cdist as scipy_distance

def compute_quantization(samples, init_assignment_pts,
                         init_assignment_codeword_lengths,
                         lagrange_mult=1., epsilon=1e-5):
  """
  Implements so-called entropy constrained vector quantization (ECVQ)

  This takes the same basic setup as generalized Lloyd but instead we augment
  the traditional distance metric for computing the partition of the samples
  with a lagrange multiplier term that can be used to effectively constrain
  the total rate of the code assuming that we allocate codewords with
  different lengths to each assignment point. We'll call this our code cost.
  Our policy will be to eliminate assignment points with zero empirical
  probability so that the final number of assignment points may be smaller
  than what we start with. We will also return the quantized samples given
  by the converged quantizer.

  Parameters
  ----------
  samples : ndarray (d, n) or (d,)
      This is an array of d samples of an n-dimensional random variable
      that we wish to find the ECVQ quantizer for. If these are scalar random
      variables, we will accept a 1d array as input.
  init_assignment_pts : ndarray (m, n) or (m,)
      This is an array of some initial guesses for m total assignment points
      for the quantizer. We may prune this list in the optimization procedure
  init_assignment_codeword_lengths : ndarray (m,)
      The starting lengths for each assignment point. This will be changed in
      the first iteration to reflect the empirical probability of this codeword.
      Each components should be in the open interval (0, inf)
  lagrange_mult : float
      This is our knob to set the rate. We might have to sweep it carefully to
      trace out a finely sampled R/D curve.
  epsilon : float, optional
      The tolerance for change in code cost after which we decided we
      have converged. Default 1e-5.

  Returns
  -------
  assignment_pts : ndarray (m, n) or (m,)
      The converged assignment points
  quantized_samples : ndarray (d, n) or (d,)
      The quantized values of the samples. Each should be a n-dimensional
      floating point vector contained in the set of assignment points
  MSE : float
      The mean squared error (the mean l2-normed-squared to be precise) for the
      returned quantization.
  shannon_entropy : float
      The (empirical) Shannon entropy for this code. We can say that assuming
      we use a lossless binary source code, that our expected codeword length
      is precisely this value
  """
  if samples.ndim == 2:
    if samples.shape[1] == 1:
      # we'll just work w/ 1d vectors for the scalar case
      samples = np.squeeze(samples)
      assignment_pts = np.squeeze(init_assignment_pts)
    else:
      # Euclidean distances in N-dimensional spaces are sensitive to
      # the relative variances of each component. We will run the algorithm
      # on "normalized" lengths and then at the end we can 'unnormalize'
      # the values to get back our original space. During the iteration though
      # we will be computing MSE in the normalized space
      saved_component_stds = np.std(samples, axis=0)
      samples = samples / saved_component_stds[None, :]
      assignment_pts = init_assignment_pts / saved_component_stds[None, :]
      lagrange_mult = lagrange_mult / np.sum(saved_component_stds[None, :])
      codeword_lengths = np.copy(init_assignment_codeword_lengths)
  if samples.ndim == 1:
    assignment_pts = np.copy(init_assignment_pts)
    codeword_lengths = np.copy(init_assignment_codeword_lengths)
    lagrange_mult = lagrange_mult * np.std(samples)
    #^ put effective lagrange mult on a sort of normalized scale
    #  with the variance of our samples

  # partition the data into appropriate clusters
  quantized_code, cluster_assignments, assignment_pts, codeword_lengths = \
      greedy_partition(samples, assignment_pts, codeword_lengths, lagrange_mult)
  cword_probs = calculate_assignment_probabilites(cluster_assignments,
                                                  assignment_pts.shape[0])
  if samples.ndim == 1:
    MSE = np.mean(np.square(quantized_code - samples))
  else:
    MSE = np.mean(np.sum(np.square(quantized_code - samples), axis=1))

  shannon_entropy = np.sum(cword_probs * codeword_lengths)
  code_cost = MSE + lagrange_mult * shannon_entropy

  while True:
    old_code_cost = np.copy(code_cost)
    # update the centroids based on the current partition
    for bin_idx in range(assignment_pts.shape[0]):
      binned_samples = samples[cluster_assignments == bin_idx]
      assert len(binned_samples) > 0
      assignment_pts[bin_idx] = np.mean(binned_samples, axis=0)
      # the centroid rule doesn't change from unconstrained LBG

    # partition the data into appropriate clusters
    quantized_code, cluster_assignments, assignment_pts, codeword_lengths = \
        greedy_partition(samples, assignment_pts,
                         codeword_lengths, lagrange_mult)

    # given the current assignments compute the (asymptotically)
    # optimal codeword lengths
    cword_probs = calculate_assignment_probabilites(cluster_assignments,
                                                    assignment_pts.shape[0])
    codeword_lengths = -1 * np.log2(cword_probs)

    if samples.ndim == 1:
      MSE = np.mean(np.square(quantized_code - samples))
    else:
      MSE = np.mean(np.sum(np.square(quantized_code - samples), axis=1))
    shannon_entropy = np.sum(cword_probs * codeword_lengths)
    code_cost = MSE + lagrange_mult * shannon_entropy

    if not np.isclose(old_code_cost, code_cost):
      assert code_cost <= old_code_cost, 'uh-oh, code cost increased'

    if old_code_cost == 0.0:  # avoid divide by zero below
      break
    if (np.abs(old_code_cost - code_cost) / old_code_cost) < epsilon:
      break
      #^ this algorithm provably reduces this cost or leaves it unchanged at
      #  each iteration so the boundedness of this cost means this is a
      #  valid stopping criterion

  if samples.ndim != 1:
    # we want to return the code and quantization in the original space
    quantized_code = quantized_code * saved_component_stds[None, :]
    assignment_pts = assignment_pts * saved_component_stds[None, :]
    samples = samples * saved_component_stds[None, :]
    # compute the MSE of the quantized code in the original space
    MSE = np.mean(np.sum(np.square(quantized_code - samples), axis=1))

  return assignment_pts, quantized_code, MSE, shannon_entropy


def quantize(raw_vals, assignment_vals, codeword_lengths, l_weight,
             return_cluster_assignments=False):

  assert len(assignment_vals) == len(codeword_lengths)
  # I could not easily find an implementation of nearest neighbors that would
  # use a generalized cost function rather than the l2-norm-squared to assign
  # the partition. Therefore, we'll (for now) calculate the cost of assigning
  # each point to each interval and then just take the minimum.
  if raw_vals.ndim == 1:
    l2_distance = np.square(scipy_distance(raw_vals[:, None],
                            assignment_vals[:, None], metric='euclidean'))
  else:
    l2_distance = np.square(scipy_distance(raw_vals,
                            assignment_vals, metric='euclidean'))
  assignment_cost = l2_distance + l_weight * codeword_lengths[None, :]
  c_assignments = np.argmin(assignment_cost, axis=1)

  if return_cluster_assignments:
    return assignment_vals[c_assignments], c_assignments
  else:
    return assignment_vals[c_assignments]


def greedy_partition(raw_vals, a_vals, c_lengths, l_weight):
  """
  Partition the data according to the assignment values.

  If there are clusters with no data, we will DROP these clusters. According
  to Chou et al. (1989) the splitting procedure we used before is not
  well motivated in this case and we will pay no coding cost by
  removing this cluster, so that will be our policy here.
  """
  fresh_a_vals = np.copy(a_vals)
  fresh_c_lengths = np.copy(c_lengths)
  quant_code, c_assignments = quantize(raw_vals, fresh_a_vals,
                                       fresh_c_lengths, l_weight, True)
  cword_probs = calculate_assignment_probabilites(c_assignments,
                                                  fresh_a_vals.shape[0])
  if not np.any(cword_probs == 0):
    return quant_code, c_assignments, fresh_a_vals, fresh_c_lengths
  else:
    nonzero_prob_pts = np.where(cword_probs != 0)
    # Clusters w/ zero empirical probability are problematic (they aren't
    # necessarily in suboptimal LBG) because this will cause the codeword
    # lengths to be infinity, thus no points will ever get assigned to
    # these clusters. Therefore following the advice of Chou et al. (1989) we will
    # drop these clusters.
    fresh_a_vals = fresh_a_vals[nonzero_prob_pts]
    fresh_c_lengths = fresh_c_lengths[nonzero_prob_pts]
    # the indexes of c_assignments will be stale now so we recompute...
    quant_code, c_assignments = quantize(raw_vals, fresh_a_vals,
                                         fresh_c_lengths, l_weight, True)
    return quant_code, c_assignments, fresh_a_vals, fresh_c_lengths


def calculate_assignment_probabilites(assignments, num_clusters):
  temp = np.arange(num_clusters)
  hist_b_edges = np.hstack([-np.inf, (temp[:-1] + temp[1:]) / 2, np.inf])
  assignment_counts, _ = np.histogram(assignments, hist_b_edges)
  empirical_density = assignment_counts / np.sum(assignment_counts)
  return empirical_density
