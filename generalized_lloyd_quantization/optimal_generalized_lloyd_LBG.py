"""
An implementation of the 'optimal' generalized Lloyd algorithm

This is sometimes called 'Entropy Constrained Vector Quantization'. Instead
of trying to find assignment points that just minimize the mean-squared error
of the quantization, our fitting procedure also takes into account the codeword
lengths, via a lagrange multiplier which can be interpreted as (loosely)
enforcing an entropy constraint on the solution. This extra sensitivity of the
quantization to the actual entropy of the resulting code is what makes this
method 'optimal'.

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
  than what we start with.

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
      Each of the components should be in the open interval (0, inf)
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
  cluster_assignments : ndarray (d, )
      For each sample, the index of the codeword to which optimal Lloyd
      quantization assigns this datapoint. We can compute the actual quantized
      values outside this function by invoking
      `assignment_pts[cluster_assignments]`
  MSE : float
      The mean squared error (the mean l2-normed-squared to be precise) for the
      returned quantization.
  shannon_entropy : float
      The (empirical) Shannon entropy for this code. We can say that assuming
      we use a lossless binary source code, that our expected codeword length
      is precisely this value
  """
  # get rid of the original references to make sure we don't modify the data
  # in the calling scope
  samples = np.copy(samples)
  assignment_pts = np.copy(init_assignment_pts)
  codeword_lengths = np.copy(init_assignment_codeword_lengths)
  if samples.ndim == 2 and samples.shape[1] == 1:
    # we'll just work w/ 1d vectors for the scalar case
    samples = np.squeeze(samples)
    assignment_pts = np.squeeze(assignment_pts)
  if samples.ndim == 1:
    # we use a more efficient partition strategy for scalar vars that
    # requires the assignment points to be sorted
    assignment_pts = np.sort(assignment_pts)
    assert np.all(np.diff(assignment_pts) > 0)  # monotonically increasing
    lagrange_mult = lagrange_mult * np.std(samples)
  else:
    lagrange_mult = lagrange_mult * np.mean(np.std(samples, axis=0))
    #^ put effective lagrange mult on a sort of normalized scale
    #  with the standard deviation of our samples

  # partition the data into appropriate clusters
  quantized_code, cluster_assignments, assignment_pts, codeword_lengths = \
      partition_with_drops(samples, assignment_pts,
                           codeword_lengths, lagrange_mult)

  if samples.ndim == 1:
    MSE = np.mean(np.square(quantized_code - samples))
  else:
    MSE = np.mean(np.sum(np.square(quantized_code - samples), axis=1))

  cword_probs = np.power(2., -1 * codeword_lengths)
  shannon_entropy = np.sum(cword_probs * codeword_lengths)
  code_cost = MSE + lagrange_mult * shannon_entropy

  iter_idx = 0
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
        partition_with_drops(samples, assignment_pts,
                             codeword_lengths, lagrange_mult)

    if samples.ndim == 1:
      MSE = np.mean(np.square(quantized_code - samples))
    else:
      MSE = np.mean(np.sum(np.square(quantized_code - samples), axis=1))

    cword_probs = np.power(2., -1 * codeword_lengths)
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
    iter_idx += 1

  print('Completed quantization in ', iter_idx, 'iterations')
  return assignment_pts, cluster_assignments, MSE, shannon_entropy, cword_probs


def quantize(raw_vals, assignment_vals, codeword_lengths,
             l_weight, return_cluster_assignments=False):
  """
  Makes a quantization according to BOTH nearest neighbor and resulting code len

  We could assign the raw values to their nearest neighbor in assignment_vals,
  but that would ignore the resulting entropy of the assignment. Assuming that
  we use an optimal lossless code (Huffman, Arithmetic, etc.) for the
  assignments, the expected length of the code is arbitrarily close to the
  entropy. Instead, we will minimize a function which includes not only the
  distance to assignment points, but also has a lagrange multiplier term
  that accounts for the codeword length.

  Parameters
  ----------
  raw_vals : ndarray (d, n) or (d,)
      The raw values to be quantized according to the assignment points and lens
  assignment_vals : ndarray (m, n) or (m,)
      The allowable assignment values. Every raw value will be assigned one
      of these values instead.
  codeword_lengths : ndarray (m,)
      The expected lengths of the codewords for each of the assignment vals.
  l_weight : float
      A value for the lagrange multiplier used in the augmented distance we use
      to make the quantization
  return_cluster_assignments : bool, optional
      Our default behavior is to just return the actual quantized values
      (determined by the assingment points). If this parameter is true,
      also return the index of assigned point for each of the rows in
      raw_vals (this is the identifier of which codeword was used to quantize
      this datapoint). Default False.
  """
  assert len(assignment_vals) == len(codeword_lengths)
  # I could not easily find an implementation of nearest neighbors that would
  # use a generalized cost function rather than the l2-norm-squared to assign
  # the partition. Therefore, we'll (for now) calculate the cost of assigning
  # each point to each interval and then just take the minimum.

  if raw_vals.ndim == 1:
    l2_distance = np.square(scipy_distance(raw_vals[:, None],
                            assignment_vals[:, None], metric='euclidean'))
    assignment_cost = l2_distance + l_weight * codeword_lengths[None, :]
    c_assignments = np.argmin(assignment_cost, axis=1)
  else:
    if len(assignment_vals) * len(raw_vals) > 1e9:
      # do this in chunks to avoid memory overflow
      chunksize = 10000
      num_chunks = raw_vals.shape[0] // chunksize
      leftover = raw_vals.shape[0] - (chunksize * num_chunks)
      c_assignments = np.zeros((raw_vals.shape[0], ), dtype='int')
      for chunk_idx in range(num_chunks):
        l2_distance = \
          np.square(scipy_distance(
            raw_vals[chunk_idx * chunksize : (chunk_idx + 1) * chunksize],
            assignment_vals, metric='euclidean'))
        assignment_cost = l2_distance + l_weight * codeword_lengths[None, :]
        c_assignments[chunk_idx * chunksize : (chunk_idx + 1) * chunksize] = \
            np.argmin(assignment_cost, axis=1)
      if leftover > 0:
        l2_distance = np.square(scipy_distance(
          raw_vals[-leftover:, :], assignment_vals, metric='euclidean'))
        assignment_cost = l2_distance + l_weight * codeword_lengths[None, :]
        c_assignments[-leftover:] = np.argmin(assignment_cost, axis=1)
    else:
      l2_distance = np.square(scipy_distance(
        raw_vals, assignment_vals, metric='euclidean'))
      assignment_cost = l2_distance + l_weight * codeword_lengths[None, :]
      c_assignments = np.argmin(assignment_cost, axis=1)

  if return_cluster_assignments:
    return assignment_vals[c_assignments], c_assignments
  else:
    return assignment_vals[c_assignments]


def partition_with_drops(raw_vals, a_vals, c_lengths, l_weight):
  """
  Partition the data according to the assignment values.

  This is just a wrapper on the quantize() function above which, following the
  advice of Chou et al. (1989), drops assignment points from the quantization
  whenever there are quantization bins with no data in them.

  Parameters
  ----------
  raw_vals : ndarray (d, n) or (d,)
      The raw values to be quantized according to the assignment points
  a_vals : ndarray (m, n) or (m,)
      The *initial* allowable assignment values. These may change according to
      whether quantizing based on these initial points results in empty bins.
  c_lengths : ndarray (m, )
      The (precomputed) codeword lengths for each of the assignment points.
      These will have been computed purely based on the empirical entropy of the
      quantized code from the previous iteration of the algorithm.
  l_weight : float
      The value of the lagrange multiplier in the augmented cost function
  """
  quant_code, c_assignments = quantize(raw_vals, a_vals,
                                       c_lengths, l_weight, True)

  cword_probs = calculate_assignment_probabilites(c_assignments,
                                                  a_vals.shape[0])
  if np.any(cword_probs == 0):
    nonzero_prob_pts = np.where(cword_probs != 0)
    # the indexes of c_assignments should reflect these dropped bins
    temp = np.arange(a_vals.shape[0])
    temp = temp[nonzero_prob_pts]
    reassigned_inds = {old_idx: new_idx for new_idx, old_idx in enumerate(temp)}
    for pt in range(len(c_assignments)):
      c_assignments[pt] = reassigned_inds[c_assignments[pt]]
    a_vals = a_vals[nonzero_prob_pts]
    cword_probs = cword_probs[nonzero_prob_pts]
  # update c_lengths so that the returned values reflect the current assignment
  c_lengths = -1 * np.log2(cword_probs)

  return quant_code, c_assignments, a_vals, c_lengths


def calculate_assignment_probabilites(assignments, num_clusters):
  """
  Just counts the occurence of each assignment to get an empirical pdf estimate
  """
  temp = np.arange(num_clusters)
  hist_b_edges = np.hstack([-np.inf, (temp[:-1] + temp[1:]) / 2, np.inf])
  assignment_counts, _ = np.histogram(assignments, hist_b_edges)
  empirical_density = assignment_counts / np.sum(assignment_counts)
  return empirical_density
