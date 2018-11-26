"""
An implementation of the generalized Lloyd algorithm for vector quantization

This is currently implemented only for the 2-norm-squared error distortion
metric but is fully general for p-norms and the general quadratic form
e^T R e where e is the error vector. The computational or analytical details
of using these other distortion measures may be less desirable than for the
canonical 2-norm-squared distortion metric.

This approach really only makes sense when one has a codebook with equal
codeword lengths (sometimes called a {F}ixed {L}ength {C}ode). In that case,
we just care about minimizing the distortion given some number of codewords. If
the code can be variable length, reflecting the probability of each codeword,
(a so-called {V}ariable {L}ength {C}ode), then *you don't want to use this*.
Instead, you want to use the optimal generalized Lloyd algorithm which will
make quantization assignments that account for the different codeword
probabilities and thus the different codeword lengths

.. [1] Lloyd, S. (1957, unpublished Bell Labs technical report).
       Least squares quantization in PCM. Later published as:

       Lloyd, S. (1982). Least squares quantization in PCM. IEEE transactions
       on information theory, 28(2), 129-137.

.. [2] Linde, Y., Buzo, A., & Gray, R. (1980). An algorithm for vector quantizer
       design. IEEE transactions on information theory, 28(1), 84-95.
"""
import copy
import numpy as np
from scipy.spatial.distance import cdist as scipy_distance

def compute_quantization(samples, init_assignment_pts,
                         force_const_num_assignment_pts=True, epsilon=1e-5):
  """
  Generalized Lloyd-Max alg. for quantizing vector-valued random variables

  The quantizer minimally consists of the set of assignment points. The optimal
  quantization region edges are precisely the hyperplanes that define the region
  for which a point is closer to the assignment point than any other.
  This is the voronoi tesselation of the space filled with assignment points.
  In the scalar case these are the midpoints between any two consecutive
  assignment points.

  Parameters
  ----------
  samples : ndarray (d, n) or (d,)
      This is an array of d samples of an n-dimensional random variable
      that we wish to find the LM quantizer for. If these are scalar random
      variables, we will accept a 1d array as input.
  init_assignment_pts : ndarray (m, n) or (m,)
      This is an array of some initial guesses for m total assignment points
      for the quantizer. This is the main knob we have in determining the
      fidelity of this quantizer. If the quantization is scalar we will accept
      a 1d array as input.
  force_const_num_assignment_pts : bool, optional
      If true, whenever we encounter a partition that leaves an assignment point
      with no assigned datapoints (the 'bin' is empty), then we will split the
      most populous 'bin' and recompute the partition, keeping the same number
      of total assignment points. If false, whenever we encounter an empty 'bin'
      we will just drop that assignment point. Empty assignment point bins will
      be much more common in the vector quantization setting. Default True.
  epsilon : float, optional
      The tolerance for change in MSE after which we decide we have converged
      Default 1e-5.

  Returns
  -------
  assignment_pts : ndarray (m, n) or (m,)
      The converged assignment points
  cluster_assignments : ndarray (d, )
      For each sample, the index of the codeword to which Lloyd quantization
      assigns this datapoint. We can compute the actual quantized values outside
      this function by invoking `assignment_pts[cluster_assignments]`
  MSE : float
      The mean squared error (the mean l2-normed-squared to be precise) for the
      returned quantization.
  shannon_entropy : float
      The (empirical) Shannon entropy for this code. We can say that assuming
      we use a lossless binary source code, that our expected codeword length
      is precisely this value.
  """
  samples = np.copy(samples)  # get rid of the original reference to make sure
                              # we don't modify the data in the calling scope
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
  if samples.ndim == 1:
    # we use a more efficient partition strategy for scalar vars that
    # requires the assignment points to be sorted
    assignment_pts = np.sort(init_assignment_pts)
    assert np.all(np.diff(assignment_pts) > 0)  # monotonically increasing

  # partition the data into appropriate clusters
  if force_const_num_assignment_pts:
    partition = partition_with_splitting
  else:
    partition = partition_with_drops
  quantized_code, cluster_assignments, assignment_pts = partition(
      samples, assignment_pts)

  if samples.ndim == 1:
    MSE = np.mean(np.square(quantized_code - samples))
  else:
    MSE = np.mean(np.sum(np.square(quantized_code - samples), axis=1))

  while True:
    old_MSE = np.copy(MSE)
    # update the centroids based on the current partition
    for bin_idx in range(assignment_pts.shape[0]):
      binned_samples = samples[cluster_assignments == bin_idx]
      assert len(binned_samples) > 0
      assignment_pts[bin_idx] = np.mean(binned_samples, axis=0)

    # partition the data into appropriate clusters
    quantized_code, cluster_assignments, assignment_pts = partition(
        samples, assignment_pts)

    if samples.ndim == 1:
      MSE = np.mean(np.square(quantized_code - samples))
    else:
      MSE = np.mean(np.sum(np.square(quantized_code - samples), axis=1))

    if not np.isclose(old_MSE, MSE):
      assert MSE <= old_MSE, 'uh-oh, MSE increased'

    if old_MSE == 0.0:  # avoid divide by zero below
      break
    if (np.abs(old_MSE - MSE) / old_MSE) < epsilon:
      break
      #^ this algorithm provably reduces MSE or leaves it unchanged at each
      #  iteration so the boundedness of MSE means this is a valid stopping
      #  criterion

  if samples.ndim != 1:
    # we want to return the code and quantization in the original space
    assignment_pts = assignment_pts * saved_component_stds[None, :]
    quantized_code = assignment_pts[cluster_assignments]
    samples = samples * saved_component_stds[None, :]
    # compute the MSE of the quantized code in the original space
    MSE = np.mean(np.sum(np.square(quantized_code - samples), axis=1))

  # the last thing we'll do is compute the (empirical)
  # shannon entropy of this code
  cword_probs = calculate_assignment_probabilites(cluster_assignments,
                                                  assignment_pts.shape[0])
  assert np.isclose(np.sum(cword_probs), 1.0)
  shannon_entropy = -1 * np.sum(cword_probs * np.log2(cword_probs))

  return assignment_pts, cluster_assignments, MSE, shannon_entropy


def quantize(raw_vals, assignment_vals, return_cluster_assignments=False):
  """
  Makes a quantization according to the nearest neighbor in assignment_vals

  Parameters
  ----------
  raw_vals : ndarray (d, n) or (d,)
      The raw values to be quantized according to the assignment points
  assignment_vals : ndarray (m, n) or (m,)
      The allowable assignment values. Every raw value will be assigned one
      of these values instead.
  return_cluster_assignments : bool, optional
      Our default behavior is to just return the actual quantized values
      (determined by the assingment points). If this parameter is true,
      also return the index of assigned point for each of the rows in
      raw_vals (this is the identifier of which codeword was used to quantize
      this datapoint). Default False.
  """
  if raw_vals.ndim == 1:
    if len(assignment_vals) == 1:
      # everything gets assigned to this point
      c_assignments = np.zeros((len(raw_vals),), dtype='int')
    else:
      bin_edges = (assignment_vals[:-1] + assignment_vals[1:]) / 2
      c_assignments = np.digitize(raw_vals, bin_edges)
      #^ This is more efficient than our vector quantization because here we use
      #  sorted bin edges and the assignment complexity is (I believe)
      #  logarithmic in the number of intervals.
  else:
    c_assignments = np.argmin(scipy_distance(raw_vals, assignment_vals,
                                             metric='euclidean'), axis=1)
    #^ This is just a BRUTE FORCE nearest neighbor search. I tried to find a
    #  fast implementation of this based on KD-trees or Ball Trees, but wasn't
    #  successful. I also tried scipy's vq method from the clustering
    #  module but it's also just doing brute force search (albeit in C).
    #  This approach might have decent performance when the number of
    #  assignment points is small (low fidelity, very lossy regime). In the
    #  future we should be able to roll a much faster search implementation and
    #  speed up this part of the algorithm...

  if return_cluster_assignments:
    return assignment_vals[c_assignments], c_assignments
  else:
    return assignment_vals[c_assignments]


def partition_with_splitting(raw_vals, a_vals):
  """
  Partitions the data according to the assignment values.

  This is just a wrapper on the quantize() function above which, following the
  advice of Linde et al. (1980), reallocates assignment points by splitting the
  most populous quantization bin when there are bins with no data in them.

  Parameters
  ----------
  raw_vals : ndarray (d, n) or (d,)
      The raw values to be quantized according to the assignment points
  a_vals : ndarray (m, n) or (m,)
      The *initial* allowable assignment values. These may change according to
      whether quantizing based on these initial points results in empty bins.
  """
  fresh_a_vals = np.copy(a_vals)
  while True:
    # try a partition
    quant_code, c_assignments = quantize(raw_vals, fresh_a_vals, True)
    cword_probs = calculate_assignment_probabilites(c_assignments,
                                                    fresh_a_vals.shape[0])
    # if there are no empty clusters we're done
    if not np.any(cword_probs == 0):
      return quant_code, c_assignments, fresh_a_vals
    # otherwise reallocate the empty cluster assignment points and try again
    else:
      most_probable = np.argmax(cword_probs)
      if raw_vals.ndim == 1:
        offset_sz = 1
        max_offset = 0.01 * np.square(np.std(raw_vals))
      else:
        offset_sz = fresh_a_vals.shape[1]
        max_offset = 0.01
        #^ each component has variance 1 due to our normalization in the calling
        # function so a maximum perturbation of 0.01 seems reasonable...
      zero_prob_bins = np.where(cword_probs == 0)
      for zero_prob_bin in zero_prob_bins:
        rand_offset = np.random.uniform(-1*max_offset, max_offset,
                                        size=offset_sz)
        fresh_a_vals[zero_prob_bin] = (fresh_a_vals[most_probable] +
                                       rand_offset)
      if raw_vals.ndim == 1:
        # we have to re-sort the assignment points b/c our 1d partition method
        # requires it
        fresh_a_vals = np.sort(fresh_a_vals)


def partition_with_drops(raw_vals, a_vals):
  """
  Partition the data according to the assignment values.

  This is just a wrapper on the quantize() function above which, following the
  advice of Chou et al. (1989), drops assignment points from the quantization
  whenever there are quantization bins with no data in them. You obviously
  shouldn't use this if you want the code to have a fixed number of codewords.

  Parameters
  ----------
  raw_vals : ndarray (d, n) or (d,)
      The raw values to be quantized according to the assignment points
  a_vals : ndarray (m, n) or (m,)
      The *initial* allowable assignment values. These may change according to
      whether quantizing based on these initial points results in empty bins.
  """
  quant_code, c_assignments = quantize(raw_vals, a_vals, True)
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

  return quant_code, c_assignments, a_vals


def calculate_assignment_probabilites(assignments, num_clusters):
  """
  Just counts the occurence of each assignment to get an empirical pdf estimate
  """
  temp = np.arange(num_clusters)
  hist_b_edges = np.hstack([-np.inf, (temp[:-1] + temp[1:]) / 2, np.inf])
  assignment_counts, _ = np.histogram(assignments, hist_b_edges)
  empirical_density = assignment_counts / np.sum(assignment_counts)
  return empirical_density
