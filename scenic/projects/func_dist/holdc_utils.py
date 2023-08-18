"""Jax implementation of single-view time-contrastive loss.

Based on the TF1 implementation of Time-Contrasive Networks in
https://github.com/google-research/google-research/tree/36101ab4095065a4196ff4f6437e94f0d91df4e9/research/tcn
"""
from typing import Optional, Tuple, Union
import jax.numpy as jnp


def pairwise_squared_distance(feature):
  """Computes the squared pairwise distance matrix.

  output[i, j] = || feature[i, :] - feature[j, :] ||_2^2

  Args:
    feature: 2-D Tensor of size [number of data, feature dimension]

  Returns:
    pairwise_squared_distances: 2-D Tensor of size
      [number of data, number of data]
  """
  pairwise_squared_distances = (
      jnp.sum(jnp.square(feature), axis=1, keepdims=True)
      + jnp.sum(jnp.square(jnp.transpose(feature)), axis=0, keepdims=True)
      - 2.0 * jnp.matmul(feature, jnp.transpose(feature)))

  # Deal with numerical inaccuracies. Set small negatives to zero.
  pairwise_squared_distances = jnp.maximum(pairwise_squared_distances, 0.0)
  return pairwise_squared_distances


def masked_maximum(data, mask, dim=1):
  """Computes the axis wise maximum over chosen elements.

  Args:
    data: N-D Tensor.
    mask: N-D Tensor of zeros or ones.
    dim: The dimension over which to compute the maximum.

  Returns:
    masked_maximums: N-D Tensor.
      The maximized dimension is of size 1 after the operation.
  """
  axis_minimums = jnp.min(data, dim, keepdims=True)
  masked_maximums = jnp.max(
      jnp.multiply(
          data - axis_minimums, mask), dim, keepdims=True) + axis_minimums
  return masked_maximums


def masked_minimum(data, mask, dim=1):
  """Computes the axis wise minimum over chosen elements.

  Args:
    data: 2-D Tensor of size [n, m].
    mask: 2-D Boolean Tensor of size [n, m].
    dim: The dimension over which to compute the minimum.

  Returns:
    masked_minimums: N-D Tensor.
      The minimized dimension is of size 1 after the operation.
  """
  axis_maximums = jnp.max(data, dim, keepdims=True)
  masked_minimums = jnp.min(
      jnp.multiply(
          data - axis_maximums, mask), dim, keepdims=True) + axis_maximums
  return masked_minimums


def get_num_positives(
    embeddings: jnp.ndarray,
    timesteps: jnp.ndarray,
    pos_radius: int = 6,  # TODO: Set separately for each video
    neg_radius: int = 12,  # TODO: Set separately for each video
    margin: float = 0.2,
    sequence_ids: Optional[jnp.ndarray] = None,
    multiseq: bool = False) -> jnp.ndarray:
  del embeddings
  assert neg_radius > pos_radius

  # If timesteps shape isn't [batchsize, 1], reshape to [batch_size, 1].
  tshape = jnp.shape(timesteps)
  assert len(tshape) == 2 or len(tshape) == 1
  if len(tshape) == 1:
    timesteps = jnp.reshape(timesteps, [tshape[0], 1]) 

  if multiseq:
    # If sequence_ids shape isn't [batchsize, 1], reshape to [batch_size, 1].
    tshape = jnp.shape(sequence_ids)
    assert len(tshape) == 2 or len(tshape) == 1
    if len(tshape) == 1:
      sequence_ids = jnp.reshape(sequence_ids, [tshape[0], 1]) 

    # Build pairwise binary adjacency matrix based on sequence_ids
    sequence_adjacency = jnp.equal(sequence_ids, jnp.transpose(sequence_ids))

    # Invert so we can select negatives only.
    sequence_adjacency_not = jnp.logical_not(sequence_adjacency)

    in_pos_range = jnp.logical_and(
        jnp.less_equal(
            jnp.abs(timesteps - jnp.transpose(timesteps)), pos_radius),
        sequence_adjacency)
  else:
    in_pos_range = jnp.less_equal(
        jnp.abs(timesteps - jnp.transpose(timesteps)), pos_radius)

  batch_size = jnp.size(timesteps)

  mask_positives = ( 
      in_pos_range.astype(jnp.float32) - jnp.diag(jnp.ones([batch_size])))

  # In lifted-struct, the authors multiply 0.5 for upper triangular
  #   in semihard, they take all positive pairs except the diagonal.
  num_positives = jnp.sum(mask_positives)

  return num_positives


def tc_loss(
    embeddings: jnp.ndarray,
    timesteps: jnp.ndarray,
    pos_radius: int = 6,  # TODO: Set separately for each video
    neg_radius: int = 12,  # TODO: Set separately for each video
    margin: float = 0.2,
    sequence_ids: Optional[jnp.ndarray] = None,
    multiseq: bool = False) -> jnp.ndarray:
  """Computes the single view triplet loss with semi-hard negative mining.

  The loss encourages the positive distances (between a pair of embeddings with
  the same labels) to be smaller than the minimum negative distance among
  which are at least greater than the positive distance plus the margin constant
  (called semi-hard negative) in the mini-batch. If no such negative exists,
  uses the largest negative distance instead.

  Anchor, positive, negative selection is as follow:
  Anchors: We consider every embedding timestep as an anchor.
  Positives: pos_radius defines a radius (in timesteps) around each anchor from
    which positives can be drawn. E.g. An anchor with t=10 and a pos_radius of
    2 produces a set of 4 (anchor,pos) pairs [(a=10, p=8), ... (a=10, p=12)].
  Negatives: neg_radius defines a boundary (in timesteps) around each anchor,
    outside of which negatives can be drawn. E.g. An anchor with t=10 and a
    neg_radius of 4 means negatives can be any t_neg where t_neg < 6 and
    t_neg > 14.

  Args:
    embeddings: 2-D Tensor of embedding vectors.
    timesteps: 1-D Tensor with shape [batch_size, 1] of sequence timesteps.
    pos_radius: int32; the size of the window (in timesteps) around each anchor
      timestep that a positive can be drawn from.
    neg_radius: int32; the size of the window (in timesteps) around each anchor
      timestep that defines a negative boundary. Negatives can only be chosen
      where negative timestep t is < negative boundary min or > negative
      boundary max.
    margin: Float; the triplet loss margin hyperparameter.
    sequence_ids: (Optional) 1-D Tensor with shape [batch_size, 1] of sequence
      ids. Together (sequence_id, sequence_timestep) give us a unique index for
      each image if we have multiple sequences in a batch.
    multiseq: Boolean, whether or not the batch is composed of multiple
      sequences (with possibly colliding timesteps).

  Returns:
    triplet_loss: jnp.float32 scalar."""
  assert neg_radius > pos_radius

  # If timesteps shape isn't [batchsize, 1], reshape to [batch_size, 1].
  tshape = jnp.shape(timesteps)
  assert len(tshape) == 2 or len(tshape) == 1
  if len(tshape) == 1:
    timesteps = jnp.reshape(timesteps, [tshape[0], 1])

  # Build pairwise squared distance matrix.
  pdist_matrix = pairwise_squared_distance(embeddings)

  # Build pairwise binary adjacency matrix, where adjacency[i,j] is True
  # if timestep j is inside the positive range for timestep i and both
  # timesteps come from the same sequence.
  # pos_radius = pos_radius.astype(jnp.int32)

  if multiseq:
    # If sequence_ids shape isn't [batchsize, 1], reshape to [batch_size, 1].
    tshape = jnp.shape(sequence_ids)
    assert len(tshape) == 2 or len(tshape) == 1
    if len(tshape) == 1:
      sequence_ids = jnp.reshape(sequence_ids, [tshape[0], 1])

    # Build pairwise binary adjacency matrix based on sequence_ids
    sequence_adjacency = jnp.equal(sequence_ids, jnp.transpose(sequence_ids))

    # Invert so we can select negatives only.
    sequence_adjacency_not = jnp.logical_not(sequence_adjacency)

    in_pos_range = jnp.logical_and(
        jnp.less_equal(
            jnp.abs(timesteps - jnp.transpose(timesteps)), pos_radius),
        sequence_adjacency)
    # Build pairwise binary discordance matrix, where discordance[i,j] is True
    # if timestep j is inside the negative range for timestep i or if the
    # timesteps come from different sequences.
    in_neg_range = jnp.logical_or(
        jnp.greater(jnp.abs(timesteps - jnp.transpose(timesteps)), neg_radius),
        sequence_adjacency_not
    )
  else:
    in_pos_range = jnp.less_equal(
        jnp.abs(timesteps - jnp.transpose(timesteps)), pos_radius)
    in_neg_range = jnp.greater(jnp.abs(timesteps - jnp.transpose(timesteps)),
                              neg_radius)

  batch_size = jnp.size(timesteps)

  # compute the mask
  pdist_matrix_tile = jnp.tile(pdist_matrix, [batch_size, 1])
  mask = jnp.logical_and(
      jnp.tile(in_neg_range, [batch_size, 1]),
      jnp.greater(pdist_matrix_tile,
                 jnp.reshape(jnp.transpose(pdist_matrix), [-1, 1])))
  mask_final = jnp.reshape(
      jnp.greater(
          jnp.sum(mask.astype(jnp.float32), 1, keepdims=True),
          0.0), [batch_size, batch_size])
  mask_final = jnp.transpose(mask_final)

  in_neg_range = in_neg_range.astype(jnp.float32)
  mask = mask.astype(jnp.float32)

  # negatives_outside: smallest D_an where D_an > D_ap
  negatives_outside = jnp.reshape(
      masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
  negatives_outside = jnp.transpose(negatives_outside)

  # negatives_inside: largest D_an
  negatives_inside = jnp.tile(
      masked_maximum(pdist_matrix, in_neg_range), [1, batch_size])
  semi_hard_negatives = jnp.where(
      mask_final, negatives_outside, negatives_inside)

  loss_mat = jnp.add(margin, pdist_matrix - semi_hard_negatives)

  mask_positives = (
      in_pos_range.astype(jnp.float32) - jnp.diag(jnp.ones([batch_size])))

  # In lifted-struct, the authors multiply 0.5 for upper triangular
  #   in semihard, they take all positive pairs except the diagonal.
  num_positives = jnp.sum(mask_positives)

  triplet_loss = (
      jnp.sum(jnp.maximum(jnp.multiply(loss_mat, mask_positives), 0.0)))
  # triplet_loss = jnp.divide(
  #     jnp.sum(jnp.maximum(jnp.multiply(loss_mat, mask_positives), 0.0)),
  #     num_positives)

  return triplet_loss, num_positives

