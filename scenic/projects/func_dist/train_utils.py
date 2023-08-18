"""Extend scenic.train_lib.train_utils with custom datasets.
"""
import collections
import json
import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

# from absl import logging
import numpy as np
import flax
import flax.linen as nn
from flax import optim
import jax.numpy as jnp


from scenic.projects.func_dist.datasets import ssv2_regression  # pylint: disable=unused-import
from scenic.train_lib import train_utils

get_dataset = train_utils.get_dataset
TrainState = train_utils.TrainState

Array = Union[jnp.ndarray, np.ndarray]
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                    Dict[str, Tuple[float, int]]]
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]


def load_json_metrics(workdir, split):
  best_metrics = collections.defaultdict(list)
  if os.path.exists(os.path.join(workdir, f'best_{split}_metrics.json')):
    with open(os.path.join(workdir, f'best_{split}_metrics.json')) as f:
      best_metrics = json.load(f)
  return best_metrics


def save_new_checkpoints(workdir, ckpt_names, train_state, lead_host,
                         max_to_keep, chrono, report_progress):
  """Sync train state and create a checkpoint for each name in ckpt_names."""
  if ckpt_names:
    # Sync model state across replicas.
    train_state = train_utils.sync_model_state_across_replicas(
        train_state)
    if lead_host:
      train_state.replace(  # pytype: disable=attribute-error
          accum_train_time=chrono.accum_train_time)
  for ckpt in ckpt_names:
    with report_progress.timed('checkpoint'):
      if lead_host:
        train_utils.save_checkpoint(
            os.path.join(workdir, ckpt),
            train_state,
            max_to_keep=max_to_keep)


def add_eval_scores(eval_scores, scores_to_add, n_ckpts_to_keep, loss_metrics):
  """Add new evaluation scores from scores_to_add to eval_scores."""
  for k, v2 in scores_to_add.items():
    v = eval_scores[k] if k in eval_scores else []
    included_steps = set([score_step[1] for score_step in v])
    new_vs = v + [new_v for new_v in v2 if new_v[1] not in included_steps]
    sorted_vs = sorted(new_vs, reverse=k not in loss_metrics)[:n_ckpts_to_keep]
    eval_scores[k] = sorted_vs
  return eval_scores


def compare_eval_scores(eval_summary, best_eval_metrics, n_ckpts_to_keep,
                        loss_metrics, step, split):
  """Find evaluation metrics whose new performance is in top-n_ckpts_to_keep."""
  new_ckpts_to_save = []
  for k, v in eval_summary.items():
    if (len(best_eval_metrics[k]) < n_ckpts_to_keep
        or (k in loss_metrics and best_eval_metrics[k][-1][0] > v)
        or (k not in loss_metrics and best_eval_metrics[k][-1][0] > v)):
      if k in loss_metrics:
        # Keep the smallest value checkpoints.
        best_eval_metrics[k] = sorted(
             best_eval_metrics[k][:n_ckpts_to_keep - 1] + [[v, step]])
      else:
        # Keep the largest value checkpoints.
        best_eval_metrics[k] = sorted(
            best_eval_metrics[k][:n_ckpts_to_keep - 1] + [[v, step]],
            reverse=True)
      new_ckpts_to_save.append(f'best_{split}_{k}')
  return best_eval_metrics, new_ckpts_to_save


def keep_best_checkpoints(eval_summary, best_eval_metrics, n_ckpts_to_keep,
                          loss_metrics, split, workdir, train_state, step,
                          lead_host, chrono, report_progress):
  """Save checkpoints for eval metrics performing in top-n_ckpts_to_keep."""
  # Evaluation metrics where new performance is in top-n_ckpts_to_keep.
  best_eval_metrics, new_ckpts_to_save = compare_eval_scores(
      eval_summary, best_eval_metrics, n_ckpts_to_keep, loss_metrics,
      step, split)
  if new_ckpts_to_save:
    # Save checkpoints for metrics that are in top-n_ckpts_to_keep.
    save_new_checkpoints(workdir, new_ckpts_to_save, train_state, lead_host,
        2 * n_ckpts_to_keep, chrono, report_progress)
    with open(
        os.path.join(workdir, f'best_{split}_metrics.json'), 'w') as f:
      json.dump(best_eval_metrics, f)
    # Remove outdated checkpoints.
    trim_checkpoints(workdir, split, best_eval_metrics)


def trim_checkpoints(workdir, split, best_eval_metrics):
  """Delete superfluous checkpoints."""
  for k in best_eval_metrics:
    ckpt_dir = os.path.join(workdir, f'best_{split}_{k}')
    steps_to_keep = [kv[1] for kv in best_eval_metrics[k]]
    for ckpt in os.listdir(ckpt_dir):
      step = int(ckpt.replace('checkpoint_', ''))
      if step not in steps_to_keep:
        os.remove(os.path.join(ckpt_dir, ckpt))


def get_sequence_ids(batch_mask, video_lengths):
  padded_size = batch_mask.shape[1]
  video_lengths = jnp.squeeze(video_lengths, axis=1)
  sequence_ids = jnp.array(
      [jnp.floor_divide(jnp.arange(padded_size), jnp.minimum(l, padded_size))
       for l in video_lengths])
  last_lengths = jnp.mod(padded_size, video_lengths)
  n_full_repeats = jnp.floor_divide(padded_size, video_lengths)
  batched_lengths = jnp.minimum(video_lengths, padded_size)
  timesteps = jnp.stack([
      jnp.concatenate(
          [jnp.tile(jnp.arange(batched_lengths[i]), (n_full_repeats[i])),
           jnp.arange(last_lengths[i])])
      for i in range(len(video_lengths))])
  return sequence_ids, timesteps


def mask_from_video_lengths(inputs, video_lengths):
  mask = np.zeros((inputs.shape[0], inputs.shape[1]))
  for row, seq_len in enumerate(video_lengths):
    mask[row, :seq_len[0]] = 1 
  return jnp.array(mask)


def full_seq_test_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    num_stacked_frames: int,
    metrics_fn: MetricFn,
    test_batch_size: int,
    compute_distances: bool = False,
    return_logits_and_labels: bool = False,
    return_confusion_matrix: bool = False,
    softmax_logits: bool = False,
    debug: Optional[bool] = False
) -> Union[Tuple[Dict[str, Tuple[float, int]], jnp.ndarray, jnp.array],
           Tuple[Dict[str, Tuple[float, int]], jnp.ndarray],
           Dict[str, Tuple[float, int]]]:
  inputs = jnp.squeeze(batch['inputs'], axis=0)# [batch['batch_mask']
  goal = inputs[-1:]
  batch['batch_mask'] = batch['batch_mask'][1:]
  vid_len = inputs.shape[0]
  stacked_inputs = jnp.tile(inputs[0],
                            (vid_len, num_stacked_frames, 1, 1, 1))
  for i in range(num_stacked_frames):
    stacked_inputs = stacked_inputs.at[i:, num_stacked_frames - 1 - i].set(
        inputs[:-i] if i > 0 else inputs)
  inputs = stacked_inputs
  all_logits = None
  variables = { 
      'params': train_state.optimizer.target,
      **train_state.model_state
  }
  if compute_distances:
      goal_emb = flax_model.apply(variables, goal, train=False, mutable=False, debug=debug)
  # logging.info('Producing predictions')
  for idx in range(0, vid_len - 1, test_batch_size):
    temp_input = inputs[idx:min(idx + test_batch_size, vid_len - 1)]
    if not compute_distances:
      goal_image = jnp.tile(goal, [len(temp_input)] + [1] * len(goal.shape))
      temp_input = jnp.concatenate([temp_input, goal_image], axis=1)
    logits = flax_model.apply(
        variables, temp_input, train=False, mutable=False, debug=debug)
    if compute_distances:
        logits = jnp.linalg.norm(goal_emb - logits, axis=1)
        logits = jnp.expand_dims(logits, axis=1)
    if softmax_logits:
      logits = nn.softmax(logits, axis=-1)
    all_logits = (
        logits if all_logits is None else jnp.concatenate([all_logits, logits]))
  # logging.info('Entering metrics_fn')
  metrics = metrics_fn(all_logits, batch)

  if return_confusion_matrix:
    confusion_matrix = get_confusion_matrix(
        labels=batch['label'], logits=logits, batch_mask=batch['batch_mask'])
    confusion_matrix = jax.lax.all_gather(confusion_matrix, 'batch')
    return metrics, confusion_matrix

  if return_logits_and_labels:
    logits = jax.lax.all_gather(logits, 'batch')
    labels = jax.lax.all_gather(batch['label'], 'batch')
    return metrics, logits, labels

  return metrics, all_logits



# @flax.struct.dataclass
# class TrainState:
#   """Dataclass to keep track of state of training.
# 
#   The state of training is structured as a flax.struct.dataclass, which enables
#   instances of this class to be passed into jax transformations like tree_map
#   and pmap.
#   """
#   global_step: Optional[int] = 0 
#   optimizer: Optional[optim.Optimizer] = None
#   model_state: Optional[Any] = None
#   rng: Optional[jnp.ndarray] = None
#   accum_train_time: Optional[int] = 0 
# 
#   def __getitem__(self, item):
#     """Make TrainState a subscriptable object."""
#     return getattr(self, item)
# 
#   def get(self, keyname: str, default: Optional[Any] = None) -> Any:
#     """Return the value for key if it exists otherwise the default."""
#     try:
#       return self[keyname]
#     except KeyError:
#       return default
 
