"""Models for contrastive learning of representations from videos."""

import functools
from typing import Any, Dict, Optional, Tuple, Union

import flax.linen as nn
from immutabledict import immutabledict
import jax
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils as base_model_utils
from scenic.model_lib.base_models import regression_model
from scenic.model_lib.layers import nn_layers
from scenic.projects.baselines import simple_cnn
from scenic.projects.baselines import resnet
from scenic.projects.baselines import vit
from scenic.projects.func_dist import holdc_utils
from scenic.projects.func_dist import model as func_dist_model
from scenic.projects.func_dist import model_utils as func_dist_model_utils


def num_examples(logits: jnp.ndarray,
                 weights: Optional[jnp.ndarray] = None
                 ) -> Union[jnp.ndarray, int]:
  if weights is None:
    return logits.shape[0]
  return weights.sum()


def pmapped_num_videos(logits: jnp.ndarray,
                       one_hot_targets: jnp.ndarray,
                       weights: Optional[jnp.ndarray] = None
                       ) -> Union[jnp.ndarray, int]:
  return 1


def num_videos(logits: jnp.ndarray,
               one_hot_targets: jnp.ndarray,
               weights: Optional[jnp.ndarray] = None
               ) -> Union[jnp.ndarray, int]:
  return logits.shape[0]


_HOLDC_METRICS = immutabledict({
    'triplet_svtc_loss':
        holdc_utils.tc_loss,
})
_FULL_SEQUENCE_METRICS = immutabledict({
    'spearman_correlation':
        (func_dist_model_utils.pmapped_spearman_correlation,
         pmapped_num_videos),
    'misclassification_rate':
        (func_dist_model_utils.binary_misclassified_count,
         func_dist_model_utils.num_frames_sub_one),
})


def holdc_metrics_function(
    predictions: jnp.ndarray,
    batch: base_model.Batch,
    metrics: base_model.MetricNormalizerFnDict = _HOLDC_METRICS,
    predict_seconds: bool = False,
    mean_fps: float = 12.,
) -> Dict[str, Tuple[float, int]]:
  """Calculate metrics for time-contrastive learning.

  Currently we assume each metric_fn has the API:
    ```metric_fn(predictions, targets, weights)```
  and returns an array of shape [batch,]. We also assume that to compute
  the aggregate metric, one should sum across all batches, then divide by the
  total samples seen. In this way we currently only support metrics of the 1/N
  sum f(inputs, targets). Note, the caller is responsible for dividing by
  the normalizer when computing the mean of each metric.

  Args:
   predictions: Output of model in shape [batch, length].
   batch: Batch (dict) with keys 'targets' and optionally 'batch_mask'.
   metrics: The regression metrics to evaluate. The key is the
     name of the  metric, and the value is the metrics function.

  Returns:
    A dict of metrics, in which keys are metrics name and values are tuples of
    (metric, normalizer).
  """
  timesteps = batch.get('timesteps')
  sequence_ids = batch.get('sequence_ids')
  evaluated_metrics = {}
  for key, val in metrics.items():
    evaluated_metrics[key] = base_model_utils.psum_metric_normalizer(
        val(predictions, timesteps=timesteps, sequence_ids=sequence_ids))
  return evaluated_metrics


class TemporalContrastiveModel(base_model.BaseModel):

  def __init__(self, config, dataset_meta_data):
    self.loss = functools.partial(
        holdc_utils.tc_loss,
        pos_radius=config.pos_radius,
        neg_radius=config.neg_radius,
        margin=config.pos_neg_margin,
        multiseq=True)
    self.num_positives = functools.partial(
        holdc_utils.get_num_positives,
        pos_radius=config.pos_radius,
        neg_radius=config.neg_radius,
        margin=config.pos_neg_margin,
        multiseq=True)
    super().__init__(config, dataset_meta_data)


  def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one
        of the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(logits,
      label, weights)```
    """
    if split == 'test':
      return functools.partial(
          func_dist_model.temporal_regression_metrics_function,
          metrics=_FULL_SEQUENCE_METRICS,
          split=split)
    else:
        metrics = immutabledict({
            'triplet_svtc_loss': self.loss,
        })
        return functools.partial(holdc_metrics_function, metrics=metrics)

  def loss_function(self,
                    predictions: jnp.ndarray,
                    batch: base_model.Batch,
                    model_params: Optional[jnp.ndarray] = None) -> float:
    """Returns the (weighted) mean squared error.

    Args:
      predictions: Output of model in shape [batch, length].
      batch: Batch (dict) with keys 'targets' and optionally 'batch_mask'.
      model_params: Parameters of the model, for optionally applying
        regularization.

    Returns:
      The (weighted) mean squared error.
    """
    weights = batch.get('batch_mask')
    # if self.predict_seconds:
    #   targets = batch['targets_seconds']
    # else:
    #   targets = batch['targets']
    #   # Adjust for variable frame rates.
    #   predictions = predictions * batch['frame_rate'] / self.dataset_mean_fps

    total_loss = self.loss(
        predictions,
        timesteps=batch['timesteps'],
        sequence_ids=batch['sequence_ids'])
    if self.config.get('l2_decay_factor'):
      l2_loss = base_model_utils.l2_regularization(model_params)
      total_loss += 0.5 * self.config.l2_decay_factor * l2_loss
    return total_loss


class ResnetTcModel(TemporalContrastiveModel):
  """Resnet model for time-contrastive representation learning."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return resnet.ResNet(
        num_outputs=self.config.embedding_size,
        num_filters=self.config.num_filters,
        num_layers=self.config.num_layers,
        dtype=model_dtype)


class VitTcModel(TemporalContrastiveModel):
  """Vision Transformer model for time-contrastive representation learning."""

  def build_flax_model(self)-> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return vit.ViT(
        num_classes=self.config.embedding_size,
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        num_heads=self.config.model.num_heads,
        representation_size=self.config.model.representation_size,
        patches=self.config.model.patches,
        hidden_size=self.config.model.hidden_size,
        classifier=self.config.model.classifier,
        dropout_rate=self.config.model.get('dropout_rate', 0.1),
        attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                     0.1),
        stochastic_depth=self.config.model.get('stochastic_depth', 0.0),
        output_proj_kernel_init=nn.initializers.lecun_normal(),
        dtype=model_dtype,
    )

  def init_from_train_state(
      self, train_state: Any, restored_train_state: Any,
      restored_model_cfg: ml_collections.ConfigDict) -> Any:
    """Updates the train_state with data from restored_train_state.

    This function is writen to be used for 'fine-tuning' experiments. Here, we
    do some surgery to support larger resolutions (longer sequence length) in
    the transformer block, with respect to the learned pos-embeddings.
    Args:
      train_state: A raw TrainState for the model.
      restored_train_state: A TrainState that is loaded with parameters/state of
        a  pretrained model.
      restored_model_cfg: Configuration of the model from which the
        restored_train_state come from. Usually used for some asserts.
    Returns:
      Updated train_state.
    """
    return vit.init_vit_from_train_state(train_state, restored_train_state,
                                         self.config, restored_model_cfg)
