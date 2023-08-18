"""Models for learning functional distances from videos."""

import functools
from typing import Any, Dict, Optional, Tuple

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
from scenic.projects.func_dist import model_utils
from scenic.projects.func_dist import holdc_model
from scenic.projects.vivit import model as vivit_model


def get_model_cls(model_name):
  """"Selects distance model type."""
  if model_name == 'vivit_multilabel_classification':
    return vivit_model.ViViTMultilabelClassificationModel
  elif model_name == 'vivit_classification':
    return vivit_model.ViViTClassificationModel
  elif model_name == 'vivit_multihead_classification':
    return vivit_model.ViViTMultiHeadClassificationModel
  elif model_name == 'vivit_regression':
    return ViViTRegressionModel
  elif model_name == 'simple_cnn_regression':
    return SimpleCNNRegressionModel
  elif model_name == 'resnet_regression':
    return ResnetRegressionModel
  elif model_name == 'resnet_holdc':
    return holdc_model.ResnetTcModel
  elif model_name == 'vit_holdc':
    return holdc_model.VitTcModel
  else:
    raise ValueError('Unrecognized model: {}'.format(model_name))


_REGRESSION_METRICS = immutabledict({
    'mean_absolute_error':
        (base_model_utils.weighted_absolute_error,
         base_model_utils.num_examples),
    'mean_squared_error':
        (base_model_utils.weighted_squared_error,
         base_model_utils.num_examples)
})
_TEMPORAL_REGRESSION_METRICS = immutabledict({
    'mean_absolute_error_seconds':
        (base_model_utils.weighted_absolute_error,
         base_model_utils.num_examples),
    'mean_squared_error_seconds':
        (base_model_utils.weighted_squared_error,
         base_model_utils.num_examples),
    'mean_absolute_error_steps':
        (base_model_utils.weighted_absolute_error,
         base_model_utils.num_examples),
    'mean_squared_error_steps':
        (base_model_utils.weighted_squared_error,
         base_model_utils.num_examples),
     **_REGRESSION_METRICS
})
_FULL_SEQUENCE_METRICS = immutabledict({
    'spearman_correlation':
        (model_utils.pmapped_spearman_correlation,
         holdc_model.pmapped_num_videos),
    'hinge_loss':
        (model_utils.hinge_loss,
         model_utils.num_frames_sub_one),
    'misclassification_rate':
        (model_utils.binary_misclassified_count,
         model_utils.num_frames_sub_one),
    **_TEMPORAL_REGRESSION_METRICS
})


def temporal_regression_metrics_function(
    predictions: jnp.ndarray,
    batch: base_model.Batch,
    metrics: base_model.MetricNormalizerFnDict = _TEMPORAL_REGRESSION_METRICS,
    predict_seconds: bool = False,
    mean_fps: float = 12.,
    split: str = 'valid',
    pmapped: bool = True,
) -> Dict[str, Tuple[float, int]]:
  """Calculate metrics for the regression task.

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
  if split == 'test':
    targets_steps = batch['targets']
    targets_seconds = targets_steps / batch['frame_rate'][0]
  else:
    targets_steps = batch['targets']
    targets_seconds = batch['targets_seconds']
  weights = batch.get('batch_mask')
  frame_rate = batch['frame_rate']
  evaluated_metrics = {}
  for key, val in metrics.items():
    if key in ['mean_absolute_error_seconds', 'mean_squared_error_seconds',
               'hinge_loss']:
      targets = targets_seconds
      # If preds are in steps, divide by frame rate.
      preds = predictions if predict_seconds else predictions / frame_rate
    else:
      targets = targets_steps
      if key in ['mean_absolute_error_steps', 'mean_squared_error_steps']:
        # Multiply by frame rates of individual videos.
        preds = (predictions * frame_rate if predict_seconds
                else predictions * frame_rate / mean_fps)
      else:
        # Multiply by the average frame rate only.
        preds = predictions * mean_fps if predict_seconds else predictions
    metric_normalizer = (base_model_utils.psum_metric_normalizer if pmapped
                         else model_utils.sum_metric_normalizer)
    evaluated_metrics[key] = metric_normalizer(
        (val[0](preds, targets, weights), val[1](preds, targets, weights)))
  return evaluated_metrics


def full_sequence_metrics_function(
    predictions: jnp.ndarray,
    batch: base_model.Batch,
    metrics: base_model.MetricNormalizerFnDict = _FULL_SEQUENCE_METRICS,
) -> Dict[str, Tuple[float, int]]:
  weights = batch.get('batch_mask')
  evaluated_metrics = {}
  targets = jnp.transpose(batch['targets'])[:-1]
  for key, val in metrics.items():
    evaluated_metrics[key] = base_model_utils.psum_metric_normalizer(
        (val[0](predictions, targets, weights),
         val[1](predictions, targets, weights)))
  return evaluated_metrics


class TemporalRegressionModel(regression_model.RegressionModel):

  def __init__(self, config, dataset_meta_data):
    self.predict_seconds = config.get('predict_seconds', False)
    self.dataset_mean_fps = dataset_meta_data.get('mean_frame_rate', 12.)
    self.loss_type = config.get('loss_type', 'mse')
    super().__init__(config, dataset_meta_data)

    if self.predict_seconds:
      print('Predicting in seconds')
    else:
      print('Predicting in time steps')
    print('Dataset mean fps:', self.dataset_mean_fps)


  def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one
        of the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(logits,
      label, weights)```
    """
    # del split  # for all splits, we return the same metric functions

    additional_metrics = immutabledict({
        'pairwise_hinge_loss':
        (model_utils.pairwise_hinge_loss,
         model_utils.num_frames_div_two),
        })
    metrics = (_FULL_SEQUENCE_METRICS if split == 'test'
               else {**_TEMPORAL_REGRESSION_METRICS, **additional_metrics})

    return functools.partial(
        temporal_regression_metrics_function,
        metrics=metrics,
        predict_seconds=self.predict_seconds,
        mean_fps=self.dataset_mean_fps,
        split=split)

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
    if self.predict_seconds:
      targets = batch['targets_seconds']
    else:
      targets = batch['targets']
      # Adjust for variable frame rates.
      predictions = predictions * batch['frame_rate'] / self.dataset_mean_fps

    if self.loss_type == 'hinge_loss':
      total_loss = model_utils.pairwise_hinge_loss(
        predictions, targets, weights)
    else:
      total_loss = base_model_utils.weighted_mean_squared_error(
          predictions, targets, weights)
    if self.config.get('l2_decay_factor'):
      l2_loss = base_model_utils.l2_regularization(model_params)
      total_loss += 0.5 * self.config.l2_decay_factor * l2_loss
    return total_loss


class ResnetRegressionModel(TemporalRegressionModel):

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return resnet.ResNet(
        num_outputs=self.dataset_meta_data['num_targets'],
        num_filters=self.config.num_filters,
        num_layers=self.config.num_layers,
        dtype=model_dtype)


class SimpleCNNRegressionModel(TemporalRegressionModel):

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return simple_cnn.SimpleCNN(
        num_outputs=self.dataset_meta_data['num_targets'],
        output_projection_type=self.config.model.output_projection_type,
        num_filters=self.config.model.num_filters,
        kernel_sizes=self.config.model.kernel_sizes,
        use_bias=self.config.model.use_bias,
        dtype=model_dtype,
    )


class MultiOutputSpaceTimeViViT(vivit_model.SpaceTimeViViT):
  """Modified SpaceTimeViViT which returns both prediction and prelogits."""

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool, debug: bool = False):

    del debug
    x, _ = vivit_model.temporal_encode(
        x, self.temporal_encoding_config, self.patches, self.hidden_size,
        return_1d=False)
    bs, t, h, w, c = x.shape
    x = x.reshape(bs, t, h * w, c)

    def vit_body(x, mlp_dim, num_layers, num_heads, encoder_name='Transformer'):
      # If we want to add a class token, add it here.
      if self.classifier in ['token']:
        n, _, c = x.shape
        cls = self.param(f'cls_{encoder_name}', nn.initializers.zeros,
                         (1, 1, c), x.dtype)
        cls = jnp.tile(cls, [n, 1, 1])
        x = jnp.concatenate([cls, x], axis=1)

      x = vivit_model.Encoder(
          temporal_dims=0,  # This is unused for Factorised-Encoder
          mlp_dim=mlp_dim,
          num_layers=num_layers,
          num_heads=num_heads,
          attention_config=self.attention_config,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          stochastic_droplayer_rate=self.stochastic_droplayer_rate,
          dtype=self.dtype,
          name=encoder_name)(x, train=train)

      if self.classifier in ['token', '0']:
        x = x[:, 0]
      elif self.classifier in ('gap', 'gmp', 'gsp'):
        fn = {'gap': jnp.mean, 'gmp': jnp.max, 'gsp': jnp.sum}[self.classifier]
        x = fn(x, axis=list(range(1, x.ndim - 1)))
      return x

    # run attention across spacec, per frame
    x = jax.vmap(
        functools.partial(
            vit_body,
            mlp_dim=self.spatial_mlp_dim,
            num_layers=self.spatial_num_layers,
            num_heads=self.spatial_num_heads,
            encoder_name='SpatialTransformer'),
        in_axes=1,
        out_axes=1,
        axis_name='time')(
            x)
    assert x.ndim == 3 and x.shape[:2] == (bs, t)

    # run attention across time, over all frames
    if not self.attention_config.get('spatial_only_baseline', False):
      x = vit_body(
          x,
          mlp_dim=self.temporal_mlp_dim,
          num_layers=self.temporal_num_layers,
          num_heads=self.temporal_num_heads,
          encoder_name='TemporalTransformer')
    else:
      # Do global average pooling instead, as method of combining temporal info.
      x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))

    if self.representation_size is not None:
      x = nn.Dense(self.representation_size, name='pre_logits')(x)
      x = nn.tanh(x)
    else:
      x = nn_layers.IdentityLayer(name='pre_logits')(x)

    return_values = tuple()
    if self.return_prelogits:
      return_values = (x,) + return_values
    x = nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.zeros,
        name='output_projection')(x)
    if return_values:
      return (x,) + return_values
    else:
      return x


class ViViTRegressionModel(TemporalRegressionModel):
  """Video Transformer model for regression."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    attention_type = self.config.model.attention_config.get(
        'type', 'spacetime')
    if attention_type in [
        'spacetime', 'factorized_transformer_block',
        'factorized_self_attention_block', 'factorized_dot_product_attention'
    ]:
      return vivit_model.ViViT(
          num_classes=self.dataset_meta_data['num_targets'],
          mlp_dim=self.config.model.mlp_dim,
          num_layers=self.config.model.num_layers,
          num_heads=self.config.model.num_heads,
          representation_size=self.config.model.representation_size,
          patches=self.config.model.patches,
          hidden_size=self.config.model.hidden_size,
          temporal_encoding_config=self.config.model.temporal_encoding_config,
          attention_config=self.config.model.attention_config,
          classifier=self.config.model.classifier,
          dropout_rate=self.config.model.get('dropout_rate', 0.1),
          attention_dropout_rate=self.config.model.get(
              'attention_dropout_rate', 0.1),
          stochastic_droplayer_rate=self.config.model.get(
              'stochastic_droplayer_rate', 0),
          return_prelogits=self.config.model.get('return_prelogits', False),
          return_preclassifier=self.config.model.get(
              'return_preclassifier', False),
          dtype=model_dtype,
      )
    elif attention_type == 'factorized_encoder':
      return MultiOutputSpaceTimeViViT(
          num_classes=self.dataset_meta_data['num_targets'],
          spatial_mlp_dim=self.config.model.spatial_transformer.mlp_dim,
          spatial_num_layers=self.config.model.spatial_transformer.num_layers,
          spatial_num_heads=self.config.model.spatial_transformer.num_heads,
          temporal_mlp_dim=self.config.model.temporal_transformer.mlp_dim,
          temporal_num_layers=self.config.model.temporal_transformer
          .num_layers,
          temporal_num_heads=self.config.model.temporal_transformer.num_heads,
          representation_size=self.config.model.representation_size,
          patches=self.config.model.patches,
          hidden_size=self.config.model.hidden_size,
          temporal_encoding_config=self.config.model.temporal_encoding_config,
          attention_config=self.config.model.attention_config,
          classifier=self.config.model.classifier,
          dropout_rate=self.config.model.get('dropout_rate', 0.1),
          attention_dropout_rate=self.config.model.get(
              'attention_dropout_rate', 0.1),
          stochastic_droplayer_rate=self.config.model.get(
              'stochastic_droplayer_rate', 0),
          return_prelogits=self.config.model.get('return_prelogits', False),
          dtype=model_dtype,
      )
    else:
      raise ValueError(f'Attention type {attention_type} does not exist.')

  def init_from_train_state(self,
                            train_state: Any,
                            restored_train_state: Any,
                            restored_model_cfg: ml_collections.ConfigDict,
                            restore_output_proj: bool = False) -> Any:
    """Updates the train_state with data from restored_train_state."""
    attention_type = self.config.model.attention_config.get(
        'type', 'spacetime')
    if attention_type in [
        'spacetime', 'factorized_transformer_block',
        'factorized_self_attention_block', 'factorized_dot_product_attention'
    ]:
      vivit_transformer_key = 'Transformer'
    elif attention_type == 'factorized_encoder':
      vivit_transformer_key = 'SpatialTransformer'
    else:
      raise ValueError(f'Attention type {attention_type} does not exist.')
    return model_utils.initialise_from_train_state(
        self.config,
        train_state,
        restored_train_state,
        restored_model_cfg,
        restore_output_proj,
        vivit_transformer_key=vivit_transformer_key)
