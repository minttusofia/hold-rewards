"""Evaluate saved checkpoints for Functional Distances."""

import copy
import collections
import functools
import json
import operator
import os
import pickle
import sys
import time
from typing import Any, Dict, Optional, Tuple

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
from immutabledict import immutabledict
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.projects.vivit import evaluation_lib
from scenic.projects.vivit import train_utils as vivit_train_utils
from scenic.projects.func_dist import model_utils as func_dist_model_utils
from scenic.projects.func_dist import train_utils as func_dist_train_utils
from scenic.projects.func_dist import model as func_dist_model
from scenic.projects.func_dist import holdc_model
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils


def log_all_summaries(writer, best_test_metrics):
  summary_per_step = collections.defaultdict(dict)
  for k, v in best_test_metrics.items():
    for score_step in v:
      key = f'test/{k}'
      score, step = score_step
      summary_per_step[step][key] = score
  for step, summary in sorted(summary_per_step.items()):
    writer.write_scalars(step, summary)
  writer.flush()


def glob_checkpoints(workdir, train_metric):
  return glob.glob(
      os.path.join(workdir, f'best_val_{train_metric}', 'checkpoint_*'))


def get_next_best_step(workdir, train_metric, test_scores):
  best_val_metrics = func_dist_train_utils.load_json_metrics(workdir, 'val')
  tested_steps = [score_step[1] for score_step in test_scores]
  for score_step in best_val_metrics[train_metric]:
    score, ckpt_step = score_step
    if ckpt_step > 1 and not ckpt_step in tested_steps:
      step = ckpt_step
      logging.info(
          f'Testing step {step} with validation {train_metric} {score}')
      return step
  logging.info(f'No checkpoints left to test')
  return None


def get_already_tested(workdir):
  already_tested = {}
  test_path = os.path.join(workdir, 'test_results.pkl')
  if os.path.exists(test_path):
    with open(test_path, 'rb') as f:
      already_tested = pickle.load(f)
  return already_tested


def save_already_tested(workdir, already_tested):
  test_path = os.path.join(workdir, 'test_results.pkl')
  with open(test_path, 'wb') as f:
    pickle.dump(already_tested, f)


def test(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Any,
    dataset: dataset_utils.Dataset,
    test_dataset: Optional[dataset_utils.Dataset] = None,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Tuple[train_utils.TrainState, Dict[str, Any], Dict[str, Any]]:
  """Evaluate previously saved checkpoints in workdir.

  Given the model class and dataset, it prepares the items needed to run the
  training, including the TrainState.

  Args:
    rng: Jax rng key.
    config: Configurations of the experiment.
    model_cls: Model class; A model has a flax_module, a loss_fn, and a
      metrics_fn associated with it.
    dataset: The dataset that has train_iter, eval_iter, meta_data, and
      optionally, test_iter.
    workdir: Directory for checkpointing.
    writer: CLU metrics writer instance.

  Returns:
    train_state that has the state of training (including current
      global_step, model_state, rng, and the optimizer), train_summary
      and eval_summary which are dict of metrics. These outputs are used for
      regression testing.
  """
  lead_host = jax.process_index() == 0
  # Build the loss_fn, metrics, and flax_model.
  model = model_cls(config, dataset.meta_data)
  is_multilabel_model = (config.model_name == 'vivit_multilabel_classification')
  get_confusion_matrix = (config.get('confusion_matrix_metrics', False)
                          and not is_multilabel_model)

  dataset = test_dataset or dataset
  # Initialize model.
  rng, init_rng = jax.random.split(rng)
  (params, model_state, num_trainable_params,
   gflops) = train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=[(dataset.meta_data['input_shape'],
                    dataset.meta_data.get('input_dtype', jnp.float32))],
       config=config,
       rngs=init_rng)

  # Create optimizer.
  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  optimizer = jax.jit(
      optimizers.get_optimizer(config).create, backend='cpu')(params)
  rng, train_rng = jax.random.split(rng)
  train_state = train_utils.TrainState(
      global_step=0,
      optimizer=optimizer,
      model_state=model_state,
      rng=train_rng,
      accum_train_time=0)
  start_step = train_state.global_step

  # Replicate the optimizier, state, and rng.
  train_state = jax_utils.replicate(train_state)
  del params  # Do not keep a copy of the initial params.

  # Calculate the total number of training steps.
  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data)

  test_batch_size = config.dataset_configs.get('test_batch_size')

  is_holdc_model = isinstance(model, holdc_model.TemporalContrastiveModel)
  num_stacked_frames = (
      1 if is_holdc_model else config.dataset_configs.get('num_frames') - 1)
  test_step_pmapped = jax.pmap(
      functools.partial(
          func_dist_train_utils.full_seq_test_step,
          flax_model=model.flax_model,
          num_stacked_frames=num_stacked_frames,
          metrics_fn=model.get_metrics_fn('test'),
          test_batch_size=test_batch_size,
          return_logits_and_labels=is_multilabel_model,
          return_confusion_matrix=get_confusion_matrix,
          # n_clips=config.get('multicrop_clips_per_device', 2),
          compute_distances=is_holdc_model,
          debug=config.debug_eval),
      axis_name='batch',
      # We can donate the test_batch's buffer.
      # donate_argnums=(1,),
    )

  videos_per_test_step = 1
  total_test_steps = int(
      np.ceil(dataset.meta_data['num_test_examples'] /
              (videos_per_test_step *
               config.get('dataset_configs.num_test_clips') *
               jax.process_count())))
  steps_per_test = config.get('steps_per_test') or total_test_steps

  train_metrics, extra_training_logs = [], []
  train_summary, eval_summary = None, None

  # chrono = train_utils.Chrono(
  #     first_step=start_step,
  #     total_steps=total_steps,
  #     steps_per_epoch=steps_per_epoch,
  #     global_bs=config.batch_size,
  #     accum_train_time=int(jax_utils.unreplicate(train_state.accum_train_time)))

  # Manually defragment memory before starting training, if we are using the
  # tfrt runtime.
  do_memory_defrag = False
  if config.get('do_memory_defrag', False):
    client = jax.lib.xla_bridge.get_backend()
    try:
      logging.info('Defragmenting memory')
      client.defragment()
      do_memory_defrag = True
    except RuntimeError:
      logging.warn('Memory defragmentation not possible, use the tfrt runtime')

  if is_holdc_model:
    train_metric = 'triplet_svtc_loss'
  else:
    train_metric = ('mean_squared_error_seconds' if model.predict_seconds
                    else 'mean_squared_error_steps')
  test_metric = 'misclassification_rate'
  best_test_metrics = func_dist_train_utils.load_json_metrics(
      workdir, 'testonly')
  logging.info(f'Already tested: {best_test_metrics[test_metric]}')
  # Evaluation metrics where lower is better. Higher is better is assumed for 
  # the rest.
  loss_metrics = [
      'mean_absolute_error', 'mean_squared_error',
      'mean_absolute_error_steps', 'mean_squared_error_steps',
      'mean_absolute_error_seconds', 'mean_squared_error_seconds',
      'triplet_svtc_loss',
      'misclassification_rate', 'hinge_loss',
  ]
  n_ckpts_to_keep = config.get('num_training_epochs')
  logging.info(f'Num ckpts to keep: {n_ckpts_to_keep}')

  # additional_metrics_fn = functools.partial(
  #     func_dist_model.temporal_regression_metrics_function,
  #     metrics=immutabledict({
  #         'spearman_correlation': (
  #             func_dist_model_utils.spearman_correlation,
  #             holdc_model.num_videos)
  #     }),
  #     split='test',
  #     pmapped=False)
  wait_for_ckpts = False

  step = get_next_best_step(
      workdir, train_metric, best_test_metrics[test_metric])

  wait_counter = 0
  while wait_for_ckpts or step is not None:
    if step is None:
      if len(best_test_metrics[test_metric]) >= n_ckpts_to_keep:
          break
      time.sleep(10)
      if wait_counter % 12 == 0:
        logging.info(f'Waiting {wait_counter * 10}s for new checkpoints')
      wait_counter += 1
    else:
      wait_counter = 0
    train_state, start_step = train_utils.restore_checkpoint(
        os.path.join(workdir, f'best_val_{train_metric}'), train_state,
        assert_exist=True, step=step)
    logging.info(f'Restored start step {start_step}')
    train_state = jax_utils.replicate(train_state)
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=steps_per_test, writer=writer)
    hooks = [report_progress]
    if config.get('xprof', True) and lead_host:
      hooks.append(periodic_actions.Profile(
          num_profile_steps=25, logdir=workdir))

    ############# TESTING ############################
    with report_progress.timed('test'):
      if do_memory_defrag:
        logging.info('Defragmenting memory')
        client.defragment()

      logging.info(f'Starting test for step {step}')
      already_tested = get_already_tested(workdir)
      test_metrics = list(already_tested.values())
      print('Already tested', len(already_tested), 'videos')
      for test_step in range(steps_per_test):
        with report_progress.timed('test: load_batch'):
          test_batch = next(dataset.test_iter)
        sys.stdout.flush()
        sys.stderr.flush()
        # import pdb; pdb.set_trace()
        assert len(test_batch['video_id']) == 1 and len(test_batch['video_id'][0]) == 1
        vid_id = int(test_batch['video_id'][0][0])
        if vid_id in already_tested:
          # print('Already tested', test_batch['video_id'][0])
          continue
        # print('step:', test_step, ':', test_batch['video_length'])
        with report_progress.timed('test: get_batch_mask'):
          test_batch['batch_mask'] = (
              func_dist_train_utils.mask_from_video_lengths(
                  jnp.transpose(test_batch['inputs'], (0, 2, 1, 3, 4, 5)),
                  test_batch['video_length']))
        test_batch['targets'] = jnp.transpose(test_batch['targets'], (0, 2, 1))
        with report_progress.timed('test: test_pmapped'):
          t_metrics, preds = test_step_pmapped(train_state, test_batch)
        # with report_progress.timed('test: spearman'):
        #   additional_metrics = additional_metrics_fn(preds, test_batch)
        with report_progress.timed('test: unreplication'):
          # Fetch t_metrics to host and store.
          t_metrics = train_utils.unreplicate_and_get(t_metrics)
        # t_metrics = {**t_metrics, **additional_metrics}
        already_tested[vid_id] = t_metrics
        test_metrics.append(t_metrics)
        save_already_tested(workdir, already_tested)

        for h in hooks:
          # Catch exception in case XProf fails.
          try:
            h(test_step)
          except ValueError as error:
            logging.exception('Hook failed: %r', error)

      # test_metrics = jax.tree_map(train_utils.unreplicate_and_get, test_metrics)
      # Log test summary.
      test_summary = train_utils.log_eval_summary(
          step=step,
          eval_metrics=test_metrics,
          writer=writer,
          prefix='test',
          key_separator='/')
      logging.info(f'Completed test for step {step}')
      writer.flush()

      best_test_metrics, _ = func_dist_train_utils.compare_eval_scores(
          test_summary, best_test_metrics, n_ckpts_to_keep, loss_metrics, step,
          'test')
      # Reload in case the metrics were updated.
      best_test_metrics_from_file = func_dist_train_utils.load_json_metrics(
          workdir, 'testonly')
      best_test_metrics = func_dist_train_utils.add_eval_scores(
          best_test_metrics_from_file, best_test_metrics, n_ckpts_to_keep,
          loss_metrics)
      log_all_summaries(writer, best_test_metrics)
      for k, v in best_test_metrics.items():
        print(k)
        for score_step in v:
          print(f'{score_step[1]} : {score_step[0]:.3f}')
      with open(os.path.join(workdir, f'best_testonly_metrics.json'), 'w') as f:
        json.dump(best_test_metrics, f)

      # Free up some space.
      del test_metrics
      if do_memory_defrag:
        logging.info('Defragmenting memory')
        client.defragment()

    step = get_next_best_step(
        workdir, train_metric, best_test_metrics[test_metric])
    # chrono.resume()  # un-pause now

  logging.info(
      f'Finished testing {len(best_test_metrics[test_metric])} checkpoints')
  # Wait until computations are done before exiting.
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
  # Return the train and eval summary after last step for regression testing.
  return train_state, train_summary, eval_summary
