"""Main file for learning functional distances."""

from typing import Any, Callable
import copy
import glob
import os

from datetime import datetime

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.func_dist import model
from scenic.projects.func_dist import train_utils
from scenic.projects.func_dist import trainer as func_dist_trainer
from scenic.projects.func_dist import holdc_trainer
from scenic.projects.func_dist import test_ckpts

flags.DEFINE_boolean('test_only', False,
                     'If True, only test saved checkpoints.')
FLAGS = flags.FLAGS


def get_trainer(trainer_name: str) -> Callable[..., Any]:
  """Returns trainer given its name."""
  if trainer_name == 'func_dist_trainer':
    return func_dist_trainer.train
  if trainer_name == 'holdc_trainer':
    return holdc_trainer.train
  raise ValueError(f'Unsupported trainer: {trainer_name}.')


def get_tester(trainer_name: str) -> Callable[..., Any]:
  """Returns test function given its name."""
  if trainer_name == 'func_dist_trainer':
    return test_ckpts.test
  if trainer_name == 'holdc_trainer':
    return test_ckpts.test
  raise ValueError(f'Unsupported trainer: {trainer_name}.')


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for learning functional distances."""
  del writer
  # if not FLAGS.test_only:
  #   for empty_summary in glob.glob(os.path.join(workdir, 'events.out.tfevents*')):
  #     os.remove(empty_summary)

  model_cls = model.get_model_cls(config.model_name)
  data_rng1, data_rng2, rng = jax.random.split(rng, 3)
  dataset = train_utils.get_dataset(
      config, data_rng1, dataset_service_address=FLAGS.dataset_service_address)
  if config.get('test_dataset_name', None):
    test_dataset_config = copy.deepcopy(config)
    test_dataset_config.dataset_name = config.test_dataset_name
    test_dataset = train_utils.get_dataset(
        test_dataset_config, data_rng2,
        dataset_service_address=FLAGS.dataset_service_address)
  else:
      test_dataset = None
  if FLAGS.test_only:
    trainer = get_tester(config.trainer_name)
    summary_dir = os.path.join(workdir, 'test')
  else:
    trainer = get_trainer(config.trainer_name)
    if 'OAR_JOB_ID' in os.environ:
      job_id = os.environ['OAR_JOB_ID']
    else:
      job_id = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    workdir = os.path.join(workdir, job_id)
    summary_dir = workdir
  writer = metric_writers.create_default_writer(
      summary_dir, just_logging=jax.process_index() > 0, asynchronous=True)

  trainer(
      rng=rng,
      config=config,
      model_cls=model_cls,
      dataset=dataset,
      test_dataset=test_dataset,
      workdir=workdir,
      writer=writer)


if __name__ == '__main__':
  app.run(main=main)
