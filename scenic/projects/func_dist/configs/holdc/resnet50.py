r"""Config for training HOLD-C with ResNet-50 architecture.

"""


import os
from absl import logging
import ml_collections

NUM_CLASSES = 174
SSV2_TRAIN_SIZE = 68913
SSV2_VAL_SIZE = 24777

DATA_DIR = '/PATH/TO/DATA_DIR'  # Set the data directory.
NUM_DEVICES = 1  # Set the number of devices.


def get_config():
  """Returns the base experiment configuration."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'holdc_resnet_ssv2'

  # Dataset.
  config.dataset_configs = ml_collections.ConfigDict()
  config.data_dtype_str = 'float32'
  config.dataset_name = 'video_tfrecord_dataset'
  config.test_dataset_name = 'ssv2_regression_tfrecord'
  config.dataset_configs.base_dir = os.path.join(DATA_DIR, '20bn-something-something-v2/tfrecords')
  config.dataset_configs.tables = {
      'train': 'something-something-v2-train.rgb.tfrecord@128',
      'validation': 'something-something-v2-validation.rgb.tfrecord@128',
      'test': 'something-something-v2-validation.rgb.tfrecord@128',
  }
  config.dataset_configs.examples_per_subset = {
      'train': SSV2_TRAIN_SIZE,
      'validation': SSV2_VAL_SIZE,
      'test': SSV2_VAL_SIZE,
  }
  config.dataset_configs.num_classes = NUM_CLASSES
  config.dataset_configs.included_tasks_path = None
  config.dataset_configs.train_metadata_path = None
  config.dataset_configs.validation_metadata_path = None

  # This is going to sample 3 consecutive frames and a future goal frame.
  config.dataset_configs.num_frames = 32
  config.dataset_configs.stride = 1
  config.dataset_configs.min_resize = 288
  config.dataset_configs.crop_size = 256
  config.dataset_configs.zero_centering = True

  # Multicrop eval settings
  # In multi-crop testing, we assume that num_crops consecutive entries in the
  # batch are from the same example, and average the logits over these examples.
  config.dataset_configs.do_multicrop_test = False
  config.dataset_configs.eval_test_metrics = False  # Do during training.
  config.dataset_configs.log_test_epochs = 1
  # The effective batch size per host when testing is num_test_clips * test_batch_size  # pylint: disable=line-too-long
  config.dataset_configs.num_test_clips = 1
  config.dataset_configs.test_batch_size = 128
  # Leaving this empty means that a full test is done each time.
  # config.steps_per_test = 1000  # Number of test steps taken by each host.

  config.dataset_configs.augmentation_params = ml_collections.ConfigDict()
  config.dataset_configs.augmentation_params.do_jitter_scale = True
  config.dataset_configs.augmentation_params.scale_min_factor = 0.9
  config.dataset_configs.augmentation_params.scale_max_factor = 1.33
  config.dataset_configs.augmentation_params.prob_scale_jitter = 1.0
  config.dataset_configs.augmentation_params.do_color_augment = False
  config.dataset_configs.augmentation_params.prob_color_augment = 0.8
  config.dataset_configs.augmentation_params.prob_color_drop = 0.1

  # Only used for test_dataset
  config.dataset_configs.augmentation_params.augment_goals = True

  # This does Mixup in the data-loader. Done on Numpy CPU, so its slow
  # config.dataset_configs.augmentation_params.do_mixup = False
  # config.dataset_configs.augmentation_params.mixup_alpha = 0.0

  # This does Mixup in the train loop. This is fast. But make sure that device
  # batch size is more than 1. On a 4x4 TPU, this means that your batch size
  # needs to be at least 64.
  # For Kinetics, we have not been using Mixup
  # config.mixup = ml_collections.ConfigDict()
  # config.mixup.alpha = 0.3

  config.dataset_configs.augmentation_params.do_rand_augment = True
  config.dataset_configs.augmentation_params.rand_augment_num_layers = 2
  config.dataset_configs.augmentation_params.rand_augment_magnitude = 20

  config.dataset_configs.prefetch_to_device = 2

  config.model_name = 'resnet_holdc'
  config.model = ml_collections.ConfigDict()
  config.num_filters = 64
  config.num_layers = 50
  config.embedding_size = 32
  config.model_dtype_str = 'float32'
  config.pos_radius = 2
  config.neg_radius = 2 * config.pos_radius
  config.pos_neg_margin = 0.2

  config.predict_seconds = True

  # Training.
  config.trainer_name = 'holdc_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.l2_decay_factor = None
  config.max_grad_norm = None
  config.label_smoothing = None
  config.num_training_epochs = 100
  config.batch_size = NUM_DEVICES
  config.rng_seed = 0

  # Learning rate.
  steps_per_epoch = SSV2_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = steps_per_epoch // 2
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = 1e-4

  # Logging.
  config.write_summary = True
  config.checkpoint = True  # Do checkpointing.
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.
  config.checkpoint_steps = 500  # Checkpoint more frequently than a val epoch
  config.log_summary_steps = 100

  # config.do_memory_defrag = True

  return config

