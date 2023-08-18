"""DMVR input pipeline utilities for temporal regression datasets.
"""
from typing import Dict, Optional

from dmvr import builders
from dmvr import processors
import jax

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def add_video_length(sequence: Dict[str, tf.Tensor],
                     img_feature_name: str,
                     output_feature_name: str
                     ) -> Dict[str, tf.Tensor]:
  seq_len = tf.shape(input=sequence[img_feature_name])[0]
  sequence['video_length'] = seq_len
  return sequence


def add_history(imgs: tf.Tensor, num_stacked_frames: int) -> tf.Tensor:
  """Include num_stacked_frames of history at each time step."""
  stacked_imgs = tf.tile(tf.expand_dims(imgs[:1], 0),
                         (len(imgs), num_stacked_frames, 1, 1, 1))
  stacked_imgs = []
  for i in range(num_stacked_frames):
    first_img_repeats = tf.tile(imgs[1:], (i, 1, 1, 1))
    seq = imgs[:-i]
    stacked_imgs = (
        [tf.concat([first_img_repeats, imgs[:-i]], axis=0)] + stacked_imgs)
    # stacked_imgs[num_stacked_frames - 1 - i, i:].assign(imgs[:-i])
  stacked_imgs = tf.stack(stacked_imgs, axis=1)
  return stacked_imgs


def sample_triplet(
    sequence: Dict[str, tf.Tensor],
    history_length: int = 1,
    img_feature_name: str = builders.IMAGE_FEATURE_NAME,
    output_feature_name: str = builders.IMAGE_FEATURE_NAME + '_triplet',
    seed_feature_name: Optional[str] = None,
    seed: Optional[int] = None,
    randomize_goal = True,
    ) -> Dict[str, tf.Tensor]:
  """Sample start, current and goal frames from the sequence."""
  seq_len = tf.shape(input=sequence[img_feature_name])[0]

  if seed is None and seed_feature_name:
    seed = sequence[seed_feature_name]
    seed = tf.stack([0, seed])
  middle_seed = None
  goal_seed = None
  if seed is not None:
    seed, middle_seed, goal_seed = tfp.random.split_seed(seed, 3)
  # Draw each unique start-middle-end triplet at uniform: implemented as
  # weighted sampling of the start frame (relative to how many potential pairs
  # follow it).
  # Have I taken into account that start can be seq_len - 2, at most?
  num_successors = tf.range(seq_len - 1, 0, -1)
  # {1, ..., seq_len - 1}: seq_len - 1 items with mean
  # (1 + seq_len - 1) / 2
  num_pairs = (seq_len - 1) * seq_len / 2
  num_succ_pairs = tf.math.cumsum(num_successors, reverse=True)
  # seq_len options for 1st item, seq_len - 1 for 2nd, and seq_len - 2 for 3rd.
  # 6 permutations for a set of 3.
  num_triplets = (seq_len - 2) * (seq_len - 1) * seq_len / 6
  print('Analytical num triplets', num_triplets, 'vs sum',
        tf.reduce_sum(num_succ_pairs))
  probs = (
      tf.cast(num_succ_pairs, tf.float32) / tf.cast(num_triplets, tf.float32))
  if seed is None:
    start = tfd.Categorical(probs=probs).sample()
  else:
    start = tf.random.stateless_categorical(
        [tf.math.log(probs)], 1, seed=seed, dtype=tf.int32)[0, 0]

  # Remaning frames: seq_len - start - 1
  num_remaining_pairs = (seq_len - start - 2) * (seq_len - start - 1) / 2
  print('Analytical num pairs', num_remaining_pairs, 'vs sum',
        tf.reduce_sum(num_successors[start + 1:]))
  middle_probs = (
      tf.cast(num_successors[start + 1:], tf.float32)
      / tf.cast(num_remaining_pairs, tf.float32))
  if middle_seed is None:
    middle = tfd.Categorical(probs=middle_probs).sample()
  else:
    middle = tf.random.stateless_categorical(
        [tf.math.log(middle_probs)], 1, seed=middle_seed, dtype=tf.int32)[0, 0]

  if randomize_goal:
    if goal_seed is None:
      uniform_probs = (
          tf.ones(seq_len - (middle + 1))
          / tf.cast(seq_len - (middle + 1),  tf.float32))
      goal = tfd.Categorical(probs=uniform_probs).sample((1,)) + middle + 1
    else:
      goal = tf.random.stateless_uniform(
          (1,),
          minval=tf.cast(middle + 1, dtype=tf.int32),
          maxval=tf.cast(seq_len, dtype=tf.int32),
          dtype=tf.int32,
          seed=goal_seed)
  else:
    goal = tf.cast(seq_len - 1, dtype=tf.int32)
    goal = tf.reshape(goal, [1])

  indices = tf.concat(
      [tf.maximum(0, tf.range(start - history_length + 1, start + 1)), goal,
       tf.maximum(0, tf.range(middle - history_length + 1, middle + 1)), goal],
      axis=0)
  indices.set_shape(((history_length + 1) * 2,))
  frames = tf.gather(sequence[img_feature_name], indices)
  sequence[output_feature_name] = frames
  target_steps = [goal[0] - start, goal[0] - middle]
  target = tf.cast(target_steps, tf.float32) / sequence['frame_rate']
  sequence['targets'] = target_steps
  sequence['targets_seconds'] = target
  sequence['indices'] = [start, middle, goal[0]]
  sequence['video_length'] = seq_len
  return sequence


def sample_start_and_goal(
    sequence: Dict[str, tf.Tensor],
    history_length: int = 1,
    img_feature_name: str = builders.IMAGE_FEATURE_NAME,
    seed: Optional[int] = None,
    seed_feature_name: Optional[str] = None,
    ) -> Dict[str, tf.Tensor]:
  """Sample start and goal frames from the sequence."""
  seq_len = tf.shape(input=sequence[img_feature_name])[0]

  if seed is None and seed_feature_name:
    seed = sequence[seed_feature_name]
    seed = tf.stack([0, seed])
  goal_seed = None
  if seed is not None:
    seed, goal_seed = tfp.random.split_seed(seed, 2)
  # Draw each unique start-end frame pair at uniform: implemented as weighted
  # sampling of the start frame (relative to how many potential end frames
  # follow it) and uniform sampling of the end frame.

  num_successors = tf.range(seq_len - 1, 0, -1)
  # {1, ..., seq_len - 1}: seq_len - 1 items with mean
  # (1 + seq_len - 1) / 2
  num_pairs = (seq_len - 1) * seq_len / 2
  probs = tf.cast(num_successors, tf.float32) / tf.cast(num_pairs, tf.float32)
  if seed is None:
    start = tfd.Categorical(probs=probs).sample()
    uniform_probs = (
        tf.ones(seq_len - (start + 1))
        / tf.cast(seq_len - (start + 1),  tf.float32))
    goal = tfd.Categorical(probs=uniform_probs).sample((1,)) + start + 1
  else:
    start = tf.random.stateless_categorical(tf.math.log(probs), 1, seed=seed)
    goal = tf.random.stateless_uniform(
        (1,),
        minval=tf.cast(start + 1, dtype=tf.int32),
        maxval=tf.cast(seq_len, dtype=tf.int32),
        dtype=tf.int32,
        seed=goal_seed)

  indices = tf.concat(
      [tf.maximum(0, tf.range(start - history_length + 1, start + 1)),
       goal], axis=0)
  indices.set_shape((history_length + 1,))
  frames = tf.gather(sequence[img_feature_name], indices)
  sequence[img_feature_name] = frames
  target = tf.cast(goal - start, tf.float32) / sequence['frame_rate']
  sequence['targets'] = goal - start
  sequence['targets_seconds'] = target
  sequence['indices'] = [start, goal[0]]
  sequence['video_length'] = seq_len
  return sequence


def sample_start(
    sequence: Dict[str, tf.Tensor],
    history_length: int = 1,
    img_feature_name: str = builders.IMAGE_FEATURE_NAME,
    seed: Optional[int] = None,
    seed_feature_name: Optional[str] = None,
    ) -> Dict[str, tf.Tensor]:
  """Sample start frame from the sequence, treating the last image as a goal."""
  seq_len = tf.shape(input=sequence[img_feature_name])[0]

  if seed is None and seed_feature_name:
    seed = sequence[seed_feature_name]
    seed = tf.stack([0, seed])
  if seed is None:
    uniform_probs = tf.ones(seq_len) / tf.cast(seq_len,  tf.float32)
    start = tfd.Categorical(probs=uniform_probs).sample((1,))
    # start = tfd.Uniform(low=0, high=seq_len - 1).sample()
  else:
    start = tf.random.stateless_uniform(
        (),
        minval=tf.cast(0, dtype=tf.int32),
        maxval=tf.cast(seq_len - 1, dtype=tf.int32),
        dtype=tf.int32,
        seed=seed)
  goal = tf.convert_to_tensor([seq_len - 1], dtype=tf.int32)

  indices = tf.concat(
      [tf.maximum(0, tf.range(start - history_length + 1, start + 1)),
       goal], axis=0)
  indices.set_shape((history_length + 1,))
  frames = tf.gather(sequence[img_feature_name], indices)
  sequence[img_feature_name] = frames
  # Target in seconds.
  target = tf.cast(goal - start, tf.float32) / sequence['frame_rate']
  sequence['targets'] = goal - start
  sequence['targets_seconds'] = target
  sequence['indices'] = [start, goal[0]]
  sequence['video_length'] = seq_len
  return sequence


def add_time_index(
    sequence: Dict[str, tf.Tensor],
    img_feature_name: str = builders.IMAGE_FEATURE_NAME,
    output_feature_name: str = 'targets'):
  seq_len = tf.shape(input=sequence[img_feature_name])[0]
  sequence[output_feature_name] = tf.range(seq_len - 1)
  sequence['video_length'] = seq_len
  return sequence


def load_full_video(
    parser_builder: builders.BaseParserBuilder,
    sampler_builder: builders.SamplerBuilder,
    decoder_builder: builders.DecoderBuilder,
    preprocessor_builder: builders.PreprocessorBuilder,
    input_feature_name: str = 'image/encoded',
    output_feature_name: str = builders.IMAGE_FEATURE_NAME,
    is_training: bool = True,
    # Video related parameters.
    # num_stacked_frames: int = 1,
    stride: int = 1,
    min_resize: int = 224,
    crop_size: int = 200,
    zero_centering_image: bool = False,
    sync_random_state: bool = True,
    is_rgb: Optional[bool] = True,
    is_flow: bool = False):
  """Adds functions to load and optionally augment all frames of a video."""
  if isinstance(parser_builder, builders.SequenceExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_feature_name,
        feature_type=tf.io.FixedLenSequenceFeature((), dtype=tf.string),
        output_name=output_feature_name)
  else:
    raise ValueError('`parser_builder` has an unexpected type.')

  decoder_builder.add_fn(
      fn=lambda x: add_time_index(x, img_feature_name=output_feature_name),
      fn_name=f'add_time_index_targets')

  # Decode JPEG string to `tf.uint8`.
  # Note that for flow, 3 channels are stored in the JPEG: the first two
  # corresponds to horizontal and vertical displacement, respectively.
  # The last channel contains zeros and is dropped later in the preprocessing.
  # Hence the output number of channels for flow is 2.
  num_raw_channels = 3 if (is_rgb or is_flow) else 1
  decoder_builder.add_fn(
      fn=lambda x: processors.decode_jpeg(x, channels=num_raw_channels),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_decode_jpeg')

  # Resize images (resize happens only if necessary to save compute).
  preprocessor_builder.add_fn(
      fn=lambda x: processors.resize_smallest(x, min_resize, is_flow=is_flow),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_resize_smallest')

  if is_training:
    # Note: Random flip can be problematic for tasks with left-right asymmetry,
    # e.g. "push something from left to right".
    # Standard image data augmentation: random crop and random flip.
    preprocessor_builder.add_fn(
        # pylint: disable=g-long-lambda
        fn=lambda x, s=None: processors.crop_image(
            x, crop_size, crop_size, True, state=s),
        # pylint: enable=g-long-lambda
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_random_crop',
        # Use state to keep coherence between modalities if requested.
        stateful=sync_random_state)
    preprocessor_builder.add_fn(
        # pylint: disable=g-long-lambda
        fn=lambda x, s=None: processors.random_flip_left_right(
            x, state=s, is_flow=is_flow),
        # pylint: enable=g-long-lambda
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_random_flip',
        # Use state to keep coherence between modalities if requested.
        stateful=sync_random_state)
  else:
    # Central crop of the frames.
    preprocessor_builder.add_fn(
        fn=lambda x: processors.crop_image(x, crop_size, crop_size, False),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_central_crop')

  # Cast the frames to `tf.float32`, normalizing according to
  # `zero_centering_image`.
  preprocessor_builder.add_fn(
      fn=lambda x: processors.normalize_image(x, zero_centering_image),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_normalize')

  # if num_stacked_frames > 1:
  #   preprocessor_builder.add_fn(
  #       fn=lambda x: add_history(x, num_stacked_frames),
  #       feature_name=output_feature_name,
  #       fn_name=f'{output_feature_name}_add_history')


def sample_frames(
    parser_builder: builders.BaseParserBuilder,
    sampler_builder: builders.SamplerBuilder,
    decoder_builder: builders.DecoderBuilder,
    preprocessor_builder: builders.PreprocessorBuilder,
    input_feature_name: str = 'image/encoded',
    output_feature_name: str = builders.IMAGE_FEATURE_NAME,
    is_training: bool = True,
    # Video related parameters.
    num_frames: int = 3,
    stride: int = 1,
    sample_triplets: bool = False,
    min_resize: int = 224,
    crop_size: int = 200,
    zero_centering_image: bool = False,
    sync_random_state: bool = True,
    augment_goals: bool = True,
    is_rgb: Optional[bool] = True,
    is_flow: bool = False):
  """Adds functions to process start and goal images to builders."""
  if isinstance(parser_builder, builders.SequenceExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_feature_name,
        feature_type=tf.io.FixedLenSequenceFeature((), dtype=tf.string),
        output_name=output_feature_name)
  else:
    raise ValueError('`parser_builder` has an unexpected type.')

  # TODO(minttu): Support stride.
  if stride != 1:
    raise NotImplementedError('stride != 1 not supported.')
  # Num_frames includes goal frame.
  if augment_goals:
      # pylint: disable=g-long-lambda
    if sample_triplets:
      decoder_builder.add_fn(
          fn=lambda x: sample_triplet(
              x, num_frames - 1, output_feature_name,
              output_feature_name,  # + '_triplet',
              seed_feature_name=None if is_training else 'video_id'),
          fn_name=f'{output_feature_name}_sample_random_goal_triplet',)
    else:
      decoder_builder.add_fn(
          fn=lambda x: sample_start_and_goal(
              x, num_frames - 1, output_feature_name,
              seed_feature_name=None if is_training else 'video_id'),
          fn_name=f'{output_feature_name}_sample_random_goal_pair',)
      # pylint: enable=g-long-lambda
  else:
    # TODO: Pass randomize_goal = False as an arg?
    if sample_triplets:
      decoder_builder.add_fn(
          fn=lambda x: sample_triplet(x, num_frames - 1,
              output_feature_name, output_feature_name,  # + '_triplet',
              seed_feature_name=None if is_training else 'video_id',
              randomize_goal=False),
          fn_name=f'{output_feature_name}_sample_last_goal_triplet',)
    else:
      decoder_builder.add_fn(
          fn=lambda x: sample_start(
              x, num_frames - 1, output_feature_name,
              seed_feature_name=None if is_training else 'video_id'),
          fn_name=f'{output_feature_name}_sample_last_goal_pair',)

  # Decode JPEG string to `tf.uint8`.
  # Note that for flow, 3 channels are stored in the JPEG: the first two
  # corresponds to horizontal and vertical displacement, respectively.
  # The last channel contains zeros and is dropped later in the preprocessing.
  # Hence the output number of channels for flow is 2.
  num_raw_channels = 3 if (is_rgb or is_flow) else 1
  decoder_builder.add_fn(
      fn=lambda x: processors.decode_jpeg(x, channels=num_raw_channels),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_decode_jpeg')

  # Resize images (resize happens only if necessary to save compute).
  preprocessor_builder.add_fn(
      fn=lambda x: processors.resize_smallest(x, min_resize, is_flow=is_flow),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_resize_smallest')

  if is_training:
    # Note: Random flip can be problematic for tasks with left-right asymmetry,
    # e.g. "push something from left to right".
    # Standard image data augmentation: random crop and random flip.
    preprocessor_builder.add_fn(
        # pylint: disable=g-long-lambda
        fn=lambda x, s=None: processors.crop_image(
            x, crop_size, crop_size, True, state=s),
        # pylint: enable=g-long-lambda
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_random_crop',
        # Use state to keep coherence between modalities if requested.
        stateful=sync_random_state)
    preprocessor_builder.add_fn(
        # pylint: disable=g-long-lambda
        fn=lambda x, s=None: processors.random_flip_left_right(
            x, state=s, is_flow=is_flow),
        # pylint: enable=g-long-lambda
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_random_flip',
        # Use state to keep coherence between modalities if requested.
        stateful=sync_random_state)
  else:
    # Central crop of the frames.
    preprocessor_builder.add_fn(
        fn=lambda x: processors.crop_image(x, crop_size, crop_size, False),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_central_crop')

  # Cast the frames to `tf.float32`, normalizing according to
  # `zero_centering_image`.
  preprocessor_builder.add_fn(
      fn=lambda x: processors.normalize_image(x, zero_centering_image),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_normalize')


