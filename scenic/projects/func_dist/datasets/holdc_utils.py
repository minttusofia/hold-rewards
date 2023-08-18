def _get_random_sampling_offset(
    sequence: Dict[str, tf.Tensor],
    num_steps: int,
    stride: int,
    seed: Optional[int] = None,
    img_feature_name: str = builders.IMAGE_FEATURE_NAME,
    seed_feature_name: Optional[str] = None,
    ) -> tf.Tensor:
  """Calculates the initial offset for a sequence where all steps will fit.
  Args:
    sequence: Any tensor where the first dimension is timesteps.
    num_steps: Number of steps (e.g. frames) to take.
    stride: Distance to sample between timesteps.
    seed: A deterministic seed to use when sampling.
  Returns:
    The first index to begin sampling from. A best effort is made to provide a
    starting index such that all requested steps fit within the sequence (i.e.
    `offset + 1 + (num_steps - 1) * stride` < len(sequence)`). If this is not
    satisfied, the starting index is always 0.
  """
  if seed is None and seed_feature_name:
    seed = sequence[seed_feature_name]
  sequence_length = tf.shape(input=sequence)[0]
  max_offset = tf.maximum(sequence_length - (num_steps - 1) * stride, 1)
  return tf.random.uniform((),
                           maxval=tf.cast(max_offset, dtype=tf.int32),
                           dtype=tf.int32,
                           seed=seed)


def sample_or_pad_sequence_indices(sequence: tf.Tensor, num_steps: int,
                                   repeat_sequence: bool, stride: int,
                                   offset: int) -> tf.Tensor:
  """Returns indices to take for sampling or padding a sequence to fixed size.
  Args:
    sequence: Any tensor where the first dimension is timesteps.
    num_steps: Number of steps (e.g. frames) to take.
    repeat_sequence: A boolean indicates whether the sequence will repeat to
      have enough steps for sampling. If `False`, a runtime error is thrown if
      `num_steps` * `stride` is longer than sequence length.
    stride: Distance to sample between timesteps.
    offset: Offset(s) to be used for sampling.
  Returns:
    Indices to gather from the sequence tensor to get a fixed size sequence.
  """
  sequence_length = tf.shape(input=sequence)[0]
  sel_idx = tf.range(sequence_length)

  if repeat_sequence:
    # Repeats sequence until `num_steps` are available in total.
    num_repeats = tf.cast(
        tf.math.ceil(
            tf.divide(
                tf.cast(num_steps * stride + offset, dtype=tf.float32),
                tf.cast(sequence_length, dtype=tf.float32))), dtype=tf.int32)
    sel_idx = tf.tile(sel_idx, [num_repeats])
  steps = tf.range(offset, offset + num_steps * stride, stride)

  return tf.gather(sel_idx, steps)

def sample_sequence(
    sequence: tf.Tensor,
    num_steps: int,
    random: bool,
    stride: int = 1,
    seed: Optional[int] = None,
    state: Optional[builders.ProcessorState] = None) -> tf.Tensor:
  """Samples a single segment of size `num_steps` from a given sequence.
  If `random` is not `True`, this function will simply sample the central window
  of the sequence. Otherwise, a random offset will be chosen in a way that the
  desired `num_steps` might be extracted from the sequence.
  In order to keep coherence among different sequences sampled using random true
  (e.g. image and audio), an optional state is accepted as parameter and used to
  keep track of the first offset, using a proportional offset to sample from the
  second sequence.
  Args:
    sequence: Any tensor where the first dimension is timesteps.
    num_steps: Number of steps (e.g. frames) to take.
    random: A boolean indicating whether to random sample the single window. If
      `True`, the offset is randomized. If `False`, the middle frame minus half
      of `num_steps` is the first frame.
    stride: Distance to sample between timesteps.
    seed: A deterministic seed to use when sampling.
    state: A mutable dictionary where keys are strings. The dictionary might
      contain 'sample_offset_proportion' as key with metadata useful for
      sampling. It will be modified with added metadata if needed. This can be
      used to keep consistency between sampling of different sequences.
  Returns:
    A single tensor with first dimension `num_steps` with the sampled segment.
  """
  sequence_length = tf.shape(input=sequence)[0]
  sequence_length = tf.cast(sequence_length, tf.float32)

  if random:
    if state and 'sample_offset_proportion' in state:
      # Read offset from state to ensure consistent offsets for different
      # modalities.
      offset = state['sample_offset_proportion'] * sequence_length
      offset = tf.cast(tf.math.round(offset), tf.int32)
    else:
      offset = _get_random_sampling_offset(
          sequence=sequence,
          num_steps=num_steps,
          stride=stride,
          seed=seed)

      if state is not None:
        # Update state.
        sample_offset_proportion = tf.cast(offset, tf.float32) / sequence_length
        state['sample_offset_proportion'] = sample_offset_proportion

  else:
    offset = tf.maximum(
        0, tf.cast((sequence_length - num_steps * stride) // 2, tf.int32))

  indices = sample_or_pad_sequence_indices(
      sequence=sequence,
      num_steps=num_steps,
      repeat_sequence=True,  # Will repeat the sequence if request more.
      stride=stride,
      offset=offset)
  indices.set_shape((num_steps,))
  output = tf.gather(sequence, indices)

  return output


def add_image(
    parser_builder: builders.BaseParserBuilder,
    sampler_builder: builders.SamplerBuilder,
    decoder_builder: builders.DecoderBuilder,
    preprocessor_builder: builders.PreprocessorBuilder,
    postprocessor_builder: builders.PostprocessorBuilder,
    input_feature_name: str = 'image/encoded',
    output_feature_name: str = builders.IMAGE_FEATURE_NAME,
    is_training: bool = True,
    # Video related parameters.
    num_frames: int = 32,
    stride: int = 1,
    num_test_clips: int = 1,
    min_resize: int = 224,
    crop_size: int = 200,
    zero_centering_image: bool = False,
    sync_random_state: bool = True,
    is_rgb: Optional[bool] = True,
    is_flow: bool = False,
    random_flip: bool = True):
  """Adds functions to process image feature to builders.
  This function expects the input to be either a `tf.train.SequenceExample` (for
  videos) and have the following structure:
  ```
  feature_lists {
    feature_list {
      key: input_feature_name
      value {
        feature {
          bytes_list {
            value: jpeg_bytes
          }
        }
      }
    }
  }
  ```
  Or a `tf.train.Example` (for image only) and have the following structure:
  ```
  features {
    feature {
      key: input_feature_name
      value {
        bytes_list {
          value: "JPEG"
        }
      }
    }
  }
  ```
  The corresponding `builders.ExampleParserBuilder` or
  `builders.SequenceExampleParserBuilder` has to be given as parameter.
  Args:
    parser_builder: An instance of a `builders.BaseParserBuilder`.
    sampler_builder: An instance of a `builders.SamplerBuilder`.
    decoder_builder: An instance of a `builders.DecoderBuilder`.
    preprocessor_builder: An instance of a `builders.PreprocessorBuilder`.
    postprocessor_builder: An instance of a `builders.PostprocessorBuilder`.
    input_feature_name: Name of the feature in the input `tf.train.Example` or
      `tf.train.SequenceExample`. Exposing this as an argument allows using this
      function for different image features within a single dataset.
    output_feature_name: Name of the feature in the output features dictionary.
      Exposing this as an argument allows using this function for different
      image features within a single dataset.
    is_training: Whether or not in training mode. If `True`, random sample, crop
      and left right flip is used.
    num_frames: Number of frames per subclip. For single images, use 1.
    stride: Temporal stride to sample frames.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each video at test time.
      If 1, then a single clip in the middle of the video is sampled. The clips
      are aggreagated in the batch dimension.
    min_resize: Frames are resized so that `min(height, width)` is `min_resize`.
    crop_size: Final size of the frame after cropping the resized frames. Both
      height and width are the same.
    zero_centering_image: If `True`, frames are normalized to values in [-1, 1].
      If `False`, values in [0, 1].
    sync_random_state: Whether to use stateful option to keep random operations
      in sync between different modalities. All modalities having this option
      `True` will use the same outcome in random operations such as sampling and
      cropping.
    is_rgb: If `True`, the number of channels in the JPEG is 3, if False, 1.
      If is_flow is `True`, `is_rgb` should be set to `None` (see below).
    is_flow: If `True`, the image is assumed to contain flow and will be
      processed as such. Note that the number of channels in the JPEG for flow
      is 3, but only two channels will be output corresponding to the valid
      horizontal and vertical displacement.
    random_flip: If `True`, a random horizontal flip is applied to the input
      image. This augmentation may not be used if the label set contains
      direction related classes, such as `pointing left`, `pointing right`, etc.
  """

  # Validate parameters.
  if is_flow and is_rgb is not None:
    raise ValueError('`is_rgb` should be `None` when requesting flow.')

  if is_flow and not zero_centering_image:
    raise ValueError('Flow contains displacement values that can be negative, '
                     'but `zero_centering_image` was set to `False`.')

  if is_training and num_test_clips != 1:
    logging.info('`num_test_clips` %d is ignored since `is_training` is true.',
                 num_test_clips)

  # Parse frames or single image.
  if isinstance(parser_builder, builders.SequenceExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_feature_name,
        feature_type=tf.io.FixedLenSequenceFeature((), dtype=tf.string),
        output_name=output_feature_name)
  elif isinstance(parser_builder, builders.ExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_feature_name,
        feature_type=tf.io.FixedLenFeature((), dtype=tf.string),
        output_name=output_feature_name)
    # Expand dimensions so single images have the same structure as videos.
    sampler_builder.add_fn(
        fn=lambda x: tf.expand_dims(x, axis=0),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_expand_dims')
  else:
    raise ValueError('`parser_builder` has an unexpected type.')

  # Temporal sampler.
  if is_training:
    # Sample random clip.
    sampler_builder.add_fn(
        # pylint: disable=g-long-lambda
        fn=lambda x, s=None: sample_sequence(
            x, num_frames, True, stride, state=s,
            # TODO: Finish (left off here)
            seed_feature_name=None if is_training else 'video_id'),
        # pylint: enable=g-long-lambda
        # feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_random_sample',
        # Use state to keep coherence between modalities if requested.
        stateful=sync_random_state)
  else:
    if num_test_clips > 1:
      # Sample linspace clips.
      sampler_builder.add_fn(
          # pylint: disable=g-long-lambda
          fn=lambda x: processors.sample_linspace_sequence(
              x, num_test_clips, num_frames, stride),
          # pylint: enable=g-long-lambda
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_linspace_sample')
    else:
      # Sample middle clip.
      sampler_builder.add_fn(
          fn=lambda x: processors.sample_sequence(x, num_frames, False, stride),
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_middle_sample')

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

  if is_flow:
    # Cast the flow to `tf.float32`, normalizing between [-1.0, 1.0].
    preprocessor_builder.add_fn(
        fn=lambda x: processors.normalize_image(x, zero_centering_image=True),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_normalize')

  # Resize images (resize happens only if necessary to save compute).
  preprocessor_builder.add_fn(
      fn=lambda x: processors.resize_smallest(x, min_resize, is_flow=is_flow),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_resize_smallest')

  if is_training:
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
    if random_flip:
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

  if is_flow:
    # Keep only two channels for the flow: horizontal and vertical displacement.
    preprocessor_builder.add_fn(
        fn=lambda x: x[:, :, :, :2],
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_extract_flow_channels')

    # Clip the flow to stay between [-1.0 and 1.0]
    preprocessor_builder.add_fn(
        fn=lambda x: tf.clip_by_value(x, -1.0, 1.0),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_clip_flow')
  else:
    # Cast the frames to `tf.float32`, normalizing according to
    # `zero_centering_image`.
    preprocessor_builder.add_fn(
        fn=lambda x: processors.normalize_image(x, zero_centering_image),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_normalize')

  if num_test_clips > 1 and not is_training:
    # In this case, multiple clips are merged together in batch dimenstion which
    # will be `B * num_test_clips`.
    postprocessor_builder.add_fn(
        fn=lambda x: tf.reshape(  # pylint: disable=g-long-lambda
            x, (-1, num_frames, x.shape[2], x.shape[3], x.shape[4])),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_reshape')
