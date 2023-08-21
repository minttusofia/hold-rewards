"""Extend scenic.train_lib.train_utils with custom datasets.
"""
from typing import Any, Optional

import flax
from flax import optim
import jax.numpy as jnp


# from scenic.projects.func_dist.datasets import ssv2_regression  # pylint: disable=unused-import
# from scenic.train_lib import train_utils

# get_dataset = train_utils.get_dataset
# TrainState = train_utils.TrainState

@flax.struct.dataclass
class TrainState:
  """Dataclass to keep track of state of training.

  The state of training is structured as a flax.struct.dataclass, which enables
  instances of this class to be passed into jax transformations like tree_map
  and pmap.
  """
  global_step: Optional[int] = 0 
  optimizer: Optional[optim.Optimizer] = None
  model_state: Optional[Any] = None
  rng: Optional[jnp.ndarray] = None
  accum_train_time: Optional[int] = 0 

  def __getitem__(self, item):
    """Make TrainState a subscriptable object."""
    return getattr(self, item)

  def get(self, keyname: str, default: Optional[Any] = None) -> Any:
    """Return the value for key if it exists otherwise the default."""
    try:
      return self[keyname]
    except KeyError:
      return default

