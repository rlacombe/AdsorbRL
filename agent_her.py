
"""DQN agent HER implementation."""

import copy
from typing import Optional
from acme import types

from acme import datasets
from acme import specs
from acme.adders import reverb as adders
from acme.agents import agent
from acme.agents.tf import actors
from acme.agents.tf.dqn import learning
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import loggers
import reverb
import sonnet as snt
import tensorflow as tf
import trfl
from acme.agents.tf import dqn


class DQNHER(dqn.DQN):
  """DQN-HER agent.

  """

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      network: snt.Module,
      batch_size: int = 256,
      prefetch_size: int = 4,
      target_update_period: int = 100,
      samples_per_insert: float = 32.0,
      min_replay_size: int = 1000,
      max_replay_size: int = 1000000,
      importance_sampling_exponent: float = 0.2,
      priority_exponent: float = 0.6,
      n_step: int = 5,
      epsilon: Optional[tf.Variable] = None,
      learning_rate: float = 1e-3,
      discount: float = 0.99,
      logger: Optional[loggers.Logger] = None,
      checkpoint: bool = True,
      checkpoint_subpath: str = '~/acme',
      policy_network: Optional[snt.Module] = None,
      max_gradient_norm: Optional[float] = None,
  ):
    """Initialize the agent.

    Args:
      environment_spec: description of the actions, observations, etc.
      network: the online Q network (the one being optimized)
      batch_size: batch size for updates.
      prefetch_size: size to prefetch from replay.
      target_update_period: number of learner steps to perform before updating
        the target networks.
      samples_per_insert: number of samples to take from replay for every insert
        that is made.
      min_replay_size: minimum replay size before updating. This and all
        following arguments are related to dataset construction and will be
        ignored if a dataset argument is passed.
      max_replay_size: maximum replay size.
      importance_sampling_exponent: power to which importance weights are raised
        before normalizing.
      priority_exponent: exponent used in prioritized sampling.
      n_step: number of steps to squash into a single transition.
      epsilon: probability of taking a random action; ignored if a policy
        network is given.
      learning_rate: learning rate for the q-network update.
      discount: discount to use for TD updates.
      logger: logger object to be used by learner.
      checkpoint: boolean indicating whether to checkpoint the learner.
      checkpoint_subpath: string indicating where the agent should save
        checkpoints and snapshots.
      policy_network: if given, this will be used as the policy network.
        Otherwise, an epsilon greedy policy using the online Q network will be
        created. Policy network is used in the actor to sample actions.
      max_gradient_norm: used for gradient clipping.
    """

    # Create a replay server to add data to. This uses no limiter behavior in
    # order to allow the Agent interface to handle it.
    self._alternative_goals = []


    super().__init__(
      environment_spec =environment_spec,
      network=network,
      batch_size=batch_size,
      prefetch_size=prefetch_size,
      target_update_period = target_update_period,
      samples_per_insert=samples_per_insert,
      min_replay_size=min_replay_size,
      max_replay_size=max_replay_size,
      importance_sampling_exponent= importance_sampling_exponent,
      priority_exponent=priority_exponent,
      n_step=n_step,
      epsilon=epsilon,
      learning_rate=learning_rate,
      discount=discount,
      logger=logger,
      checkpoint=checkpoint,
      checkpoint_subpath=checkpoint_subpath,
      policy_network=policy_network,
      max_gradient_norm=max_gradient_norm)

def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
  self._num_observations += 1

  # Store original observation and action
  original_observation = next_timestep.observation
  original_action = action

  # Perform HER
  for alternative_goal in self._alternative_goals:
    # Set the alternative goal as the new desired goal
    next_timestep.observation['desired_goal'] = alternative_goal

    # Store the modified observation and action
    modified_observation = next_timestep.observation
    modified_action = action

    # Pass the modified observation and action to the actor for storage
    self._actor.observe(modified_action, dm_env.TimeStep(
        step_type=next_timestep.step_type,
        reward=next_timestep.reward,
        discount=next_timestep.discount,
        observation=modified_observation,
    ))

  # Restore the original observation and action
  next_timestep.observation = original_observation
  action = original_action

  # Pass the original observation and action to the actor for storage
  self._actor.observe(action, next_timestep)

def update(self):
  # if self._iterator:
  super().update()

  # Perform HER for the stored experiences
  for alternative_goal in self._alternative_goals:
    # Set the alternative goal as the new desired goal
    for replay_table in self._replay_tables:
      replay_table.py_client.mutate_priorities(
          table=table.name,
          updates=[
              reverb.ReplayTable.Diff(
                  info=transition.info,
                  data={
                      'desired_goal': alternative_goal
                  }
              ) for transition in replay_table.py_client.sample(
                  table=table.name,
                  num_samples=self._batch_size_upper_bounds[table_index],
                  remove_from_table=False
              )
          ]
      )
