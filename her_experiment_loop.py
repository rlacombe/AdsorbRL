# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple agent-environment training loop."""

import operator
import time
from typing import List, Optional, Sequence

from acme import core
from acme.utils import counting
from acme.utils import loggers
from acme.utils import observers as observers_lib
from acme.utils import signals
from acme import core
from acme import specs
from acme import types

import operator
import time
from typing import Optional, Sequence

from acme import core
from acme.utils import counting
from acme.utils import loggers
from acme.utils import observers as observers_lib
from acme.utils import signals
import replay_buffer as experience_buffer

import dm_env
from dm_env import specs
import numpy as np
import tree
import dm_env
from dm_env import specs
import numpy as np
import tree
import util
from util import HERType

class EnvironmentLoopHer(core.Worker):
  """A simple RL environment loop.

  This takes `Environment` and `Actor` instances and coordinates their
  interaction. Agent is updated if `should_update=True`. This can be used as:

    loop = EnvironmentLoop(environment, actor)
    loop.run(num_episodes)

  A `Counter` instance can optionally be given in order to maintain counts
  between different Acme components. If not given a local Counter will be
  created to maintain counts between calls to the `run` method.

  A `Logger` instance can also be passed in order to control the output of the
  loop. If not given a platform-specific default logger will be used as defined
  by utils.loggers.make_default_logger. A string `label` can be passed to easily
  change the label associated with the default logger; this is ignored if a
  `Logger` instance is given.

  A list of 'Observer' instances can be specified to generate additional metrics
  to be logged by the logger. They have access to the 'Environment' instance,
  the current timestep datastruct and the current action.
  """

  def __init__(
      self,
      environment: dm_env.Environment,
      actor: core.Actor,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      should_update: bool = True,
      label: str = 'environment_loop',
      observers: Sequence[observers_lib.EnvLoopObserver] = (),
      buffer_size =1e6,
      batch_size = 128,
      num_relabeled=4,
      opt_steps=4
  ):
    # Internalize agent and environment.
    self._environment = environment
    self._actor = actor
    self._counter =  counting.Counter()
    dir(counting.Counter)

    self._logger = logger or loggers.make_default_logger(
        label, steps_key=self._counter.get_steps_key())
    self._should_update = should_update
    self._observers = observers
    self._replay_buffer = experience_buffer.Buffer(buffer_size, batch_size)
    self._num_relabeled = num_relabeled
    self._opt_steps=opt_steps

  def update_replay_buffer(
    self,
    replay_buffer,
    episode_experience,
    her_type=HERType.NO_HINDSIGHT,
    env_reward_function=None, # pylint: disable=unused-argument
    num_relabeled=4, # pylint: disable=unused-argument
):
    """Adds past experience to the replay buffer. Training is done with
    episodes from the replay buffer. When HER is used, relabeled
    experiences are also added to the replay buffer.

    Args:
        replay_buffer (ReplayBuffer): replay buffer to store experience
        episode_experience (list): list containing the transitions
            (state, action, reward, next_state, goal_state)
        HER (HERType): type of hindsight experience replay to use
        env_reward_function ((ndarray, ndarray) -> float):
            reward function for relabelling transitions
        num_relabeled (int): number of relabeled transition per transition
    """

    for timestep in range(len(episode_experience)):

        # copy experience from episode_experience to replay_buffer
        state, action, reward, next_state, goal = episode_experience[timestep]
        # use replay_buffer.add
        replay_buffer.add(np.append(state, goal),
                          action,
                          reward,
                          np.append(next_state, goal))

        # ======================== TODO modify code ========================

        if her_type == HERType.FINAL:
            # relabel episode based on final state in episode
            final_goal = episode_experience[-1][3]
            new_reward = env_reward_function(next_state, final_goal)
            replay_buffer.add(np.append(state, final_goal),
                              action,
                              new_reward,
                              np.append(next_state, final_goal))
            # get final goal

            # compute new reward

            # add to buffer

        elif her_type == HERType.FUTURE:
            # future: relabel episode based on randomly sampled future state.
            # At each timestep t, relabel the goal with a randomly selected
            # timestep between t and the end of the episode
            for _ in range(num_relabeled):
                future_index = np.random.randint(timestep, len(episode_experience))
                future_goal = episode_experience[future_index][3]
                new_reward = env_reward_function(next_state, future_goal)
                replay_buffer.add(np.append(state, future_goal),
                                  action,
                                  new_reward,
                                  np.append(next_state, future_goal))
            # for every transition, add num_relabeled transitions to the buffer

            # get random future goal

            # compute new reward

            # add to replay buffer

        elif her_type == HERType.RANDOM:
            # random: relabel episode based on a random state from the episode
            for _ in range(num_relabeled):
                random_index = np.random.randint(len(episode_experience))
                random_goal = episode_experience[random_index][3]
                new_reward = env_reward_function(next_state, random_goal)
                replay_buffer.add(np.append(state, random_goal),
                                  action,
                                  new_reward,
                                  np.append(next_state, random_goal))
            # for every transition, add num_relabeled transitions to the buffer

            # get random goal

            # compute new reward

            # add to replay buffer

        # ========================      END TODO       ========================

  def run_episode(self) -> loggers.LoggingData:
    """Run one episode.

    Each episode is a loop which interacts first with the environment to get an
    observation and then give that observation to the agent in order to retrieve
    an action.

    Returns:
      An instance of `loggers.LoggingData`.
    """
    # Reset any counts and start the environment.
    episode_start_time = time.time()
    select_action_durations: List[float] = []
    env_step_durations: List[float] = []
    episode_steps: int = 0

    # For evaluation, this keeps track of the total undiscounted reward
    # accumulated during the episode.
    episode_return = tree.map_structure(self._generate_zeros_from_spec,
                                        self._environment.reward_spec())
    env_reset_start = time.time()
    timestep = self._environment.reset()
    env_reset_duration = time.time() - env_reset_start
    # Make the first observation.
    self._actor.observe_first(timestep)
    for observer in self._observers:
      # Initialize the observer with the current state of the env after reset
      # and the initial timestep.
      observer.observe_first(self._environment, timestep)
    #   episode_experience (list): list containing the transitions
    # (state, action, reward, next_state, goal_state)
    episode_experience = []
    # Run an episode.
    while not timestep.last():
      # Book-keeping.
      episode_steps += 1

      # Generate an action from the agent's policy.
      select_action_start = time.time()
      action = self._actor.select_action(timestep.observation)
      select_action_durations.append(time.time() - select_action_start)
      old_state = timestep.observation
      # Step the environment with the agent's selected action.
      env_step_start = time.time()
      timestep = self._environment.step(action)
      env_step_durations.append(time.time() - env_step_start)

      # Have the agent and observers observe the timestep.
      self._actor.observe(action, next_timestep=timestep)
      for observer in self._observers:
        # One environment step was completed. Observe the current state of the
        # environment, the current timestep and the action.
        observer.observe(self._environment, timestep, action)

      # Give the actor the opportunity to update itself.
      if self._should_update:
        self._actor.update()

      # Equivalent to: episode_return += timestep.reward
      # We capture the return value because if timestep.reward is a JAX
      # DeviceArray, episode_return will not be mutated in-place. (In all other
      # cases, the returned episode_return will be the same object as the
      # argument episode_return.)
      episode_return = tree.map_structure(operator.iadd,
                                          episode_return,
                                          timestep.reward)
      transition = (old_state,action,episode_return,timestep.observation,self._environment.goal)
      np.append(episode_experience,transition)

    self.update_replay_buffer(
        self._replay_buffer,
        episode_experience,
        her_type=HERType.FINAL,
        env_reward_function= self._environment.rewardFunction,
        num_relabeled=self._num_relabeled
    )
    # Record counts.
    counts = self._counter.increment(episodes=1, steps=episode_steps)
    for _ in range(self._opt_steps):
      self._actor.observe(action, next_timestep=timestep)
      for observer in self._observers:
        # One environment step was completed. Observe the current state of the
        # environment, the current timestep and the action.
        observer.observe(self._environment, timestep, action)

      # Give the actor the opportunity to update itself.
      if self._should_update:
        self._actor.update()
        episode_return = tree.map_structure(operator.iadd,
                                      episode_return,
                                      timestep.reward)
        transition = (old_state.observation,action,episode_return,timestep.observation,self._environment.goal)
    # Collect the results and combine with counts.
    steps_per_second = episode_steps / (time.time() - episode_start_time)
    result = {
        'episode_length': episode_steps,
        'episode_return': episode_return,
        'steps_per_second': steps_per_second,
        'env_reset_duration_sec': env_reset_duration,
        'select_action_duration_sec': np.mean(select_action_durations),
        'env_step_duration_sec': np.mean(env_step_durations),
    }
    result.update(counts)
    for observer in self._observers:
      result.update(observer.get_metrics())
    state, action, reward, next_state = replay_buffer.sample()

    return result

  def run(
      self,
      num_episodes: Optional[int] = None,
      num_steps: Optional[int] = None,
  ) -> int:
    """Perform the run loop.

    Run the environment loop either for `num_episodes` episodes or for at
    least `num_steps` steps (the last episode is always run until completion,
    so the total number of steps may be slightly more than `num_steps`).
    At least one of these two arguments has to be None.

    Upon termination of an episode a new episode will be started. If the number
    of episodes and the number of steps are not given then this will interact
    with the environment infinitely.

    Args:
      num_episodes: number of episodes to run the loop for.
      num_steps: minimal number of steps to run the loop for.

    Returns:
      Actual number of steps the loop executed.

    Raises:
      ValueError: If both 'num_episodes' and 'num_steps' are not None.
    """

    if not (num_episodes is None or num_steps is None):
      raise ValueError('Either "num_episodes" or "num_steps" should be None.')

    def should_terminate(episode_count: int, step_count: int) -> bool:
      return ((num_episodes is not None and episode_count >= num_episodes) or
              (num_steps is not None and step_count >= num_steps))

    episode_count: int = 0
    step_count: int = 0
    with signals.runtime_terminator():
      while not should_terminate(episode_count, step_count):
        episode_start = time.time()
        result = self.run_episode()
        result = {**result, **{'episode_duration': time.time() - episode_start}}
        episode_count += 1
        step_count += int(result['episode_length'])
        # Log the given episode results.
        self._logger.write(result)

    return step_count


  def _generate_zeros_from_spec(self, spec: specs.Array) -> np.ndarray:
    return np.zeros(spec.shape, spec.dtype)
