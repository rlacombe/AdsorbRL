import acme
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

import dm_env
from dm_env import specs
import numpy as np
import tree


class LinearExplorationSchedule:
    def __init__(self, initial_epsilon, final_epsilon, decay_steps):
        self.initial_epsilon = float(initial_epsilon)
        self.final_epsilon = float(final_epsilon)
        self.decay_steps = float(decay_steps)

    def get_epsilon(self, t):
        decay_factor = min(1.0, float(t) / self.decay_steps)
        epsilon = self.initial_epsilon + (self.final_epsilon - self.initial_epsilon) * decay_factor
        return epsilon

class DQNExplorer(core.Actor):
    def __init__(self, agent, exploration_schedule, num_actions):
        self.agent = agent
        self.exploration_schedule = exploration_schedule
        self.timestep = 0.0
        self.num_actions = num_actions

    def select_action(self, observation):
        epsilon = self.exploration_schedule.get_epsilon(self.timestep)
        self.timestep += 1.0
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions, dtype=np.int32)  # Choose a random action
        else:
            return self.agent.select_action(observation)  # Choose the action with the highest Q-value

    def observe_first(self, timestep):
        self.agent.observe_first(timestep)

    def observe(self, action, next_timestep):
        self.agent.observe(action, next_timestep)

    def update(self):
        self.agent.update()
        self.timestep += 1.0




class EpsilonGreedyEnvironmentLoop(core.Worker):
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
      explorer: DQNExplorer,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      should_update: bool = True,
      label: str = 'explore_environment_loop',
      observers: Sequence[observers_lib.EnvLoopObserver] = (),
  ):
    # Internalize agent and environment.
    self._environment = environment
    self._explorer = explorer
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(label)
    self._should_update = should_update
    self._observers = observers

  def run_episode(self) -> loggers.LoggingData:
    """Run one episode.

    Each episode is a loop which interacts first with the environment to get an
    observation and then give that observation to the agent in order to retrieve
    an action.

    Returns:
      An instance of `loggers.LoggingData`.
    """
    # Reset any counts and start the environment.
    start_time = time.time()
    episode_steps = 0

    # For evaluation, this keeps track of the total undiscounted reward
    # accumulated during the episode.
    episode_return = tree.map_structure(_generate_zeros_from_spec,
                                        self._environment.reward_spec())
    timestep = self._environment.reset()
    # Make the first observation.
    self._explorer.observe_first(timestep)
    for observer in self._observers:
      # Initialize the observer with the current state of the env after reset
      # and the initial timestep.
      observer.observe_first(self._environment, timestep)

    # Run an episode.
    while not timestep.last():
      # Generate an action from the agent's policy and step the environment.
      action = self._explorer.select_action(timestep.observation)
      timestep = self._environment.step(action)

      # Have the agent observe the timestep and let the actor update itself.
      self._explorer.observe(action, next_timestep=timestep)
      for observer in self._observers:
        # One environment step was completed. Observe the current state of the
        # environment, the current timestep and the action.
        observer.observe(self._environment, timestep, action)
      if self._should_update:
        self._explorer.update()
        # if not self._logger == None: self._logger.write({'Learner loss': loss})

      # Book-keeping.
      episode_steps += 1

      # Equivalent to: episode_return += timestep.reward
      # We capture the return value because if timestep.reward is a JAX
      # DeviceArray, episode_return will not be mutated in-place. (In all other
      # cases, the returned episode_return will be the same object as the
      # argument episode_return.)
      episode_return = tree.map_structure(operator.iadd,
                                          episode_return,
                                          timestep.reward)

    # Record counts.
    counts = self._counter.increment(episodes=1, steps=episode_steps)

    # Collect the results and combine with counts.
    steps_per_second = episode_steps / (time.time() - start_time)
    result = {
        'episode_length': episode_steps,
        'episode_return': episode_return,
        'steps_per_second': steps_per_second,
    }
    result.update(counts)
    for observer in self._observers:
      result.update(observer.get_metrics())
    return result

  def run(self,
          num_episodes: Optional[int] = None,
          num_steps: Optional[int] = None):
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

    Raises:
      ValueError: If both 'num_episodes' and 'num_steps' are not None.
    """

    if not (num_episodes is None or num_steps is None):
      raise ValueError('Either "num_episodes" or "num_steps" should be None.')

    def should_terminate(episode_count: int, step_count: int) -> bool:
      return ((num_episodes is not None and episode_count >= num_episodes) or
              (num_steps is not None and step_count >= num_steps))

    episode_count, step_count = 0, 0
    with signals.runtime_terminator():
      while not should_terminate(episode_count, step_count):
        result = self.run_episode()
        episode_count += 1
        step_count += result['episode_length']
        # Log the given episode results.
        self._logger.write(result)

# Placeholder for an EnvironmentLoop alias


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)


class QLearningAgent(acme.Actor):
    
    def __init__(self, env_specs=None, step_size=0.1, epsilon=0.1):
        
        # Black Jack dimensions
        self.Q = np.zeros((86,5))
        
        # set step size
        self.step_size = step_size
        
        # set behavior policy
        # self.policy = None
        self.behavior_policy = lambda q_values: self.epsilon_greedy(q_values, epsilon=0.1)
        
        # store timestep, action, next_timestep
        self.timestep = None
        self.action = None
        self.next_timestep = None

    def epsilon_greedy(self, q_values, epsilon=0.1):
      if np.random.rand() < epsilon:
            return np.random.randint(len(q_values), dtype=np.int32)  # Choose a random action
      else:
            return np.argmax(q_values)  # Choose the action with the highest Q-value

    #def state_to_index(self, state):
        #state = *map(int, state),
        #return state
    
    def transform_state(self, state):
        # this is specifally required for the blackjack environment
        state = int(np.argmax(state))
        return state
    
    def select_action(self, observation):
        state = self.transform_state(observation)
        return self.behavior_policy(self.Q[state])

    def observe_first(self, timestep):
        self.timestep = timestep

    def observe(self, action, next_timestep):
        self.action = action
        self.next_timestep = next_timestep  

    def update(self):
        # get variables for convenience
        state = self.timestep.observation
        _, reward, discount, next_state = self.next_timestep
        action = self.action
        
        # turn states into indices
        state = self.transform_state(state)
        next_state = self.transform_state(next_state)
        
        # Q-value update
        td_error = reward + discount * np.max(self.Q[next_state]) - self.Q[state][action]        
        self.Q[state][action] += self.step_size * td_error
        
        # finally, set timestep to next_timestep
        self.timestep = self.next_timestep  