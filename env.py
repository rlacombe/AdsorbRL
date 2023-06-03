from collections import defaultdict
import pickle

import dm_env
from dm_env import specs
import numpy as np


def _read_transitions(filename):
  pickle_file = open(filename, 'rb')
  transitions = pickle.load(pickle_file)
  pickle_file.close()
  return transitions


class CatEnv(dm_env.Environment):

  def __init__(self, max_episode_len=10, data_filename='SARS.pkl'):
    transitions_raw = _read_transitions(data_filename)
    print()
    self.states = {}  # key=55-dim vector, value=energy
    self.initial_states = set()
    self.goal = None
    self.max = None
    for _, _, reward, state in transitions_raw:
      # Numpy arrays aren't hashable, so convert to tuples.
      # Maximize rewards, but want min energy --> flip sign.
      if self.max==None or self.max>=float(-reward):
          self.max=float(-reward)
          self.goal = tuple(np.array(state, dtype=np.float32))
      self.states[tuple(np.array(state, dtype=np.float32))] = float(-reward)

      if np.sum(state) == 1:
        self.initial_states.add(tuple(state))

    self.initial_states = [np.array(t, dtype=np.float32) for t in self.initial_states]
    self.dim = len(self.initial_states[0])
    self.max_episode_len = max_episode_len

    self.curr_state = None
    self.episode_len = 0
    self._reset_next_step = True

  def rewardFunctionPrediction(self,state) -> dm_env.TimeStep:
    return self.states[tuple(state)]

  def reset(self) -> dm_env.TimeStep:
    self._reset_next_step = False
    rand_start_idx = np.random.choice(len(self.initial_states))
    self.curr_state = np.copy(self.initial_states[rand_start_idx])
    self.episode_len = 0
    return dm_env.TimeStep(dm_env.StepType.FIRST,
                           self.states[tuple(self.curr_state)],
                           1.0,
                           self.curr_state)

  def step(self, int_action):
    if self._reset_next_step:
      return self.reset()

    action = np.zeros(self.dim, dtype=np.float32)
    if int_action < self.dim:
      action[int_action] = 1.0
    elif int_action < self.dim + 3:
      # Delete either the 0th, 1st or 2nd element in curr_state.  If deletion idx is
      # greater than the total number, deletes the last one.
      to_delete = int(min(np.sum(self.curr_state) - 1, int_action - self.dim))
      idx_to_remove = np.where(self.curr_state != 0)[0][to_delete]
      action[idx_to_remove] = -1.0
    else:
      # This is our "end" action; it's just a 0 vector no-op.
      pass

    # Let the action 0 mean that the agent asks for the episode to end.
    if np.sum(action) == 0 or self.episode_len == self.max_episode_len:
      self._reset_next_step = True
      # We must be at an existing state; fetch reward.
      cumulative_reward = self.states[tuple(self.curr_state)]
      print('Final state: ', self.episode_len, np.where(self.curr_state)[0], cumulative_reward)
      return dm_env.TimeStep(dm_env.StepType.LAST, 0.0, 1.0, self.curr_state)

    # Take action
    prev_reward = self.states[tuple(self.curr_state)]
    self.curr_state += action

    # Check if action is valid.  If not, end the episode and give a negative reward.
    invalid = False
    if np.sum(self.curr_state) > 3 or np.sum(self.curr_state) <= 0:
      invalid = True
    elif np.sum(action) < 0 and np.any((self.curr_state) < 0):
      invalid = True
    elif np.any((self.curr_state) == 2):
      invalid = True

    if invalid:
      self._reset_next_step = True
      return dm_env.TimeStep(dm_env.StepType.LAST, -1.0, 1.0, self.curr_state)

    # Check if the action is valid but we don't have data there.

    if not tuple(self.curr_state) in self.states:
      self._reset_next_step = True
      return dm_env.TimeStep(dm_env.StepType.LAST, -1.0, 1.0, self.curr_state)
    else:
      # We're arriving at a valid state.
      self.episode_len += 1
      reward = self.states[tuple(self.curr_state)] - prev_reward
      return dm_env.TimeStep(dm_env.StepType.MID, reward, 1.0, self.curr_state)

  def action_spec(self):
    # Represent the action as adding one of 55 elements, removing one element (either
    # the first, second, or third), or terminate.  Some of these actions are invalid
    # depending on the state, and we mark those in the step() function.
    return specs.DiscreteArray(self.dim + 3 + 1)

  def observation_spec(self):
    return specs.BoundedArray(
        shape=(self.dim,),
        dtype=np.float32,
        minimum=0,
        maximum=1,
    )
