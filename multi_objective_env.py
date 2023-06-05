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
  
  def __init__(self, data_filenames, signs, max_comparison, max_episode_len=20):
    self.max_comparison = max_comparison
    self.unfiltered_states = defaultdict(list)
    self.unfiltered_energies = defaultdict(list)
    for data_filename, sign in zip(data_filenames, signs):
      transitions_raw = _read_transitions(data_filename)
      seen = set()
      for _, _, reward, state in transitions_raw:
        # Numpy arrays aren't hashable, so convert to tuples.
        # Maximize rewards, but want min energy --> flip sign.
        state_tuple = tuple(np.array(state, dtype=np.float32))
        if state_tuple not in seen:
          self.unfiltered_states[state_tuple].append(-sign * reward / 10.0)
          self.unfiltered_energies[state_tuple].append(reward)
          seen.add(state_tuple)

    # Filter for the states that have overlap
    self.states = {}
    self.energies = {}
    self.initial_states = set()
    for state, rewards in self.unfiltered_states.items():
      if len(rewards) < len(data_filenames):
        continue

      if np.sum(state) == 1:
        self.initial_states.add(tuple(state))

      self.states[state] = rewards
      self.energies[state] = self.unfiltered_energies[state]

    self.initial_states = [np.array(t, dtype=np.float32) for t in self.initial_states]
    print(len(self.initial_states), len(self.states))
    self.dim = len(self.initial_states[0])
    self.max_episode_len = max_episode_len

    self.curr_state = None
    self.episode_len = 0
    self._reset_next_step = True

  def reset(self) -> dm_env.TimeStep:
    self._reset_next_step = False
    rand_start_idx = np.random.choice(len(self.initial_states))
    self.curr_state = np.copy(self.initial_states[rand_start_idx])
    self.episode_len = 0
    return dm_env.TimeStep(dm_env.StepType.FIRST,
                           0.0,
                           1.0,
                           self.curr_state)

  def step(self, int_action):
    if self._reset_next_step:
      return self.reset()

    if int_action == 0 or self.episode_len == self.max_episode_len:
      self._reset_next_step = True
      cumulative_reward = self.energies[tuple(self.curr_state)]
      print('Final state: ', self.episode_len, np.where(self.curr_state)[0], cumulative_reward)
      rewards = self.states[tuple(self.curr_state)]
      return dm_env.TimeStep(dm_env.StepType.LAST,
                             np.sum(np.random.choice(rewards, self.max_comparison, replace=False)),
                             1.0,
                             self.curr_state)

    # Take action
    if int_action == 1:
      # Add a random valid element
      next_states = []
      for state in self.states:
        delta = np.array(state) - self.curr_state
        if np.sum(delta) == 1 and np.all(delta >= 0):
          next_states.append(state)

      if len(next_states) > 0:
        chosen = np.random.choice(len(next_states))
        next_state = np.array(next_states[chosen], dtype=np.float32)
      else:
        next_state = self.curr_state

    else:
      # Remove. int_action=2 -> remove first 1, int_action=3 -> remove second 1, etc
      # If invalid removal, do nothing.
      elems = np.where(self.curr_state)[0]
      valid = True
      if len(elems) == 1 or int_action - 2 >= len(elems):
        valid = False

      if valid:
        delta = np.zeros_like(self.curr_state)
        delta[elems[int_action - 2]] = 1.0
        if tuple(self.curr_state - delta) not in self.states:
          valid = False

      if valid:
        next_state = self.curr_state - delta
      else:
        next_state = self.curr_state
  
    self.episode_len += 1
    self.curr_state = np.array(next_state, dtype=np.float32)
    return dm_env.TimeStep(dm_env.StepType.MID, 0.0, 1.0, self.curr_state)

  def action_spec(self):
    # Represent the action as adding a random element, remove a random element, or stop.
    return specs.DiscreteArray(5)

  def observation_spec(self):
    return specs.BoundedArray(
        shape=(self.dim,),
        dtype=np.float32,
        minimum=0,
        maximum=1,
    )
