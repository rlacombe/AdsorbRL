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
    self.transitions = {}
    self.initial_states = set()
    for state, action, reward, next_state in transitions_raw:
      if len(np.where(action != 0)[0]) > 1:
        continue
      # Converted action is an int representing adding/removing elements and also
      # the "choose to stop" action.  So, we have 55 places we can add an element,
      # 55 places we can remove an element, +1 action for terminating.  Some of these
      # actions are invalid depending on the state, and we mark those in the step()
      # function.
      converted_action = np.where(action != 0)[0][0]
      if np.sum(action) < 0:
        converted_action += len(action)

      key = tuple(state), converted_action
      self.transitions[key] = (-reward, next_state)  # Maximize rewards, but want min energy --> flip sign
      if np.sum(state) == 1:
        self.initial_states.add(tuple(state))

    self.initial_states = [np.array(t) for t in self.initial_states]

    self.dim = len(self.initial_states[0])
    self.max_episode_len = max_episode_len

    self.curr_state = None
    self.episode_len = 0
    self._reset_next_step = True

  def reset(self) -> dm_env.TimeStep:
    self._reset_next_step = False
    rand_start_idx = np.random.choice(len(self.initial_states))
    self.curr_state = self.initial_states[rand_start_idx]
    self.episode_len = 0
    return dm_env.TimeStep(dm_env.StepType.FIRST, None, None, self.curr_state)

  def step(self, int_action):
    if self._reset_next_step:
      return self.reset()

    action = np.zeros(self.dim)
    if int_action < self.dim:
      action[int_action] = 1.0
    elif int_action < 2 * self.dim:
      action[int_action - self.dim] = -1.0
    else:
      # This is our "end" action; it's just a 0 vector no-op.
      pass

    # Let the action 0 mean that the agent asks for the episode to end.
    if np.sum(action) == 0 or self.episode_len == self.max_episode_len:
      self._reset_next_step = True
      print('Final state: ', self.episode_len, np.where(self.curr_state)[0])
      return dm_env.TimeStep(dm_env.StepType.LAST, 0., 1.0, self.curr_state)

    # Check if action is valid.  If not, end the episode and give big negative reward.
    invalid = False
    if np.sum(action + self.curr_state) > 3:
      invalid = True
    elif np.sum(action) < 0 and np.any((self.curr_state + action) < 0):
      invalid = True
    elif np.any((self.curr_state + action) == 2):
      invalid = True

    if invalid:
      self._reset_next_step = True
      return dm_env.TimeStep(dm_env.StepType.LAST, -100., 1.0, self.curr_state + action)
      
    # Check if the action is valid but we don't have data there.
    key = tuple(self.curr_state), int_action
    if not key in self.transitions:
      self._reset_next_step = True
      return dm_env.TimeStep(dm_env.StepType.LAST, -10., 1.0, self.curr_state)
    else:
      self.curr_state += action
      self.episode_len += 1
      reward, next_state = self.transitions[key]
      return dm_env.TimeStep(dm_env.StepType.MID, reward, 1.0, next_state)

  def action_spec(self):
    return specs.DiscreteArray(2 * self.dim + 1)

  def observation_spec(self):
    return specs.BoundedArray(
        shape=(self.dim,),
        dtype=np.float32,
        minimum=0,
        maximum=1,
    )

