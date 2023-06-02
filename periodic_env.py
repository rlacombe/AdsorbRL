from collections import defaultdict
import dm_env
from dm_env import specs
import numpy as np
from data.periodic_table import PeriodicTable

class PeriodicTableEnv(dm_env.Environment):
  
  def __init__(self, max_episode_len=10, data_filename='data/periodic_table.csv'):

    # Init variables
    self.periodic_table = PeriodicTable(data_filename)
    self.MAXZ = len(self.periodic_table)
    self.action_dim = 4 # → | ← | ↓ | ↑ 
    self.EUNK = 0 # Set unknown energies at 0 
    self.max_episode_len = max_episode_len
    self.curr_state = None
    self.episode_len = 0
    self._reset_next_step = True

    # Fill in states based on periodic table
    self.states = {}
    for z in range(1, self.MAXZ):
      
      # load element z from periodic table
      e = self.periodic_table[z]

      # arrays aren't hashable, convert to tuple
      self.states.append(tuple(e.one_hot_encode(self.MAXZ))) 

    # Fill in initial states
    self.initial_states = set()
    for s in self.states:
        e = self.periodic_table[self.state_to_z(s)]

        # start on line 1: H or He, then build up
        if e['n'] == 1: self.states.add(tuple(s))

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

    curr_z = self.state_to_z(self.curr_state)

    action = np.zeros(self.action_dim, dtype=np.float32)
    if int_action < self.action_dim:
      action[int_action] = 1.0
    else:
      # This is our "end" action; it's just a 0 vector no-op.
      pass

    # Let the action 0 mean that the agent asks for the episode to end.
    if np.sum(action) == 0 or self.episode_len == self.max_episode_len:
      self._reset_next_step = True
      # We must be at an existing state; fetch reward.
      print('Final state: ', 
            self.episode_len, self.periodic_table[curr_z]['symbol'], 
            self.periodic_table[curr_z]['E_ads_OH2'])
      return dm_env.TimeStep(dm_env.StepType.LAST, 0.0, 1.0, self.curr_state)

    # Take action # → | ← | ↓ | ↑ 
    invalid = False

    if int_action == 0: # → 
        next_e = self.periodic_table.next_element(curr_z)
        if not next_e == None:
            next_z = next_e['Z']
        else: invalid = True

    if int_action == 1: # ← 
        next_e = self.periodic_table.previous_element(curr_z)
        if not next_e == None:
            next_z = next_e['Z']
        else: invalid = True

    if int_action == 2: # ↓
        next_e = self.periodic_table.element_below(curr_z)
        if not next_e == None:
            next_z = next_e['Z']
        else: invalid = True

    if int_action == 3: # ↑ 
        next_e = self.periodic_table.element_above(curr_z)
        if not next_e == None:
            next_z = next_e['Z']
        else: invalid = True

    # If action is invalid, end the episode and give a negative reward
    if invalid:
      self._reset_next_step = True
      return dm_env.TimeStep(dm_env.StepType.LAST, -1.0, 1.0, self.curr_state)
      
    # Check if the action is valid but we don't have data there then compute energy
    if self.periodic_table[next_z]['E_ads_OH2'] == None:
        next_E = self.EUNK
    else: next_E = self.periodic_table[next_z]['E_ads_OH2']
    curr_E = self.periodic_table[curr_z]['E_ads_OH2']
    reward = curr_E - next_E

    if not tuple(self.periodic_table[next_z].one_hot_encode(self.MAXZ)) in self.states:
      self._reset_next_step = True
      return dm_env.TimeStep(dm_env.StepType.LAST, -1.0, 1.0, self.curr_state)
    else:
      # We're arriving at a valid state.
      self.episode_len += 1
      return dm_env.TimeStep(dm_env.StepType.MID, reward, 1.0, self.curr_state)

  def action_spec(self):
    # Represent the action as a 4-dim one-hot vector:
    # → | ← | ↓ | ↑  
    return specs.DiscreteArray(self.action_dim)

  def observation_spec(self):
    return specs.BoundedArray(
        shape=(self.dim,),
        dtype=np.float32,
        minimum=0,
        maximum=1,
    )

  def state_to_z(state):
    return np.argmax(np.asarray(state[0])) + 1 # find argmax of state
