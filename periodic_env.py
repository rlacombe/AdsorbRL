from collections import defaultdict
import dm_env
from dm_env import specs
import numpy as np
from data.periodic_table import PeriodicTable

class PeriodicTableEnv(dm_env.Environment):
  
  def __init__(self, max_episode_len=20, data_filename='data/periodic_table.csv'):

    # Init variables
    self.periodic_table = PeriodicTable(data_filename)
    self.MAXZ = len(self.periodic_table.table)
    self.action_dim = 4 # → | ← | ↓ | ↑ 
    self.EUNK = 0 # Set unknown energies at 0 
    self.max_episode_len = max_episode_len
    self.curr_state = None
    self.episode_len = 0
    self._reset_next_step = True

    # Fill in states based on periodic table
    self.states = set()
    for z in range(1, self.MAXZ):
      
      # load element z from periodic table
      e = self.periodic_table[z]

      # arrays aren't hashable, convert to tuple
      self.states.add(tuple(np.array(e.one_hot_encode(self.MAXZ), dtype=np.float32))) 

    # Fill in initial states
    self.initial_states = self.states.copy()

    for s in self.states:
        z = self.periodic_table.state_to_z(s)
        e = self.periodic_table[z]

        # start on line 1: H or He, then build up
        #if e['n'] == 1: self.states.add(tuple(s))

  def reset(self) -> dm_env.TimeStep:
    self._reset_next_step = False
    rand_start_idx = np.random.choice(len(self.initial_states))+1
    self.curr_state = np.array(self.periodic_table[rand_start_idx]['state'], dtype=np.float32)
    self.reward = self.EUNK if self.periodic_table[rand_start_idx]['E_ads_OH2'] == None else self.periodic_table[rand_start_idx]['E_ads_OH2']
    self.episode_len = 0
    return dm_env.TimeStep(dm_env.StepType.FIRST,
            self.reward,
            1.0,
            self.curr_state)

  def step(self, int_action):
    if self._reset_next_step:
      return self.reset()

    curr_z = self.periodic_table.state_to_z(self.curr_state)
    self.curr_state = np.array(self.curr_state, dtype=np.float32)
    
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
        next_z = self.periodic_table.next_element(curr_z)
        if next_z == None:
          invalid = True

    if int_action == 1: # ← 
        next_z = self.periodic_table.previous_element(curr_z)
        if next_z == None:
          invalid = True

    if int_action == 2: # ↓
        next_z = self.periodic_table.element_below(curr_z)
        if next_z == None:
          invalid = True

    if int_action == 3: # ↑ 
        next_z = self.periodic_table.element_above(curr_z)
        if next_z == None:
          invalid = True

    # If action is invalid, end the episode and give a negative reward
    if invalid:
      self._reset_next_step = True
      return dm_env.TimeStep(dm_env.StepType.LAST, -1.0, float(1.0), self.curr_state)
      
    # Check if the action is valid but we don't have data there then compute energy
    if self.periodic_table[curr_z]['E_ads_OH2'] == None:
        curr_E = self.EUNK
    else: curr_E = self.periodic_table[curr_z]['E_ads_OH2']
    
    if self.periodic_table[next_z]['E_ads_OH2'] == None:
        next_E = self.EUNK
    else: next_E = self.periodic_table[next_z]['E_ads_OH2']
    
    reward = float(curr_E - next_E)

    # Move to next state
    self.next_state = np.array(self.periodic_table[next_z]['state'], dtype=np.float32)

    if not tuple(self.next_state) in self.states:
      self._reset_next_step = True
      return dm_env.TimeStep(dm_env.StepType.LAST, -1.0, float(1.0), self.curr_state)
    else:
      # We're arriving at a valid state.
      self.episode_len += 1
      return dm_env.TimeStep(dm_env.StepType.MID, reward,float(1.0), self.next_state)

  def action_spec(self):
    # Represent the action as a 4-dim one-hot vector:
    # → | ← | ↓ | ↑  
    return specs.DiscreteArray(
        self.action_dim, 
        dtype=np.int32, 
    )

  def observation_spec(self):
    return specs.BoundedArray(
        shape=(self.MAXZ,),
        dtype=np.float32,
        minimum=0,
        maximum=1,
    )