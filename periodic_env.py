from collections import defaultdict
import dm_env
from dm_env import specs
import numpy as np
from data.periodic_table import PeriodicTable

class PeriodicTableEnv(dm_env.Environment):
  
  def __init__(self, max_episode_len=9, data_filename='data/periodic_table.csv'):

    # Init variables
    self.periodic_table = PeriodicTable(data_filename)

    self.MAXZ = len(self.periodic_table.table)
    self.EUNK = float(0) # Set unknown energies at 0 
    self.OUT = float(-10.0) # If going out of bounds
    self.STEP = float(0.0) # Taking a step
    self.STOP = float(-1.0) # Terminating
    self.GAMMA = float(0.9)

    self.max_episode_len = max_episode_len
    self.action_dim = 5 # _ | → | ← | ↓ | ↑  
    self.curr_state = None
    self.episode_len = 0
    self._reset_next_step = True

    # Fill in states based on periodic table
    self.states = set()
    for z in range(1, self.MAXZ+1):
      
      # load element z from periodic table
      e = self.periodic_table[z]
      if e['E_ads_OH2'] == None: e['E_ads_OH2'] = self.EUNK

      # arrays aren't hashable, convert to tuple
      self.states.add(tuple(np.array(e.one_hot_encode(self.MAXZ), dtype=np.float32))) 

    # normalize rewards
    list_rewards = [int(e['E_ads_OH2'])**2 for z, e in self.periodic_table.table.items()]
    self.norm_reward = sum(list_rewards)/len(list_rewards)

    # Fill in initial states
    self.initial_states = self.states.copy()
    
    #self.initial_states = set()
    #self.initial_states.add(tuple(self.periodic_table[1]['state'].copy()))
    #h = self.periodic_table[1]
    #self.initial_states.add(tuple(np.array(e.one_hot_encode(self.MAXZ), dtype=np.float32)))

        # start on line 1: H or He, then build up
        #if e['n'] == 1: self.states.add(tuple(s))

  def reset(self) -> dm_env.TimeStep:
    self._reset_next_step = False
    rand_start_idx = np.random.choice(len(self.initial_states))+1
    self.curr_state = np.array(self.periodic_table[rand_start_idx]['state'], dtype=np.float32)
    self.reward = self.periodic_table[rand_start_idx]['E_ads_OH2']
    self.episode_len = 0
    return dm_env.TimeStep(dm_env.StepType.FIRST,
            self.reward,
            self.GAMMA,
            self.curr_state)

  def step(self, int_action):
    if self._reset_next_step:
      return self.reset()

    # Initialize
    self._reset_next_step = False
    self.curr_state = np.array(self.curr_state, dtype=np.float32)
    curr_z = self.periodic_table.state_to_z(self.curr_state)
    curr_E = self.periodic_table[curr_z]['E_ads_OH2']

    # Take action # _ | ← | → | ↓ | ↑  
    next_z = None

    if int_action == 0: # _
        next_z = curr_z

    if int_action == 1: # ← 
        next_z = self.periodic_table.previous_element(curr_z)

    if int_action == 2: # →
        next_z = self.periodic_table.next_element(curr_z)

    if int_action == 3: # ↑ 
        next_z = self.periodic_table.element_above(curr_z)

    if int_action == 4: # ↓
        next_z = self.periodic_table.element_below(curr_z)
    
    if self.episode_len == self.max_episode_len: # STOP
      self._reset_next_step = True
      print('Episode length: ', 
            self.episode_len, 
            ' | Final state: ', self.periodic_table[curr_z]['symbol'], 
            ' | Energy: ', self.periodic_table[curr_z]['E_ads_OH2'])
      return dm_env.TimeStep(dm_env.StepType.LAST, self.STOP-self.norm_reward, self.GAMMA, self.curr_state)

    # If action is invalid, give a negative reward but stay in the same state
    if next_z == None:
      print(f"Out of bounds from {self.periodic_table[curr_z]['symbol']}")
      return dm_env.TimeStep(dm_env.StepType.MID, self.OUT-self.norm_reward, self.GAMMA, self.curr_state)
      
    # Move to next state
    self.next_state = np.array(self.periodic_table[next_z].one_hot_encode(self.MAXZ), dtype=np.float32)
  
    #print(f"curr_z: {curr_z} | int_action: {int_action} | next_z {next_z}")

    # If we don't have data there then compute energy
    next_E = self.periodic_table[next_z]['E_ads_OH2']
    reward =  (next_E)**2 - self.norm_reward # (curr_E)**2
    
    # We're arriving at a valid state.
    self.episode_len += 1
    self.curr_state = self.next_state.copy()
    return dm_env.TimeStep(dm_env.StepType.MID, reward, self.GAMMA, self.curr_state)

  def action_spec(self):
    # Represent the action as a 5-dim one-hot vector:
    # _ | ← | → | ↓ | ↑  
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