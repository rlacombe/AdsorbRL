from absl import app
from absl import flags

import acme
from acme import specs
from acme import wrappers
from acme.agents.tf import dqn
from acme.utils.loggers import tf_summary
import numpy as np
import sonnet as snt
import tensorflow as tf
import datetime

from periodic_env import PeriodicTableEnv
from explore import LinearExplorationSchedule, DQNExplorer, EpsilonGreedyEnvironmentLoop, QLearningAgent

def perform_rollouts(environment, agent, num_rollouts):
    total_E = 0.0

    for _ in range(num_rollouts):
        timestep = environment.reset()

        while not timestep.last():
            action = agent.select_action(timestep.observation)
            timestep = environment.step(action)
        
        state = timestep.observation
        z = environment.periodic_table.state_to_z(state)
        total_E += - environment.periodic_table[z]['E_ads_OH2']

    average_energy = total_E / num_rollouts
    return average_energy


def main(_):
  
  # Define environment
  environment = wrappers.SinglePrecisionWrapper(PeriodicTableEnv(max_episode_len=11))
  environment_spec = specs.make_environment_spec(environment)

  # Define agent
  agent = QLearningAgent(
    env_specs=environment_spec, step_size=0.02, epsilon=0.2
  )

  # Logging
  current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  log_directory = f"logs/QL-{current_time}"
  logger = tf_summary.TFSummaryLogger(logdir = log_directory, label = 'Q-Learning')
  
  # Define main loop
  loop = acme.EnvironmentLoop(environment, agent, logger=logger)
  total_episodes = 20000
  eval_every = 1000

  # Run main lop
  for steps in range(int(total_episodes / eval_every)+1):
    if not steps == 0: loop.run(num_episodes=eval_every)  # Train in environment
    
    # Roll out policies and evaluate last state average energy
    avg_energy = perform_rollouts(environment, agent, 100)
    print(f"\n\n++++++++++++++++++++++++++++++\nAverage final energy: {avg_energy}\n++++++++++++++++++++++++++++++\n\n")

    # Log to TensorBoard
    logger.write({'Average Final Energy': avg_energy})

  
  print(agent.Q)
  print(f"\nQ values for B: {agent.Q[4]}")
  print(f"\nQ values for C: {agent.Q[5]}")
  print(f"\nQ values for N: {agent.Q[6]}")
  print(f"\nQ values for Si: {agent.Q[13]}")
  

if __name__ == '__main__':
  app.run(main)


