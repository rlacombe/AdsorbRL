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

from env import CatEnv
from explore import LinearExplorationSchedule, DQNExplorer, EpsilonGreedyEnvironmentLoop

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
  environment = wrappers.SinglePrecisionWrapper(CatEnv())
  environment_spec = specs.make_environment_spec(environment)

  # Define Q-network
  q_network = snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([128, 128, environment_spec.actions.num_values])
  ])
  
  # Define agent and epsilon-greedy exploration schedule
  agent = dqn.DQN(
    environment_spec=environment_spec,
    network=q_network,
    target_update_period=2000,
    samples_per_insert=8.,
    n_step=1,
    checkpoint=False,
    epsilon=0.2,
    learning_rate=1e-3,
    discount=0.8,
  )
  
  exploration_schedule = LinearExplorationSchedule(initial_epsilon=1.0, final_epsilon=0.05, decay_steps=50000.0)
  explorer = DQNExplorer(agent, exploration_schedule, environment.action_dim)

  # Logging
  current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  log_directory = f"logs/{current_time}"
  logger = tf_summary.TFSummaryLogger(logdir = log_directory, label = 'DQN')
  
  # Define main loop
  loop = EpsilonGreedyEnvironmentLoop(environment, explorer, logger=logger)
  total_episodes = 10000
  eval_every = 500

  # Run main lop
  for steps in range(int(total_episodes / eval_every)):
    loop.run(num_episodes=eval_every)  # Train in environment
    
    # Roll out policies and evaluate last state average energy
    avg_energy = perform_rollouts(environment, agent, 100)
    print(f"\n\n++++++++++++++++++++++++++++++\nAverage final energy: {avg_energy}\n++++++++++++++++++++++++++++++\n\n")

    # Log to TensorBoard
    logger.write({'Average Final Energy': avg_energy})
    logger.write({'Epsilon': explorer.exploration_schedule.get_epsilon(explorer.timestep)})


  state = np.zeros((1,86))
  state[0, 25] = 1.0
  q_vals = agent._learner._network(tf.constant(state, dtype=tf.float32))
  print('\nFrom Fe: should see high weight on terminating (index 0: _):')
  print(q_vals)

  state = np.zeros((1,86))
  state[0, 6] = 1.0
  q_vals = agent._learner._network(tf.constant(state, dtype=tf.float32))
  print('\nFrom N; should see high weight on going left to C (index 1: ←):')
  print(q_vals)

  state = np.zeros((1,86))
  state[0, 24] = 1.0
  q_vals = agent._learner._network(tf.constant(state, dtype=tf.float32))
  print('\nFrom Mn; should see high weight on going right to Fe (index 2: →):')
  print(q_vals)

  state = np.zeros((1,86))
  state[0, 13] = 1.0
  q_vals = agent._learner._network(tf.constant(state, dtype=tf.float32))
  print('\nFrom Si; should see high weight on going up to C (index 3: ↑):')
  print(q_vals)
  
  state = np.zeros((1,86))
  state[0, 11] = 1.0
  q_vals = agent._learner._network(tf.constant(state, dtype=tf.float32))
  print('\nFrom Mg: should see high weight on going down to Ca (index 4: ↓):')
  print(q_vals)

  q_vals = np.zeros((86,5))
  for i in range(86):
    state = np.zeros((1,86))
    state[0, i] = 1.0
    q_vals[i] = agent._learner._network(tf.constant(state, dtype=tf.float32))
  print(q_vals)


  

if __name__ == '__main__':
  app.run(main)
