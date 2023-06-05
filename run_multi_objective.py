from absl import app
from absl import flags

import acme
from acme import specs
from acme import wrappers
from acme.agents.tf import dqn
import numpy as np
import sonnet as snt
import tensorflow as tf

from multi_objective_env import CatEnv


def main(_):
  filenames = [
    'starCH2_SARS.pkl',
    'starCH4_SARS.pkl',
    'starN2_SARS.pkl',
    'starNH3_SARS.pkl',
    'starOH2_SARS.pkl',
    'starOH_SARS.pkl'
  ]
  signs = [-1, -1, -1, 1, 1, 1]  # Select for the last 3, against the first 3
  environment = wrappers.SinglePrecisionWrapper(CatEnv(filenames, signs, 2))
  environment_spec = specs.make_environment_spec(environment)

  network = snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([512, 128, environment_spec.actions.num_values])
  ])

  agent = dqn.DQN(
    environment_spec=environment_spec,
    network=network,
    target_update_period=300,
    checkpoint=False,
    epsilon=0.1,
    learning_rate=1e-3,
    discount=0.9,
  )

  loop = acme.EnvironmentLoop(environment, agent)
  loop.run(num_episodes=20000)  # pytype: disable=attribute-error

  state = np.zeros((1,55))
  state[0, 6] = 1.0
  q_vals = agent._learner._network(tf.constant(state, dtype=tf.float32))
  print(q_vals)

  state = np.zeros((1,55))
  state[0, 23] = 1.0
  q_vals = agent._learner._network(tf.constant(state, dtype=tf.float32))
  print(q_vals)

  avg_eval = np.zeros(len(filenames))
  num_episodes = 50
  for _ in range(num_episodes):
    timestep = environment.reset()

    while not timestep.last():
      state = tf.expand_dims(tf.constant(timestep.observation), 0)
      q_vals = agent._learner._network(state)
      action = tf.squeeze(tf.argmax(q_vals, axis=-1)).numpy()
      timestep = environment.step(int(action))
      state = timestep.observation
    reward = environment.energies[tuple(state)]
    avg_eval += np.array(reward)
  print(avg_eval / num_episodes)
  
if __name__ == '__main__':
  app.run(main)
