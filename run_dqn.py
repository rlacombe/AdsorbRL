from absl import app
from absl import flags

import acme
from acme import specs
from acme import wrappers
from acme.agents.tf import dqn
import numpy as np
import sonnet as snt
import tensorflow as tf

from periodic_env import PeriodicTableEnv


def main(_):
  environment = wrappers.SinglePrecisionWrapper(PeriodicTableEnv())
  environment_spec = specs.make_environment_spec(environment)

  network = snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([256, environment_spec.actions.num_values])
  ])

  agent = dqn.DQN(
    environment_spec=environment_spec,
    network=network,
    target_update_period=50,
    samples_per_insert=8.,
    n_step=1,
    checkpoint=False,
    epsilon=0.1,
    learning_rate=1e-4,
  )

  loop = acme.EnvironmentLoop(environment, agent)
  loop.run(num_episodes=50000)  # pytype: disable=attribute-error


  state = np.zeros((1,86))
  state[0, 6] = 1.0
  q_vals = agent._learner._network(tf.constant(state, dtype=tf.float32))
  print('\nFrom N; should see high weight on going left to C (index 0 ←):')
  print(q_vals)

  state = np.zeros((1,86))
  state[0, 24] = 1.0
  q_vals = agent._learner._network(tf.constant(state, dtype=tf.float32))
  print('\nFrom Mn; should see high weight on going right to Fe (index 1 →):')
  print(q_vals)

  state = np.zeros((1,86))
  state[0, 13] = 1.0
  q_vals = agent._learner._network(tf.constant(state, dtype=tf.float32))
  print('\nFrom Si; should see high weight on going up to C (index 2 ↑):')
  print(q_vals)
  
  state = np.zeros((1,86))
  state[0, 11] = 1.0
  q_vals = agent._learner._network(tf.constant(state, dtype=tf.float32))
  print('\nFrom Mg: should see high weight on going down to Ca (index 3 ↓):')
  print(q_vals)

  state = np.zeros((1,86))
  state[0, 25] = 1.0
  q_vals = agent._learner._network(tf.constant(state, dtype=tf.float32))
  print('\nFrom Fe: should see high weight on terminating (index 4 STOP):')
  print(q_vals)

  

if __name__ == '__main__':
  app.run(main)
