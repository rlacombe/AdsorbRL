from absl import app
from agent_her import DQNHER

import acme
from acme import core
from acme import specs
from acme import wrappers
from acme.agents.tf import dqn
import numpy as np
import sonnet as snt
import tensorflow as tf

from env import CatEnv

def main(_):
  environment = wrappers.SinglePrecisionWrapper(CatEnv())
  environment_spec = specs.make_environment_spec(environment)

  network = snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([256, environment_spec.actions.num_values])
  ])

  agent = DQNHER(
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
  loop.run(num_episodes=20000)  # pytype: disable=attribute-error


  state = np.zeros((1,55))
  state[0, 6] = 1.0
  q_vals = agent._learner._network(tf.constant(state, dtype=tf.float32))
  print('Starting with C; should see high weight on Si (index 42):')
  print(q_vals)

  state = np.zeros((1,55))
  state[0, 23] = 1.0
  q_vals = agent._learner._network(tf.constant(state, dtype=tf.float32))
  print('Starting with Mn; should see higher weight on Pd and Pt (indices 32 & 33):')
  print(q_vals)


if __name__ == '__main__':
    app.run(main)
