from absl import app
from absl import flags
from goal_conditioned_wrapper import GoalConditionedWrapper

import acme
from acme import datasets
from acme import specs
from acme import wrappers
from acme.agents.tf import dqn
import numpy as np
import sonnet as snt
import tensorflow as tf

from env import CatEnv


def main(_):
  environment = wrappers.SinglePrecisionWrapper(CatEnv())
  environment = GoalConditionedWrapper(environment)
  environment_spec = specs.make_environment_spec(environment)
  replay_buffer = datasets.HindsightExperienceReplayBuffer()

  network = snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([256, environment_spec.actions.num_values])
  ])

  agent = dqn.DQN(
    environment_spec=environment_spec,
    network=network,
    replay_buffer=replay_buffer,
    target_update_period=50,
    samples_per_insert=8.,
    n_step=1,
    checkpoint=False,
    epsilon=0.1,
    learning_rate=1e-4,
  )

  loop = acme.EnvironmentLoop(environment, agent)
  num_episodes=  20000
  for _ in range(num_episodes):
    replay_buffer.clear()

    loop.reset()

    while not loop.done:
        action = loop.agent.select_action(loop.environment.current_time_step)
        loop.advance(action)

        replay_buffer.add(loop.last_transition)

        her_transitions = replay_buffer.generate_hindsight_transitions(
            transition=loop.last_transition,  # Original transition
            num_additional_goals=4,  # Number of additional goals to create
        )

        for transition in her_transitions:
            replay_buffer.add(transition)

    loop.agent.update()


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
