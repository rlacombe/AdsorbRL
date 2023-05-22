from absl import app
from absl import flags

import acme
from acme import specs
from acme import wrappers
from acme.agents.tf import dqn
import sonnet as snt

from env import CatEnv


def main(_):
  environment = wrappers.SinglePrecisionWrapper(CatEnv())
  environment_spec = specs.make_environment_spec(environment)

  network = snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([50, 50, environment_spec.actions.num_values])
  ])

  agent = dqn.DQN(
    environment_spec=environment_spec,
    network=network,
    target_update_period=20,
    n_step=1,
    checkpoint=False
  )

  loop = acme.EnvironmentLoop(environment, agent)
  loop.run(num_episodes=10000)  # pytype: disable=attribute-error


if __name__ == '__main__':
  app.run(main)
