from absl import app
from absl import flags

import acme
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

    # Create a replay buffer to store transitions
    replay_buffer = acme.datasets.reverb.make_reverb_prioritized_nstep_replay_buffer(
        environment_spec=environment_spec,
        n_step=1,
        batch_size=64,
        max_replay_size=1000000,
    )

    agent = dqn.DQN(
        environment_spec=environment_spec,
        network=network,
        target_update_period=50,
        samples_per_insert=8.,
        n_step=1,
        replay_buffer=replay_buffer,
        checkpoint=False,
        epsilon=0.1,
        learning_rate=1e-4,
    )

    loop = acme.EnvironmentLoop(environment, agent)

    for _ in range(num_episodes):
        # Collect transitions
        loop.run_episode()

        # Apply HER and add augmented transitions to the replay buffer
        transitions = loop._last_episode.transitions
        augmented_transitions = apply_her(transitions)
        for transition in augmented_transitions:
            replay_buffer.add(transition)

        # Update the network using samples from the replay buffer
        agent.learn()

    # ...


def apply_her(transitions):
    augmented_transitions = []
    for transition in transitions:
        achieved_goal = transition.next_state  # Assuming the next_state represents the achieved goal
        desired_goal = transition.next_state_desired_goal  # Assuming you have access to the desired goal

        # Create a new transition with the achieved goal
        augmented_transition = acme.types.Transition(
            observation=transition.observation,
            action=transition.action,
            reward=transition.reward,
            next_observation=transition.next_observation,
            next_action=transition.next_action,
            discount=transition.discount,
            next_state=achieved_goal,
            next_state_desired_goal=desired_goal
        )

        augmented_transitions.append(transition)
        augmented_transitions.append(augmented_transition)

    return augmented_transitions


if __name__ == '__main__':
    app.run(main)
