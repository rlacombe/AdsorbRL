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


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, transition):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        return [self.buffer[index] for index in indices]


def main(_):
    environment = wrappers.SinglePrecisionWrapper(CatEnv())
    environment_spec = specs.make_environment_spec(environment)

    network = snt.Sequential([
        snt.Flatten(),
        snt.nets.MLP([256, environment_spec.actions.num_values])
    ])

    buffer_size = 1000000
    replay_buffer = ReplayBuffer(buffer_size)

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
