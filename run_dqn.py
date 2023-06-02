from absl import app
import acme
from acme import core
from acme import specs
from acme import wrappers
from acme.agents.tf import dqn
import numpy as np
import sonnet as snt
import tensorflow as tf
from acme.types import TimeStep, Transition

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


class EpisodeObserver:
    def __init__(self):
        self.transitions = []
        self.first_observation = None
        self.first_timestep = None

    def observe_first(self, environment, timestep):
        self.first_observation = timestep.observation
        self.first_timestep = timestep

    def observe(self, environment, timestep, action):
        if timestep.first():
            self.transitions.clear()
            self.first_observation = timestep.observation
            self.first_timestep = timestep

        next_observation = timestep.observation
        transition = Transition(
            observation=self.first_observation,
            action=action,
            reward=timestep.reward,
            discount=timestep.discount,
            next_observation=next_observation
        )
        self.transitions.append(transition)

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
        checkpoint=False,
        epsilon=0.1,
        learning_rate=1e-4,
    )

    observer = EpisodeObserver()
    loop = acme.EnvironmentLoop(
        environment=environment,
        actor=agent,
        observers=[observer]
    )

    num_episodes = 20000  # Define the number of episodes
    for _ in range(num_episodes):
        # Collect transitions
        loop.run_episode()

        # Access episode transitions
        transitions = observer.transitions

        # Apply HER and add augmented transitions to the replay buffer
        augmented_transitions = apply_her(transitions)
        for transition in augmented_transitions:
            replay_buffer.add(transition)

        # Sample transitions from the replay buffer
        batch_size = 64
        samples = replay_buffer.sample(batch_size)

        # Update the network using samples
        agent.update(samples)


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
        achieved_goal = transition.next_observation
        desired_goal = transition.observation

        augmented_transition = Transition(
            observation=transition.observation,
            action=transition.action,
            reward=transition.reward,
            discount=transition.discount,
            next_observation=transition.next_observation,
            extras=transition.extras
        )

        augmented_transitions.append(transition)
        augmented_transitions.append(augmented_transition)

    return augmented_transitions

if __name__ == '__main__':
    app.run(main)
