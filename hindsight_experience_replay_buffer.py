class HindsightExperienceReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, transition):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def generate_hindsight_transitions(self, transition, num_additional_goals):
        her_transitions = []

        for _ in range(num_additional_goals):
            goal_index = random.randint(0, len(self.buffer) - 1)
            goal_transition = self.buffer[goal_index]

            her_transition = {
                'observation': transition['observation'],
                'action': transition['action'],
                'reward': transition['reward'],
                'discount': transition['discount'],
                'observation_': goal_transition['observation_'],  # Set the new goal observation
            }

            her_transitions.append(her_transition)

        return her_transitions

    def clear(self):
        self.buffer.clear()
