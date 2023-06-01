class GoalConditionedWrapper:
    def __init__(self, environment):
        self.environment = environment

    def reset(self):
        return self.environment.reset()

    def step(self, action):
        obs = self.environment.step(action)
        goal_obs = copy.deepcopy(obs)  # Set the goal observation as a copy of the current observation
        return (obs, goal_obs)

    def observation_spec(self):
        return self.environment.observation_spec()

    def action_spec(self):
        return self.environment.action_spec()

    def close(self):
        self.environment.close()
