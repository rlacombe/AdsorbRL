from acme import wrappers

class GoalConditionedWrapper(wrappers.SinglePrecisionWrapper):
    def __init__(self, environment):
        super().__init__(environment)

    def step(self, action):
        obs = super().step(action)
        goal_obs = obs  # Set the goal observation as a copy of the current observation
        return (obs, goal_obs)
