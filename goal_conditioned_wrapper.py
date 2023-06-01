from acme import wrappers

class GoalConditionedWrapper(wrappers.SinglePrecisionWrapper):
    def __init__(self, environment):
        super().__init__(environment)

    def step(self, action):
        obs = super().step(action)
        goal_obs = obs  # Set the goal observation as a copy of the current observation
        return (obs, goal_obs)

class HERWrapper(wrappers.EnvironmentWrapper):
    """Wrapper for Hindsight Experience Replay (HER)."""

    def __init__(self, environment):
        super(HERWrapper, self).__init__(environment)
        self._wrapped_env = environment

    def reset(self):
        return self._wrapped_env.reset()

    def step(self, action):
        next_observation, reward, done, info = self._wrapped_env.step(action)

        # Perform HER replay
        # Modify the reward and goal based on achieved goal
        info['achieved_goal'] = next_observation['achieved_goal']
        reward = self._compute_reward(next_observation['achieved_goal'], next_observation['desired_goal'], info)

        return next_observation, reward, done, info

    def _compute_reward(self, achieved_goal, desired_goal, info):
        # Compute the HER reward based on achieved goal and desired goal
        # Modify this function according to your specific reward computation logic
        return float(np.array_equal(achieved_goal, desired_goal))
