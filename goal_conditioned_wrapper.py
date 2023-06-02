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
        timestep = self._wrapped_env.step(action)
        print(timestep)
        # Check if the step_type is LAST and access the observation field if necessary
        next_observation = timestep.observation

        # Compute the achieved_goal and desired_goal
        achieved_goal = self._compute_achieved_goal(next_observation)
        desired_goal = next_observation

        # Compute the reward using the achieved_goal and desired_goal
        reward = self._compute_reward(achieved_goal, desired_goal)

        next_timestep = dm_env.TimeStep(
            step_type=timestep.step_type,
            reward=timestep.reward,
            discount=timestep.discount,
            observation=next_observation
        )

        return next_timestep, reward

    def _compute_achieved_goal(self, next_observation):
        if isinstance(next_observation, float):
            # Handle the case when next_observation is a float value
            # Compute the achieved_goal using the next_observation directly
            achieved_goal = next_observation  # Replace None with the actual computation
        else:
            # Handle the case when next_observation is a dictionary
            # Modify this code to extract the relevant information from the next_observation
            achieved_goal = next_observation  # Replace None with the actual computation
        return achieved_goal

    def _compute_reward(self, achieved_goal, desired_goal):
            # Handle the case when next_observation is a float value
            # Compute the reward using the achieved_goal and desired_goal directly
        reward = self._reward_fn(achieved_goal, desired_goal)
