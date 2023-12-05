
import gymnasium as gym
from gymnasium.core import Wrapper

import numpy as np

class CurriculumWrapper(Wrapper):
    """Superclass of wrappers that can modify observations using :meth:`observation` for :meth:`reset` and :meth:`step`.

    If you would like to apply a function to the observation that is returned by the base environment before
    passing it to learning code, you can simply inherit from :class:`ObservationWrapper` and overwrite the method
    :meth:`observation` to implement that transformation. The transformation defined in that method must be
    defined on the base environment’s observation space. However, it may take values in a different space.
    In that case, you need to specify the new observation space of the wrapper by setting :attr:`self.observation_space`
    in the :meth:`__init__` method of your wrapper.

    For example, you might have a 2D navigation task where the environment returns dictionaries as observations with
    keys ``"agent_position"`` and ``"target_position"``. A common thing to do might be to throw away some degrees of
    freedom and only consider the position of the target relative to the agent, i.e.
    ``observation["target_position"] - observation["agent_position"]``. For this, you could implement an
    observation wrapper like this::

        class RelativePosition(gym.ObservationWrapper):
            def __init__(self, env):
                super().__init__(env)
                self.observation_space = Box(shape=(2,), low=-np.inf, high=np.inf)

            def observation(self, obs):
                return obs["target"] - obs["agent"]

    Among others, Gym provides the observation wrapper :class:`TimeAwareObservation`, which adds information about the
    index of the timestep to the observation.
    """
    def __init__(self, env: gym.Env):
        Wrapper.__init__(self, env)

    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        obs, info = self.env.reset(**kwargs)
        self.current_observation = self.observation(obs)
        return self.current_observation, info

    def step(self, action):
        """Returns a modified observation using :meth:`self.observation` after calling :meth:`env.step`."""
        next_observation, reward, terminated, truncated, info = self.env.step(action)

        reward = self.reward(self.current_observation, action, next_observation)

        self.current_observation = self.observation(next_observation)
        return self.current_observation, self.reward(reward), terminated, truncated, info

    def observation(self, observation):
        """Returns a modified observation."""
        robot_state = observation['observation']
        goal_pos = observation['desired_goal']

        new_observations = np.concatenate([robot_state, goal_pos])
        raise new_observations
    
    def reward(self, current_observation, action, next_observation):
        """Returns a modified reward."""
        if self.task == "move_to_block":
            return self.move_to_block(current_observation, action, next_observation)
        elif self.task == "push_block":
            return self.push_block(current_observation, action, next_observation)
        elif self.task == "main_task":
            return self.main_task(current_observation, action, next_observation)

    def move_to_block(self, current_observation, action, next_observation):
        """Returns a modified reward."""
        end_effector_pos = current_observation[:3]
        block_pos = current_observation[3:6]
        distance = np.linalg.norm(end_effector_pos - block_pos)
        return -distance
    
    def push_block(self, current_observation, action, next_observation):
        """Returns a modified reward."""
        block_pos = current_observation[3:6]
        next_block_pos = next_observation[3:6]
        goal_pos = current_observation[25:28]
        init_distance = np.linalg.norm(block_pos - goal_pos)
        next_distance = np.linalg.norm(next_block_pos - goal_pos)
        return init_distance - next_distance
    
    def main_task(self, current_observation, action, next_observation):
        """Returns a modified reward."""
        reward_1 = self.move_to_block(current_observation, action, next_observation)
        reward_2 = self.push_block(current_observation, action, next_observation)
        return reward_1 + reward_2