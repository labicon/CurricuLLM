"""A maze environment with the Gymnasium Ant agent (https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/ant_v4.py).

The code is inspired by the D4RL repository hosted on GitHub (https://github.com/Farama-Foundation/D4RL), published in the paper
'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine.

Original Author of the code: Justin Fu

The modifications made involve reusing the code in Gymnasium for the Ant environment and in `point_maze/maze_env.py`.
The new code also follows the Gymnasium API and Multi-goal API

This project is covered by the Apache 2.0 License.
"""

import sys
from os import path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.maze.maps import OPEN, U_MAZE
from gymnasium_robotics.envs.maze.maze_v4 import MazeEnv
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames


class AntMazeEnv(MazeEnv, EzPickle):
    """
    ### Description

    This environment was refactored from the [D4RL](https://github.com/Farama-Foundation/D4RL) repository, introduced by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine
    in ["D4RL: Datasets for Deep Data-Driven Reinforcement Learning"](https://arxiv.org/abs/2004.07219).

    The tasks found in the `AntMaze` environments are the same as the ones in the `PointMaze` environments. However, in this case the agent is the Ant quadruped from the main [Gymnaisum](https://gymnasium.farama.org/environments/mujoco/ant/) repository.
    The control frequency of the ant is of `f = 20 Hz`. Each simulation timestep is of `dt=0.01` and the ant robot repeats the same action for 5 simulation steps.

    ### Maze Variations

    #### Maze size

    The map variations for the mazes are the same as for `PointMaze`. The ant environments with fixed goal and reset locations are the following:

    * `AntMaze_UMaze-v4`
    * `AntMaze_BigMaze-v4`
    * `AntMaze_HardestMaze-v4`

    #### Diverse goal mazes

    The environments with fixed reset position for the ant and randomly selected goals, also known as diverse goal, are:

    * `AntMaze_BigMaze_DG-v4`
    * `AntMaze_HardestMaze_DG-v4`

    #### Diverse goal and reset mazes

    Finally, the environments that select the reset and goal locations randomly are:

    * `AntMaze_BigMaze_DGR-v4`
    * `AntMaze_HardestMaze_DGR-v4`

    #### Custom maze

    Also, any of the `AntMaze` environments can be initialized with a custom maze map by setting the `maze_map` argument like follows:

    ```python
    import gymnasium as gym

    example_map = [[1, 1, 1, 1, 1],
           [1, C, 0, C, 1],
           [1, 1, 1, 1, 1]]

    env = gym.make('AntMaze_UMaze-v4', maze_map=example_map)
    ```

    ### Action Space

    The action space is a `Box(-1, 1, (8,), float32)`. An action represents the torques applied at the hinge joints.

    | Num | Action                                                            | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
    | --- | ----------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
    | 0   | Torque applied on the rotor between the torso and front left hip  | -1          | 1           | hip_1 (front_left_leg)           | hinge | torque (N m) |
    | 1   | Torque applied on the rotor between the front left two links      | -1          | 1           | angle_1 (front_left_leg)         | hinge | torque (N m) |
    | 2   | Torque applied on the rotor between the torso and front right hip | -1          | 1           | hip_2 (front_right_leg)          | hinge | torque (N m) |
    | 3   | Torque applied on the rotor between the front right two links     | -1          | 1           | angle_2 (front_right_leg)        | hinge | torque (N m) |
    | 4   | Torque applied on the rotor between the torso and back left hip   | -1          | 1           | hip_3 (back_leg)                 | hinge | torque (N m) |
    | 5   | Torque applied on the rotor between the back left two links       | -1          | 1           | angle_3 (back_leg)               | hinge | torque (N m) |
    | 6   | Torque applied on the rotor between the torso and back right hip  | -1          | 1           | hip_4 (right_back_leg)           | hinge | torque (N m) |
    | 7   | Torque applied on the rotor between the back right two links      | -1          | 1           | angle_4 (right_back_leg)         | hinge | torque (N m) |

    ### Observation Space

    The observation is a `goal-aware observation space`. It consists of a dictionary with information about the robot's position and goal. The dictionary consists of the following 3 keys:

    * `observation`: Observations consist of positional values of different body parts of the ant, followed by the velocities of those individual parts (their derivatives) with all
        the positions ordered before all the velocities.

        By default, observations do not include the x- and y-coordinates of the ant's torso. These values are included in the `achieved_goal` key of the observation.
        However, by default, an observation is a `ndarray` with shape `(111,)` if the external contact forces are included with the `use_contact_forces` arguments. Otherwise, the shape will be `(27, )`
        The elements of the array correspond to the following:

        | Num | Observation                                                  | Min    | Max    | Name (in corresponding XML file)       | Joint | Unit                     |
        |-----|--------------------------------------------------------------|--------|--------|----------------------------------------|-------|--------------------------|
        | 0   | z-coordinate of the torso (centre)                           | -Inf   | Inf    | torso                                  | free  | position (m)             |
        | 1   | x-orientation of the torso (centre)                          | -Inf   | Inf    | torso                                  | free  | angle (rad)              |
        | 2   | y-orientation of the torso (centre)                          | -Inf   | Inf    | torso                                  | free  | angle (rad)              |
        | 3   | z-orientation of the torso (centre)                          | -Inf   | Inf    | torso                                  | free  | angle (rad)              |
        | 4   | w-orientation of the torso (centre)                          | -Inf   | Inf    | torso                                  | free  | angle (rad)              |
        | 5   | angle between torso and first link on front left             | -Inf   | Inf    | hip_1 (front_left_leg)                 | hinge | angle (rad)              |
        | 6   | angle between the two links on the front left                | -Inf   | Inf    | ankle_1 (front_left_leg)               | hinge | angle (rad)              |
        | 7   | angle between torso and first link on front right            | -Inf   | Inf    | hip_2 (front_right_leg)                | hinge | angle (rad)              |
        | 8   | angle between the two links on the front right               | -Inf   | Inf    | ankle_2 (front_right_leg)              | hinge | angle (rad)              |
        | 9   | angle between torso and first link on back left              | -Inf   | Inf    | hip_3 (back_leg)                       | hinge | angle (rad)              |
        | 10  | angle between the two links on the back left                 | -Inf   | Inf    | ankle_3 (back_leg)                     | hinge | angle (rad)              |
        | 11  | angle between torso and first link on back right             | -Inf   | Inf    | hip_4 (right_back_leg)                 | hinge | angle (rad)              |
        | 12  | angle between the two links on the back right                | -Inf   | Inf    | ankle_4 (right_back_leg)               | hinge | angle (rad)              |
        | 13  | x-coordinate velocity of the torso                           | -Inf   | Inf    | torso                                  | free  | velocity (m/s)           |
        | 14  | y-coordinate velocity of the torso                           | -Inf   | Inf    | torso                                  | free  | velocity (m/s)           |
        | 15  | z-coordinate velocity of the torso                           | -Inf   | Inf    | torso                                  | free  | velocity (m/s)           |
        | 16  | x-coordinate angular velocity of the torso                   | -Inf   | Inf    | torso                                  | free  | angular velocity (rad/s) |
        | 17  | y-coordinate angular velocity of the torso                   | -Inf   | Inf    | torso                                  | free  | angular velocity (rad/s) |
        | 18  | z-coordinate angular velocity of the torso                   | -Inf   | Inf    | torso                                  | free  | angular velocity (rad/s) |
        | 19  | angular velocity of angle between torso and front left link  | -Inf   | Inf    | hip_1 (front_left_leg)                 | hinge | angle (rad)              |
        | 20  | angular velocity of the angle between front left links       | -Inf   | Inf    | ankle_1 (front_left_leg)               | hinge | angle (rad)              |
        | 21  | angular velocity of angle between torso and front right link | -Inf   | Inf    | hip_2 (front_right_leg)                | hinge | angle (rad)              |
        | 22  | angular velocity of the angle between front right links      | -Inf   | Inf    | ankle_2 (front_right_leg)              | hinge | angle (rad)              |
        | 23  | angular velocity of angle between torso and back left link   | -Inf   | Inf    | hip_3 (back_leg)                       | hinge | angle (rad)              |
        | 24  | angular velocity of the angle between back left links        | -Inf   | Inf    | ankle_3 (back_leg)                     | hinge | angle (rad)              |
        | 25  | angular velocity of angle between torso and back right link  | -Inf   | Inf    | hip_4 (right_back_leg)                 | hinge | angle (rad)              |
        | 26  |angular velocity of the angle between back right links        | -Inf   | Inf    | ankle_4 (right_back_leg)               | hinge | angle (rad)              |

        The remaining 14*6 = 84 elements of the observation are contact forces (external forces - force x, y, z and torque x, y, z) applied to the center of mass of each of the links. The 14 links are: the ground link,
        the torso link, and 3 links for each leg (1 + 1 + 12) with the 6 external forces. These elements are included only if at the environments initialization the argument `use_contact_forces` is set to `True`.

    * `desired_goal`: this key represents the final goal to be achieved. In this environment it is a 2-dimensional `ndarray`, `(2,)`, that consists of the two cartesian coordinates of the desired final ant torso position `[x,y]`. The elements of the array are the following:

        | Num | Observation             | Min    | Max    | Site Name (in corresponding XML file) |Unit          |
        |-----|------------------------ |--------|--------|---------------------------------------|--------------|
        | 0   | Final goal x coordinate | -Inf   | Inf    | target                                | position (m) |
        | 1   | Final goal y coordinate | -Inf   | Inf    | target                                | position (m) |

    * `achieved_goal`: this key represents the current state of the ant's torso, as if it would have achieved a goal. This is useful for goal orientated learning algorithms such as those that use [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) (HER).
        The value is an `ndarray` with shape `(2,)`. The elements of the array are the following:

        | Num | Observation                                    | Min    | Max    | Site Name (in corresponding XML file) |Unit          |
        |-----|------------------------------------------------|--------|--------|---------------------------------------|--------------|
        | 0   | Current goal ant position in the x coordinate  | -Inf   | Inf    | torso                                 | position (m) |
        | 1   | Current goal ant position in the y coordinate  | -Inf   | Inf    | torso                                 | position (m) |

    ### Rewards

    The reward can be initialized as `sparse` or `dense`:
    - *sparse*: the returned reward can have two values: `0` if the ant hasn't reached its final target position, and `1` if the ant is in the final target position (the ant is considered to have reached the goal if the Euclidean distance between both is lower than 0.5 m).
    - *dense*: the returned reward is the negative Euclidean distance between the achieved goal position and the desired goal.

    To initialize this environment with one of the mentioned reward functions the type of reward must be specified in the id string when the environment is initialized. For `sparse` reward the id is the default of the environment, `AntMaze_UMaze-v4`. However, for `dense`
    reward the id must be modified to `AntMaze_UMazeDense-v4` and initialized as follows:

    ```python
    import gymnasium as gym

    env = gym.make('AntMaze_UMaze-v4')
    ```

    ### Starting State

    The goal and initial placement of the ant in the maze follows the same structure for all environments. A discrete cell `(i,j)` is selected for the goal and agent's initial position as previously menitoned in the **Maze** section.
    Then this cell index is converted to its cell center as an `(x,y)` continuous Cartesian coordinates in the MuJoCo simulation. Finally, a sampled noise from a uniform distribution with range `[-0.25,0.25]m` is added to the
    cell's center x and y coordinates. This allows to create a richer goal distribution.

    The goal and initial position of the agent can also be specified by the user when the episode is reset. This is done by passing the dictionary argument `options` to the gymnasium reset() function. This dictionary expects one or both of
    the following keys:

    * `goal_cell`: `numpy.ndarray, shape=(2,0), type=int` - Specifies the desired `(i,j)` cell location of the goal. A uniform sampled noise will be added to the continuous coordinates of the center of the cell.
    * `reset_cell`: `numpy.ndarray, shape=(2,0), type=int` - Specifies the desired `(i,j)` cell location of the reset initial agent position. A uniform sampled noise will be added to the continuous coordinates of the center of the cell.

    ### Episode End

    * `truncated` - The episode will be `truncated` when the duration reaches a total of `max_episode_steps`.
    * `terminated` - The task can be set to be continuing with the `continuing_task` argument. In this case the episode will never terminate, instead the goal location is randomly selected again. If the task is set not to be continuing the
    episode will be terminated when the Euclidean distance to the goal is less or equal to 0.5.

    ### Arguments

    * `maze_map` - Optional argument to initialize the environment with a custom maze map.
    * `continuing_task` - If set to `True` the episode won't be terminated when reaching the goal, instead a new goal location will be generated (unless `reset_target` argument is `True`). If `False` the environment is terminated when the ant reaches the final goal.
    * `reset_target` - If set to `True` and the argument `continuing_task` is also `True`, when the ant reaches the target goal the location of the goal will be kept the same and no new goal location will be generated. If `False` a new goal will be generated when reached.
    * `use_contact_forces` - If `True` contact forces of the ant are included in the `observation`.

    Note that, the maximum number of timesteps before the episode is `truncated` can be increased or decreased by specifying the `max_episode_steps` argument at initialization. For example,
    to increase the total number of timesteps to 100 make the environment as follows:

    ```python
    import gymnasium as gym

    env = gym.make('AntMaze_UMaze-v4', max_episode_steps=100)
    ```

    ### Version History
    * v4: Update to maze_v4. Refactor compute_terminated in MazeEnv into a pure function compute_terminated and a new function update_goal which resets the goal position. Ant bug fix: Reward is now computed before reset (i.e. sparse reward is not always zero). Maze bug fix: Ant can no longer reset within the goal radius 0.45 due to maze_size_scaling factor missing in MazeEnv. info['success'] key added.
    * v3: refactor version of the D4RL environment, also create dependency on newest [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html) maintained by the MuJoCo team in Deepmind.
    * v2 & v1: legacy versions in the [D4RL](https://github.com/Farama-Foundation/D4RL).
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        maze_map: List[List[Union[str, int]]] = U_MAZE,
        reward_type: str = "sparse",
        continuing_task: bool = True,
        reset_target: bool = False,
        **kwargs,
    ):
        # Get the ant.xml path from the Gymnasium package
        ant_xml_file_path = path.join(
            path.dirname(sys.modules[AntEnv.__module__].__file__), "assets/ant.xml"
        )
        super().__init__(
            agent_xml_path=ant_xml_file_path,
            maze_map=maze_map,
            maze_size_scaling=4,
            maze_height=0.5,
            reward_type=reward_type,
            continuing_task=continuing_task,
            reset_target=reset_target,
            **kwargs,
        )
        # Create the MuJoCo environment, include position observation of the Ant for GoalEnv
        self.ant_env = AntEnv(
            xml_file=self.tmp_xml_file_path,
            exclude_current_positions_from_observation=False,
            render_mode=render_mode,
            reset_noise_scale=0.0,
            **kwargs,
        )
        self._model_names = MujocoModelNames(self.ant_env.model)
        self.target_site_id = self._model_names.site_name2id["target"]

        self.action_space = self.ant_env.action_space
        obs_shape: tuple = self.ant_env.observation_space.shape
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(
                    -np.inf, np.inf, shape=(obs_shape[0],), dtype="float64"
                ),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
            )
        )

        self.render_mode = render_mode
        EzPickle.__init__(
            self,
            render_mode,
            maze_map,
            reward_type,
            continuing_task,
            reset_target,
            **kwargs,
        )

        self.task = None

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed, **kwargs)

        self.ant_env.init_qpos[:2] = self.reset_pos

        obs, info = self.ant_env.reset(seed=seed)
        obs_dict = self._get_obs(obs)
        info["success"] = bool(
            np.linalg.norm(obs_dict["achieved_goal"] - self.goal) <= 0.45
        )

        return obs_dict, info

    def step(self, action):
        ant_obs, _, _, _, info = self.ant_env.step(action)
        obs = self._get_obs(ant_obs)

        reward_main = self.compute_reward(obs["achieved_goal"], self.goal, info)
        reward = reward_main
        reward_dict = {"reward_main": reward_main}
        if self.task is not None:
            reward, reward_dict = self.compute_reward_curriculum(ant_obs)
        terminated = self.compute_terminated(obs["achieved_goal"], self.goal, info)
        truncated = self.compute_truncated(obs["achieved_goal"], self.goal, info)
        info["success"] = bool(np.linalg.norm(obs["achieved_goal"] - self.goal) <= 0.45)
        info["reward_main"] = reward_main
        info["reward_task"] = reward
        info["reward_dict"] = reward_dict

        if self.render_mode == "human":
            self.render()

        # Update the goal position if necessary
        self.update_goal(obs["achieved_goal"])

        return obs, reward, terminated, truncated, info

    def _get_obs(self, ant_obs: np.ndarray) -> Dict[str, np.ndarray]:
        achieved_goal = ant_obs[:2]
        observation = ant_obs[:]

        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def update_target_site_pos(self):
        self.ant_env.model.site_pos[self.target_site_id] = np.append(
            self.goal, self.maze.maze_height / 2 * self.maze.maze_size_scaling
        )

    def render(self):
        return self.ant_env.render()

    def close(self):
        super().close()
        self.ant_env.close()

    def set_task(self, task):
        self.task = task

    def compute_reward_curriculum(self, ant_obs) -> Tuple[np.float64, Dict[str, np.float64]]:
        if self.task == "basic_locomotion":
            reward_1, reward_dict_1 = self.basic_locomotion_reward(ant_obs)
            reward_2, reward_dict_2 = self.orientation_control_reward(ant_obs)
            reward_1 = 3*reward_1
            # reward_dict_1 = reward_dict_1.update({k: 10*v for k, v in reward_dict_1.items()})
            total_reward = reward_1 + reward_2
            total_reward_dict = {**reward_dict_1,**reward_dict_2}
            return total_reward, total_reward_dict
        elif self.task == "orientation_control":
            return self.orientation_control_reward(ant_obs)
        elif self.task == "goal_orientation":
            reward_1, reward_dict_1 = self.basic_locomotion_reward(ant_obs)
            reward_2, reward_dict_2 = self.goal_orientation_reward(ant_obs)
            total_reward = reward_1 + reward_2
            total_reward_dict = {**reward_dict_1,**reward_dict_2}
            return total_reward, total_reward_dict
        elif self.task == "navigation_turning":
            return self.navigation_and_turning_reward(ant_obs)
        elif self.task == "original_task":
            reward_1, reward_dict_1 = self.original_task_reward(ant_obs)
            reward_2, reward_dict_2 = self.goal_orientation_reward(ant_obs)
            total_reward = reward_1 + reward_2
            total_reward_dict = {**reward_dict_1,**reward_dict_2}
            return total_reward, total_reward_dict
        else:
            raise ValueError(f"Task {self.task} not recognized.")
        
    def basic_locomotion_reward(self, ant_obs) -> Tuple[np.float64, Dict[str, np.float64]]:
        # reward components
        reward_velocity = 0.0
        rewards_dict = {}

        # parameters
        velocity_weight = 1.0  # Adjust this parameter to tune the importance of velocity in the reward function

        # calculate the torso velocity magnitude
        torso_vel = self.torso_velocity(ant_obs)
        torso_vel_magnitude = np.linalg.norm(torso_vel)

        # we use tanh to bound the velocity in [-1,1] range for reward shaping purposes
        reward_velocity = np.tanh(torso_vel_magnitude)
        
        # total reward
        reward = velocity_weight * reward_velocity

        # populate rewards dictionary
        rewards_dict['velocity'] = reward_velocity

        return reward, rewards_dict
    
    
    
    def orientation_control_reward(self, ant_obs: np.ndarray) -> Tuple[np.float64, Dict[str, np.float64]]:
        # Define desired orientation as a 4D vector (for yaw, pitch, roll)
        desired_orientation = np.array([1.0, 0.0, 0.0, 0.0])

        # Retrieve the orientation of the torso
        current_orientation = self.torso_orientation(ant_obs)

        # Calculate the L2 norm difference between current orientation and the desired orientation
        orientation_error = np.linalg.norm(current_orientation - desired_orientation)

        # Transform the orientation error into a reward using negative exponential to penalize deviation 
        orientation_reward_comp = np.exp(-orientation_error)

        # Reward weighting factor for orientation control (this should be tuned appropriately)
        orientation_weight = 0.3

        # Compute the weighted reward for orientation control
        orientation_reward = orientation_weight * orientation_reward_comp

        reward_components = {
            'orientation_control': orientation_reward
        }

        # Total reward is just the orientation control component in this case
        total_reward = np.sum(list(reward_components.values()))

        return total_reward, reward_components
    
    # def goal_orientation_reward(self, ant_obs) -> Tuple[np.float64, Dict[str, np.float64]]:
    #     # Weight factors for each reward component
    #     orientation_weight = 0.3
    #     distance_weight = 1.0
        
    #     # Calculate the torso orientation vector and goal direction vector
    #     torso_orientation = self.torso_orientation(ant_obs)
    #     goal_position = self.goal_pos()
    #     torso_position = self.torso_coordinate(ant_obs)
    #     direction_to_goal = goal_position - torso_position[:2]  # Only consider x and y for 2D plane
        
    #     # Since the initial orientation is [1, 0, 0, 0], we consider the forward direction to be along the x-axis
    #     # We assume that torso_orientation provides sufficient information to get the forward vector
    #     # If torso_orientation is a quaternion, this part might require conversion from quaternion to directional vector
    #     forward_vector = torso_orientation[:2]  # Only consider x and y for 2D plane
        
    #     # Normalize vectors to compare directions only
    #     direction_to_goal = direction_to_goal / np.linalg.norm(direction_to_goal) if np.linalg.norm(direction_to_goal) > 0 else direction_to_goal
    #     forward_vector = forward_vector / np.linalg.norm(forward_vector) if np.linalg.norm(forward_vector) > 0 else forward_vector
        
    #     # The dot product of two normalized vectors gives the cosine of the angle between them
    #     orientation_similarity = np.dot(forward_vector, direction_to_goal)  # Larger values are better as it indicates facing towards the goal
        
    #     # Transform the goal orientation similarity to a reward component using tanh to bound it
    #     orientation_reward = np.tanh(orientation_similarity)
        
    #     # Compute the distance to the goal
    #     goal_distance = self.goal_distance(ant_obs)

    #     # Transform the distance to goal into a reward component, penalizing larger distances
    #     distance_reward = -np.linalg.norm(goal_distance)

    #     # Combine the reward components with their respective weights
    #     total_reward = (orientation_weight * orientation_reward) + (distance_weight * distance_reward)
        
    #     # Create a dictionary of individual reward components for better understanding of the different reward contributions
    #     reward_components = {
    #         'orientation_reward': orientation_weight * orientation_reward,
    #         'distance_reward': distance_weight * distance_reward
    #     }

        # return total_reward, reward_components
    
    def goal_orientation_reward(self, ant_obs) -> Tuple[np.float64, Dict[str, np.float64]]:
        # Define weight for orientation reward component
        orientation_weight = 1.0
        
        # Calculate the distance to the goal
        distance_to_goal = self.goal_distance(ant_obs)
        
        # Compute the orientation reward component using tanh to encourage the ant to face towards the goal
        orientation_reward = -np.tanh(distance_to_goal)

        # Multiply by weight
        orientation_reward *= orientation_weight

        # Sum up the total reward
        total_reward = orientation_reward
        
        # Create a dictionary for individual reward components
        reward_components = {
            'orientation_reward': orientation_reward
        }
        
        return total_reward, reward_components
    
    def navigation_and_turning_reward(self, ant_obs: np.ndarray) -> Tuple[np.float64, Dict[str, np.float64]]:
        # Extract relevant information from observations
        torso_pos = self.torso_coordinate(ant_obs)
        torso_orient = self.torso_orientation(ant_obs)
        goal_position = self.goal_pos()
        
        # Components of the reward signal
        progress_weight = 1.0  # Weighting for forward progression
        orientation_weight = 0.5  # Weighting for torso orientation adjustment
        
        # Reward for moving forward, penalizing movements that are not in the x-direction
        progress_reward = np.tanh(torso_pos[0]) - 0.5 * np.linalg.norm(torso_pos[1:])
        
        # Compute the vector pointing toward the goal from the current position
        direction_to_goal = goal_position[:2] - torso_pos[:2]
        # Normalize vectors for computing the dot product
        forward_vector = np.array([1.0, 0.0])  # Assuming that the ant's forward direction aligns with the x-axis
        direction_to_goal /= (np.linalg.norm(direction_to_goal) + 1e-8)
        
        # Reward for torso orientation aligns with vector to the goal position
        orientation_reward = np.dot(direction_to_goal, forward_vector)
        # Use an arccos to evaluate how well the torso is oriented toward the goal (0 means facing)
        orientation_reward = np.arccos(orientation_reward) / np.pi   # Normalized between [0, 1]
        orientation_reward = np.tanh(1 - orientation_reward)          # Encourage to face the goal
        
        # Compute the total reward with the associated weights for each component
        reward = progress_weight * progress_reward + orientation_weight * orientation_reward
        
        # Create a dictionary to store the individual reward components
        reward_components = {
            'progress_reward': progress_weight * progress_reward,
            'orientation_reward': orientation_weight * orientation_reward,
        }
        
        return reward, reward_components
    
    def original_task_reward(self, ant_obs: np.ndarray) -> Tuple[np.float64, Dict[str, np.float64]]:
        # Constants for reward terms
        goal_distance_target = 0.45
        distance_weight = 1.0
        
        # Compute the distance from the goal
        distance = self.goal_distance(ant_obs)
        
        # Reward component for maintaining the desired goal distance (0.45)
        distance_reward = -distance_weight * np.linalg.norm(distance - goal_distance_target)
        
        # Total reward is the sum of individual components
        reward = distance_reward
        
        # Populate reward components as a dictionary for debugging or analysis
        reward_components = {
            'distance_reward': distance_reward,
        }
        
        return reward, reward_components

    def torso_coordinate(self, ant_obs: np.ndarray):
        xyz_coordinate = ant_obs[:3]

        return xyz_coordinate
    
    def torso_orientation(self, ant_obs: np.ndarray):
        xyz_orientation = ant_obs[3:7]

        return xyz_orientation

    def torso_velocity(self, ant_obs: np.ndarray):
        xyz_velocity = ant_obs[15:17]

        return xyz_velocity

    def torso_angular_velocity(self, ant_obs: np.ndarray):
        xyz_angular_velocity = ant_obs[18:21]

        return xyz_angular_velocity

    def goal_pos(self):

        return self.goal

    def goal_distance(self, ant_obs: np.ndarray):
        goal_pos = self.goal_pos()
        xyz_coordinate = self.torso_coordinate(ant_obs)
        distance = np.linalg.norm(goal_pos - xyz_coordinate[:2])

        return distance