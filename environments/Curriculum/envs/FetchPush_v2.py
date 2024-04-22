import os
from typing import Dict, List, Optional, Union, Tuple
import numpy as np

from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.envs.fetch import MujocoFetchEnv, MujocoPyFetchEnv

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "push.xml")


class MujocoPyFetchPushEnv(MujocoPyFetchEnv, EzPickle):
    def __init__(self, reward_type="sparse", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        MujocoPyFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)


class MujocoFetchPushEnv(MujocoFetchEnv, EzPickle):
    """
    ## Description

    This environment was introduced in ["Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research"](https://arxiv.org/abs/1802.09464).

    The task in the environment is for a manipulator to move a block to a target position on top of a table by pushing with its gripper. The robot is a 7-DoF [Fetch Mobile Manipulator](https://fetchrobotics.com/) with a two-fingered parallel gripper.
    The robot is controlled by small displacements of the gripper in Cartesian coordinates and the inverse kinematics are computed internally by the MuJoCo framework. The gripper is locked in a closed configuration in order to perform the push task.
    The task is also continuing which means that the robot has to maintain the block in the target position for an indefinite period of time.

    The control frequency of the robot is of `f = 25 Hz`. This is achieved by applying the same action in 20 subsequent simulator step (with a time step of `dt = 0.002 s`) before returning the control to the robot.

    ## Action Space

    The action space is a `Box(-1.0, 1.0, (4,), float32)`. An action represents the Cartesian displacement dx, dy, and dz of the end effector. In addition to a last action that controls closing and opening of the gripper.

    | Num | Action                                                 | Control Min | Control Max | Name (in corresponding XML file)                                | Joint | Unit         |
    | --- | ------------------------------------------------------ | ----------- | ----------- | --------------------------------------------------------------- | ----- | ------------ |
    | 0   | Displacement of the end effector in the x direction dx | -1          | 1           | robot0:mocap                                                    | hinge | position (m) |
    | 1   | Displacement of the end effector in the y direction dy | -1          | 1           | robot0:mocap                                                    | hinge | position (m) |
    | 2   | Displacement of the end effector in the z direction dz | -1          | 1           | robot0:mocap                                                    | hinge | position (m) |
    | 3   | -                                                      | -1          | 1           | -                                                               | hinge | position (m) |

    ## Observation Space

    The observation is a `goal-aware observation space`. It consists of a dictionary with information about the robot's end effector state and goal. The kinematics observations are derived from Mujoco bodies known as [sites](https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=site#body-site) attached to the body of interest such as the block or the end effector.
    Also to take into account the temporal influence of the step time, velocity values are multiplied by the step time dt=number_of_sub_steps*sub_step_time. The dictionary consists of the following 3 keys:

    * `observation`: its value is an `ndarray` of shape `(25,)`. It consists of kinematic information of the block object and gripper. The elements of the array correspond to the following:

    | Num | Observation                                                                                                                           | Min    | Max    | Site Name (in corresponding XML file) | Joint Name (in corresponding XML file) |Joint Type| Unit                     |
    |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|---------------------------------------|----------------------------------------|----------|--------------------------|
    | 0   | End effector x position in global coordinates                                                                                         | -Inf   | Inf    | robot0:grip                           |-                                       |-         | position (m)             |
    | 1   | End effector y position in global coordinates                                                                                         | -Inf   | Inf    | robot0:grip                           |-                                       |-         | position (m)             |
    | 2   | End effector z position in global coordinates                                                                                         | -Inf   | Inf    | robot0:grip                           |-                                       |-         | position (m)             |
    | 3   | Block x position in global coordinates                                                                                                | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
    | 4   | Block y position in global coordinates                                                                                                | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
    | 5   | Block z position in global coordinates                                                                                                | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
    | 6   | Relative block x position with respect to gripper x position in global coordinates. Equals to x<sub>block</sub> - x<sub>gripper</sub> | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
    | 7   | Relative block y position with respect to gripper y position in global coordinates. Equals to y<sub>block</sub> - y<sub>gripper</sub> | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
    | 8   | Relative block z position with respect to gripper z position in global coordinates. Equals to z<sub>block</sub> - z<sub>gripper</sub> | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
    | 9   | Joint displacement of the right gripper finger                                                                                        | -Inf   | Inf    |-                                      | robot0:r_gripper_finger_joint          | hinge    | position (m)             |
    | 10  | Joint displacement of the left gripper finger                                                                                         | -Inf   | Inf    |-                                      | robot0:l_gripper_finger_joint          | hinge    | position (m)             |
    | 11  | Global x rotation of the block in a XYZ Euler frame rotation                                                                          | -Inf   | Inf    | object0                               |-                                       |-         | angle (rad)              |
    | 12  | Global y rotation of the block in a XYZ Euler frame rotation                                                                          | -Inf   | Inf    | object0                               |-                                       |-         | angle (rad)              |
    | 13  | Global z rotation of the block in a XYZ Euler frame rotation                                                                          | -Inf   | Inf    | object0                               |-                                       |-         | angle (rad)              |
    | 14  | Relative block linear velocity in x direction with respect to the gripper                                                              | -Inf   | Inf    | object0                               |-                                       |-         | velocity (m/s)           |
    | 15  | Relative block linear velocity in y direction with respect to the gripper                                                              | -Inf   | Inf    | object0                               |-                                       |-         | velocity (m/s)           |
    | 16  | Relative block linear velocity in z direction                                                                                         | -Inf   | Inf    | object0                               |-                                       |-         | velocity (m/s)           |
    | 17  | Block angular velocity along the x axis                                                                                               | -Inf   | Inf    | object0                               |-                                       |-         | angular velocity (rad/s) |
    | 18  | Block angular velocity along the y axis                                                                                               | -Inf   | Inf    | object0                               |-                                       |-         | angular velocity (rad/s) |
    | 19  | Block angular velocity along the z axis                                                                                               | -Inf   | Inf    | object0                               |-                                       |-         | angular velocity (rad/s) |
    | 20  | End effector linear velocity x direction                                                                                              | -Inf   | Inf    | robot0:grip                           |-                                       |-         | velocity (m/s)           |
    | 21  | End effector linear velocity y direction                                                                                              | -Inf   | Inf    | robot0:grip                           |-                                       |-         | velocity (m/s)           |
    | 22  | End effector linear velocity z direction                                                                                              | -Inf   | Inf    | robot0:grip                           |-                                       |-         | velocity (m/s)           |
    | 23  | Right gripper finger linear velocity                                                                                                  | -Inf   | Inf    |-                                      | robot0:r_gripper_finger_joint          | hinge    | velocity (m/s)           |
    | 24  | Left gripper finger linear velocity                                                                                                   | -Inf   | Inf    |-                                      | robot0:l_gripper_finger_joint          | hinge    | velocity (m/s)           |

    * `desired_goal`: this key represents the final goal to be achieved. In this environment it is a 3-dimensional `ndarray`, `(3,)`, that consists of the three cartesian coordinates of the desired final block position `[x,y,z]`. In order for the robot to perform a push trajectory, the goal position can only be placed on top of the table. The elements of the array are the following:

    | Num | Observation                                                                                                                           | Min    | Max    | Site Name (in corresponding XML file) |Unit          |
    |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|---------------------------------------|--------------|
    | 0   | Final goal block position in the x coordinate                                                                                         | -Inf   | Inf    | target0                               | position (m) |
    | 1   | Final goal block position in the y coordinate                                                                                         | -Inf   | Inf    | target0                               | position (m) |
    | 2   | Final goal block position in the z coordinate                                                                                         | -Inf   | Inf    | target0                               | position (m) |

    * `achieved_goal`: this key represents the current state of the block, as if it would have achieved a goal. This is useful for goal orientated learning algorithms such as those that use [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) (HER). The value is an `ndarray` with shape `(3,)`. The elements of the array are the following:

    | Num | Observation                                                                                                                           | Min    | Max    | Site Name (in corresponding XML file) |Unit          |
    |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|---------------------------------------|--------------|
    | 0   | Current block position in the x coordinate                                                                                            | -Inf   | Inf    | object0                               | position (m) |
    | 1   | Current block position in the y coordinate                                                                                            | -Inf   | Inf    | object0                               | position (m) |
    | 2   | Current block position in the z coordinate                                                                                            | -Inf   | Inf    | object0                               | position (m) |


    ## Rewards

    The reward can be initialized as `sparse` or `dense`:
    - *sparse*: the returned reward can have two values: `-1` if the block hasn't reached its final target position, and `0` if the block is in the final target position (the block is considered to have reached the goal if the Euclidean distance between both is lower than 0.05 m).
    - *dense*: the returned reward is the negative Euclidean distance between the achieved goal position and the desired goal.

    To initialize this environment with one of the mentioned reward functions the type of reward must be specified in the id string when the environment is initialized. For `sparse` reward the id is the default of the environment, `FetchPush-v2`. However, for `dense` reward the id must be modified to `FetchPush-v2` and initialized as follows:

    ```python
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)

    env = gym.make('FetchPushDense-v2')
    ```

    ## Starting State

    When the environment is reset the gripper is placed in the following global cartesian coordinates `(x,y,z) = [1.3419 0.7491 0.555] m`, and its orientation in quaternions is `(w,x,y,z) = [1.0, 0.0, 1.0, 0.0]`. The joint positions are computed by inverse kinematics internally by MuJoCo. The base of the robot will always be fixed at `(x,y,z) = [0.405, 0.48, 0]` in global coordinates.

    The block's position has a fixed height of `(z) = [0.42] m ` (on top of the table). The initial `(x,y)` position of the block is the gripper's x and y coordinates plus an offset sampled from a uniform distribution with a range of `[-0.15, 0.15] m`. Offset samples are generated until the 2-dimensional Euclidean distance from the gripper to the block is greater than `0.1 m`.
    The initial orientation of the block is the same as for the gripper, `(w,x,y,z) = [1.0, 0.0, 1.0, 0.0]`.

    Finally the target position where the robot has to move the block is generated. The target can be in mid-air or over the table. The random target is also generated by adding an offset to the initial grippers position `(x,y)` sampled from a uniform distribution with a range of `[-0.15, 0.15] m`. The height of the target is initialized at `(z) = [0.42] m ` on the table.


    ## Episode End

    The episode will be `truncated` when the duration reaches a total of `max_episode_steps` which by default is set to 50 timesteps.
    The episode is never `terminated` since the task is continuing with infinite horizon.

    ## Arguments

    To increase/decrease the maximum number of timesteps before the episode is `truncated` the `max_episode_steps` argument can be set at initialization.
    The default value is 50. For example, to increase the total number of timesteps to 100 make the environment as follows:

    ```python
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)

    env = gym.make('FetchPush-v2', max_episode_steps=100)
    ```

    ## Version History

    * v2: the environment depends on the newest [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html) maintained by the MuJoCo team in Deepmind.
    * v1: the environment depends on `mujoco_py` which is no longer maintained.
    """

    def __init__(self, reward_type="sparse", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        MujocoFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

    def step(self, action):
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (np.ndarray): Control action to be applied to the agent and update the simulation. Should be of shape :attr:`action_space`.

        Returns:
            observation (dictionary): Next observation due to the agent actions .It should satisfy the `GoalEnv` :attr:`observation_space`.
            reward (integer): The reward as a result of taking the action. This is calculated by :meth:`compute_reward` of `GoalEnv`.
            terminated (boolean): Whether the agent reaches the terminal state. This is calculated by :meth:`compute_terminated` of `GoalEnv`.
            truncated (boolean): Whether the truncation condition outside the scope of the MDP is satisfied. Timically, due to a timelimit, but
            it is also calculated in :meth:`compute_truncated` of `GoalEnv`.
            info (dictionary): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging). In this case there is a single
            key `is_success` with a boolean value, True if the `achieved_goal` is the same as the `desired_goal`.
        """
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)

        self._mujoco_step(action)

        self._step_callback()

        if self.render_mode == "human":
            self.render()
        obs = self._get_obs()

        info = {
            "success": self._is_success(obs["achieved_goal"], self.goal),
        }

        terminated = self.compute_terminated(obs["achieved_goal"], self.goal, info)
        truncated = self.compute_truncated(obs["achieved_goal"], self.goal, info)

        reward_main = self.compute_reward(obs["achieved_goal"], self.goal, info)
        reward, reward_dict = self.compute_reward_curriculum()
        reward_dict["main"] = reward_main
        reward_dict["task"] = reward

        info["reward_dict"] = reward_dict

        return obs, reward, terminated, truncated, info

    def end_effector_position(self):
        (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        ) = self.generate_mujoco_observations()    

        return grip_pos  

    def block_position(self):
        (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        ) = self.generate_mujoco_observations()    

        return object_pos  

    def block_relative_linear_velocity(self):
        (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        ) = self.generate_mujoco_observations()    

        return object_velp

    def end_effector_linear_velocity(self):
        (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        ) = self.generate_mujoco_observations()    

        return grip_velp

    def goal_position(self):
        return self.goal.copy()

    def compute_reward_0(self):
        # Define weights for different components of the reward
        weight_alignment = 1.0
    
        # Fixed point on the table (not occupied by the block) for alignment
        # This value should be chosen such that it does not coincide with the block's initial or target positions.
        # Assuming the fixed point as [1.1, 0.5, 0.42] for the sake of example.
        fixed_point = np.array([1.1, 0.5, 0.42])
        
        # Getting end effector's position
        end_effector_pos = self.end_effector_position()
    
        # Calculate the Euclidean distance (L2 norm) between end effector and fixed point
        alignment_error = np.linalg.norm(end_effector_pos - fixed_point)
    
        # Reward for aligning end effector with the fixed point. 
        # The reward increases as the end effector gets closer to the fixed point.
        # Using -alignment_error because we want to minimize this distance (higher reward for smaller distance)
        reward_alignment = -weight_alignment * alignment_error
    
        # Total Reward: Currently focusing on alignment only
        total_reward = reward_alignment
    
        # Dictionary containing individual components of the reward
        reward_dict = {
            "reward_alignment": reward_alignment,
        }
    
        return total_reward, reward_dict
    
    def compute_reward_1(self):
        # Constants for reward computation
        distance_weight = -1.0  # Weight for distance between end effector and block
        
        # Getting necessary observations
        end_effector_pos = self.end_effector_position()
        block_pos = self.block_position()
        
        # Compute distance between end effector and block
        distance = np.linalg.norm(end_effector_pos - block_pos)
        
        # Reward for minimizing the distance
        distance_reward = distance_weight * distance
        
        # Total reward computation
        reward = distance_reward
        
        # Dictionary for individual component rewards
        reward_dict = {
            'distance_reward': distance_reward,
        }
        
        return reward, reward_dict
    
    def compute_reward_2(self):
        # Extract relevant positions and velocities from the environment
        end_effector_pos = self.end_effector_position()
        block_pos = self.block_position()
        end_effector_vel = self.end_effector_linear_velocity()
        
        # Constants
        velocity_maintain_weight = 1.0
        alignment_maximize_weight = 1.0
        
        # Calculate the L2 norm of the end_effector velocity to encourage it to be zero
        velocity_maintain_reward = -np.linalg.norm(end_effector_vel) * velocity_maintain_weight
        
        # Calculate the alignment by taking the negative L2 norm between end_effector and block positions (only x and y)
        # We remove the z component by creating a copy of the positions with z set to 0
        alignment_maximize_reward = -np.linalg.norm(end_effector_pos[:2] - block_pos[:2]) * alignment_maximize_weight
        
        # Calculate total reward by summing individual rewards
        total_reward = velocity_maintain_reward + alignment_maximize_reward
        
        # Create a dictionary to hold the individual components of the reward for debugging and analysis
        reward_components = {
            "velocity_maintain_reward": velocity_maintain_reward,
            "alignment_maximize_reward": alignment_maximize_reward
        }
        
        # Return the total reward and the dictionary of reward components
        return total_reward, reward_components
    
    def compute_reward_3(self):
        block_relative_velocity = self.block_relative_linear_velocity()
        velocity_reward_weight = 1.0  # Weight for the velocity term in the reward function
    
        # Calculate reward components
        # We want to maximize the relative velocity of the block, indicating effective pushing.
        # np.linalg.norm provides the magnitude of the velocity vector.
        # np.tanh ensures the reward doesn't grow unbounded and remains within [-1, 1].
        velocity_reward = np.tanh(np.linalg.norm(block_relative_velocity))
    
        # Compile the total reward and detailed reward components in a dictionary for clarity
        total_reward = velocity_reward_weight * velocity_reward
        reward_components = {
            "velocity_reward": velocity_reward,
        }
    
        return total_reward, reward_components
    
    def compute_reward_4(self):
        # Constants
        distance_weight = 1.0  # Weight for the distance component of the reward
        
        # Observations
        block_pos = self.block_position()  # Get the block's position
        goal_pos = self.goal_position()  # Get the goal's position
        
        # Computes the L2 norm (euclidean distance) between the block and the goal position
        distance_to_goal = np.linalg.norm(block_pos - goal_pos)
        
        # Reward components
        distance_reward = -distance_weight * distance_to_goal  # We want to minimize the distance
        
        # Total reward
        reward = distance_reward
        
        # Dictionary containing each reward component
        reward_dict = {
            'distance_to_goal': distance_reward,
        }
        
        return reward, reward_dict
    
    # Function to loop through compute_reward_X functions and sum their outputs
    def compute_reward_curriculum(self):
        total_reward = 0
        total_reward_dict = {}
        n = 4
        for i in range(n + 1):  # Including n, hence n + 1
            # Construct the function name based on i
            function_name = f'compute_reward_{i}'
            # Get the function by name and call it
            function = getattr(self, function_name, None)
            if function:
                # Call the function and add its return value to the total sum
                reward, reward_dict = function()
                total_reward += reward
                total_reward_dict.update(reward_dict)
            else:
                raise NameError(f"Function {function_name} not found.")
        return total_reward, total_reward_dict