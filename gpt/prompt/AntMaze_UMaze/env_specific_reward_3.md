Task 1 Name
Basic Locomotion

Task 1 Description
Maximize the torso_velocity to achieve faster locomotion, while maintaining the torso_orientation close to the initial state.

```python
def compute_basic_locomotion_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Extract relevant observation information
    ant_obs = self.ant_env.get_obs()
    torso_velocity = self.torso_velocity(ant_obs)
    torso_orientation = self.torso_orientation(ant_obs)
    
    # Reward components
    velocity_reward = np.linalg.norm(torso_velocity)
    orientation_penalty = np.linalg.norm(torso_orientation - np.array([0.0, 0.75, 1.0]))
    
    # Temperature parameters for scaling
    velocity_temp = 0.1
    orientation_temp = 1.0
    
    # Normalize and scale reward components using exponential transformation
    norm_velocity_reward = np.exp(velocity_temp * velocity_reward)
    norm_orientation_penalty = np.exp(-orientation_temp * orientation_penalty)
    
    # Calculate the total reward
    reward = norm_velocity_reward - norm_orientation_penalty
    
    # Create reward components dictionary
    reward_components = {
        "velocity_reward": norm_velocity_reward,
        "orientation_penalty": norm_orientation_penalty
    }
    
    return reward, reward_components
```

Task 2 Name
Stabilized Movement

Task 2 Description
Minimize the torso_angular_velocity to ensure stability during movement. Maintain the velocity achieved in Task 1.

```python
def compute_stabilized_movement_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Extract relevant observation information
    ant_obs = self.ant_env.get_obs()
    torso_angular_velocity = self.torso_angular_velocity(ant_obs)
    torso_velocity = self.torso_velocity(ant_obs)
    
    # Reward components
    stability_reward = -np.linalg.norm(torso_angular_velocity)
    velocity_reward = np.linalg.norm(torso_velocity)
    
    # Temperature parameters for scaling
    stability_temp = 1.0
    velocity_temp = 0.1
    
    # Normalize and scale reward components using exponential transformation
    norm_stability_reward = np.exp(stability_temp * stability_reward)
    norm_velocity_reward = np.exp(velocity_temp * velocity_reward)
    
    # Calculate the total reward
    reward = norm_stability_reward + norm_velocity_reward
    
    # Create reward components dictionary
    reward_components = {
        "stability_reward": norm_stability_reward,
        "velocity_reward": norm_velocity_reward
    }
    
    return reward, reward_components
```

Task 3 Name
Goal-oriented Locomotion

Task 3 Description
Minimize the goal_distance, directing the agent towards the random goal_pos while maintaining the torso_orientation and torso_angular_velocity as stable as possible. Use the locomotion skills from Task 1 and stabilization from Task 2.

```python
def compute_goal_oriented_locomotion_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Extract relevant observation information
    ant_obs = self.ant_env.get_obs()
    torso_orientation = self.torso_orientation(ant_obs)
    torso_angular_velocity = self.torso_angular_velocity(ant_obs)
    goal_distance = self.goal_distance(ant_obs)
    
    # Reward components
    distance_reward = -goal_distance
    orientation_penalty = np.linalg.norm(torso_orientation - np.array([0.0, 0.75, 1.0]))
    stability_penalty = np.linalg.norm(torso_angular_velocity)
    
    # Temperature parameters for scaling
    distance_temp = 1.0
    orientation_temp = 1.0
    stability_temp = 1.0
    
    # Normalize and scale reward components using exponential transformation
    norm_distance_reward = np.exp(distance_temp * distance_reward)
    norm_orientation_penalty = np.exp(-orientation_temp * orientation_penalty)
    norm_stability_penalty = np.exp(-stability_temp * stability_penalty)
    
    # Calculate the total reward
    reward = norm_distance_reward - norm_orientation_penalty - norm_stability_penalty
    
    # Create reward components dictionary
    reward_components = {
        "distance_reward": norm_distance_reward,
        "orientation_penalty": norm_orientation_penalty,
        "stability_penalty": norm_stability_penalty
    }
    
    return reward, reward_components
```

Task 4 Name
Maze Navigation

Task 4 Description
Navigate towards the goal_pos while avoiding walls and obstacles, keeping the torso_coordinate within the maze boundaries. Apply the locomotion and stabilizing skills acquired in previous tasks to handle the maze's complexity.

```python
# Note: To complete this task, custom methods to calculate wall proximity and boundary checks need to be implemented in the environment class.

def compute_maze_navigation_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    # ... (Implement custom wall and boundary check methods before completing this function)
    
    return reward, reward_components
```

Task 5 Original Task
Effective Maze Solver

Task 5 Original Task Description
Maintain a goal_distance of 0.45, effectively reaching the target in the closed maze using the skills developed from all previous tasks.

```python
def compute_effective_maze_solver_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Extract relevant observation information
    ant_obs = self.ant_env.get_obs()
    goal_distance = self.goal_distance(ant_obs)
    
    # Constant to maintain goal_distance
    desired_distance = 0.45
    
    # Reward components
    distance_to_target = np.abs(goal_distance - desired_distance)
    
    # Temperature parameter for scaling
    distance_temp = 1.0
    
    # Normalize and scale reward component using exponential transformation
    norm_distance_to_target = np.exp(-distance_temp * distance_to_target)
    
    # Calculate the total reward
    reward = norm_distance_to_target
    
    # Create reward components dictionary
    reward_components = {
        "distance_to_target_reward": norm_distance_to_target
    }
    
    return reward, reward_components
```