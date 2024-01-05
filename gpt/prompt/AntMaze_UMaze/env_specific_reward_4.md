Task 1 Name
Basic Locomotion

Task 1 Description
Maximize the torso_velocity to achieve faster locomotion, while maintaining the torso_orientation close to the initial state.

```python
def compute_basic_locomotion_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    ant_obs = self.ant_env._get_obs()
    velocity_reward = np.linalg.norm(self.torso_velocity(ant_obs))

    # Define the initial orientation for reference
    initial_orientation = np.array([0.0, 0.75, 1.0])
    current_orientation = self.torso_orientation(ant_obs)
    orientation_error = np.linalg.norm(current_orientation - initial_orientation)
    
    # Temperature parameters for scaling
    velocity_temperature = 0.1
    orientation_temperature = 1.0
    
    # Normalize orientation error with e^(-error)
    orientation_penalty = -np.exp(-orientation_error / orientation_temperature)

    # Aggregate rewards and penalties
    reward = velocity_reward * velocity_temperature + orientation_penalty
    
    # Reward components dictionary
    reward_components = {
        "velocity_reward": velocity_reward * velocity_temperature,
        "orientation_penalty": orientation_penalty
    }

    return reward, reward_components
```

Task 2 Name
Stabilized Movement

Task 2 Description
Minimize the torso_angular_velocity to ensure stability during movement. Maintain the velocity achieved in Task 1.

```python
def compute_stabilized_movement_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    ant_obs = self.ant_env._get_obs()
    stability_penalty = np.linalg.norm(self.torso_angular_velocity(ant_obs))
    
    # Keep the maximized velocity from Task 1
    velocity_reward = np.linalg.norm(self.torso_velocity(ant_obs))
    
    # Temperature parameters for scaling
    stability_temperature = 0.1
    velocity_temperature = 0.1
    
    # Normalize stability penalty with e^(-penalty)
    stability_penalty_normalized = -np.exp(-stability_penalty / stability_temperature)

    # Aggregate rewards and penalties
    reward = velocity_reward * velocity_temperature + stability_penalty_normalized
    
    # Reward components dictionary
    reward_components = {
        "velocity_reward": velocity_reward * velocity_temperature,
        "stability_penalty_normalized": stability_penalty_normalized
    }

    return reward, reward_components
```

Task 3 Name
Goal-oriented Locomotion

Task 3 Description
Minimize the goal_distance, directing the agent towards the random goal_pos while maintaining the torso_orientation and torso_angular_velocity as stable as possible. Use the locomotion skills from Task 1 and stabilization from Task 2.

```python
def compute_goal_oriented_locomotion_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    ant_obs = self.ant_env._get_obs()
    distance_to_goal = self.goal_distance(ant_obs)
    orientation_penalty = np.linalg.norm(self.torso_orientation(ant_obs) - np.array([0.0, 0.75, 1.0]))
    stability_penalty = np.linalg.norm(self.torso_angular_velocity(ant_obs))
    
    # Temperature parameters for scaling
    goal_temperature = 10.0
    orientation_temperature = 0.5
    stability_temperature = 0.5
    
    # Calculate the normalized distance reward
    distance_reward = np.exp(-distance_to_goal / goal_temperature)
    
    # Normalize penalties
    orientation_penalty_normalized = -np.exp(-orientation_penalty / orientation_temperature)
    stability_penalty_normalized = -np.exp(-stability_penalty / stability_temperature)

    # Aggregate rewards and penalties
    reward = distance_reward + orientation_penalty_normalized + stability_penalty_normalized
    
    # Reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "orientation_penalty_normalized": orientation_penalty_normalized,
        "stability_penalty_normalized": stability_penalty_normalized
    }

    return reward, reward_components
```

Task 4 Name
Maze Navigation

Task 4 Description
Navigate towards the goal_pos while avoiding walls and obstacles, keeping the torso_coordinate within the maze boundaries. Apply the locomotion and stabilizing skills acquired in previous tasks to handle the maze's complexity.

```python
def compute_maze_navigation_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    ant_obs = self.ant_env._get_obs()
    distance_to_goal = self.goal_distance(ant_obs)
    stability_penalty = np.linalg.norm(self.torso_angular_velocity(ant_obs))
    body_pos = self.torso_coordinate(ant_obs)
    
    # Define the boundaries of the maze
    maze_boundaries = self.maze_size

    # Calculate the distance from boundaries (clamped at a minimum of 0)
    boundary_distances = np.clip(maze_boundaries - np.abs(body_pos[:2]), 0, np.inf)
    boundary_penalty = -np.sum(np.exp(-boundary_distances))  # Penalty for getting close to the boundaries
    
    # Temperature parameters for scaling
    goal_temperature = 10.0
    stability_temperature = 0.5
    boundary_temperature = 1.0
    
    # Normalize the distance to goal and the penalties
    distance_reward = np.exp(-distance_to_goal / goal_temperature)
    stability_penalty_normalized = -np.exp(-stability_penalty / stability_temperature)
    boundary_penalty_normalized = boundary_penalty / boundary_temperature

    # Aggregate rewards and penalties
    reward = distance_reward + stability_penalty_normalized + boundary_penalty_normalized
    
    # Reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "stability_penalty_normalized": stability_penalty_normalized,
        "boundary_penalty_normalized": boundary_penalty_normalized
    }

    return reward, reward_components
```

Task 5 Original Task
Effective Maze Solver

Task 5 Original Task Description
Maintain a goal_distance of 0.45, effectively reaching the target in the closed maze using the skills developed from all previous tasks.

```python
def compute_effective_maze_solver_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    ant_obs = self.ant_env._get_obs()
    distance_to_goal = self.goal_distance(ant_obs)
    
    # Define the ideal distance to be near the goal but not necessarily at the exact point
    ideal_distance = 0.45

    # Apply a heavy penalty if the distance is greater than the ideal distance
    if distance_to_goal > ideal_distance:
        distance_reward = -distance_to_goal
    else:
        # If within the ideal range, small positive reward for staying near the goal
        distance_reward = ideal_distance - distance_to_goal

    # Include previous tasks parameters
    orientation_penalty = np.linalg.norm(self.torso_orientation(ant_obs) - np.array([0.0, 0.75, 1.0]))
    stability_penalty = np.linalg.norm(self.torso_angular_velocity(ant_obs))

    # Temperature parameters for scaling
    distance_temperature = 1.0
    orientation_temperature = 0.5
    stability_temperature = 0.5
    
    # Normalize penalties
    orientation_penalty_normalized = -np.exp(-orientation_penalty / orientation_temperature)
    stability_penalty_normalized = -np.exp(-stability_penalty / stability_temperature)

    # Aggregate rewards and penalties
    reward = (distance_reward / distance_temperature) + orientation_penalty_normalized + stability_penalty_normalized
    
    # Reward components dictionary
    reward_components = {
        "distance_reward": distance_reward / distance_temperature,
        "orientation_penalty_normalized": orientation_penalty_normalized,
        "stability_penalty_normalized": stability_penalty_normalized
    }

    return reward, reward_components
```