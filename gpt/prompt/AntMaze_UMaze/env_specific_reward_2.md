Task 1 Name
Basic Locomotion

Task 1 Description
Maximize the torso_velocity to achieve faster locomotion, while maintaining the torso_orientation close to the initial state.

```python
def compute_basic_locomotion_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    ant_obs = self.ant_env._get_obs() # Assuming ant_env object has a method to get observations
    velocity = np.linalg.norm(self.torso_velocity(ant_obs))
    orientation_diff = np.abs(self.torso_orientation(ant_obs) - np.array([0.0, 0.75, 1.0]))
    
    # Define temperature parameters for normalizing rewards
    orientation_temp = 0.1
    velocity_temp = 1.0
    
    # Reward for moving faster
    velocity_reward = np.tanh(velocity / velocity_temp)
    # Penalty for deviating from initial torso_orientation
    orientation_penalty = -np.tanh(np.sum(orientation_diff) / orientation_temp)
    
    # Combine rewards
    reward = velocity_reward + orientation_penalty
    
    # Detail the reward components
    reward_components = {
        'velocity_reward': velocity_reward,
        'orientation_penalty': orientation_penalty
    }
    
    return np.float64(reward), reward_components
```

Task 2 Name
Stabilized Movement

Task 2 Description
Minimize the torso_angular_velocity to ensure stability during movement. Maintain the velocity achieved in Task 1.

```python
def compute_stabilized_movement_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    ant_obs = self.ant_env._get_obs()
    angular_velocity = np.linalg.norm(self.torso_angular_velocity(ant_obs))
    velocity = np.linalg.norm(self.torso_velocity(ant_obs))
    
    # Define temperature parameters
    stability_temp = 0.05
    velocity_temp = 1.0
    
    # Reward for stable movement (low angular velocity)
    stability_reward = -np.tanh(angular_velocity / stability_temp)
    # Maintain the velocity achieved in Task 1
    velocity_reward = np.tanh(velocity / velocity_temp)
    
    # Combine rewards
    reward = stability_reward + velocity_reward
    
    # Detail the reward components
    reward_components = {
        'stability_reward': stability_reward,
        'velocity_reward': velocity_reward
    }
    
    return np.float64(reward), reward_components
```

Task 3 Name
Goal-oriented Locomotion

Task 3 Description
Minimize the goal_distance, directing the agent towards the random goal_pos while maintaining the torso_orientation and torso_angular_velocity as stable as possible. Use the locomotion skills from Task 1 and stabilization from Task 2.

```python
def compute_goal_oriented_locomotion_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    ant_obs = self.ant_env._get_obs()
    distance_to_goal = self.goal_distance(ant_obs)
    orientation_diff = np.abs(self.torso_orientation(ant_obs) - np.array([0.0, 0.75, 1.0]))
    angular_velocity = np.linalg.norm(self.torso_angular_velocity(ant_obs))

    # Define temperature parameters
    goal_temp = 0.2
    orientation_temp = 0.1
    stability_temp = 0.05

    # Reward for moving towards the goal
    goal_reward = -np.tanh(distance_to_goal / goal_temp)
    # Penalty for deviating from initial torso_orientation
    orientation_penalty = -np.tanh(np.sum(orientation_diff) / orientation_temp)
    # Penalty for instability
    stability_penalty = -np.tanh(angular_velocity / stability_temp)
    
    # Combine rewards
    reward = goal_reward + orientation_penalty + stability_penalty
    
    # Detail the reward components
    reward_components = {
        'goal_reward': goal_reward,
        'orientation_penalty': orientation_penalty,
        'stability_penalty': stability_penalty
    }
    
    return np.float64(reward), reward_components
```

Task 4 Name
Maze Navigation

Task 4 Description
Navigate towards the goal_pos while avoiding walls and obstacles, keeping the torso_coordinate within the maze boundaries. Apply the locomotion and stabilizing skills acquired in previous tasks to handle the maze's complexity.

```python
# Note: Implementation of this task depends on the method available for checking collisions
# with walls and whether the ant's position is within maze boundaries. Pseudocode is provided.

def compute_maze_navigation_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    ant_obs = self.ant_env._get_obs()
    distance_to_goal = self.goal_distance(ant_obs)

    # Define temperature parameters
    goal_temp = 0.3
    
    # Reward for moving towards the goal
    goal_reward = -np.tanh(distance_to_goal / goal_temp)
    
    wall_penalty = 0
    # Check if ant has hit the wall or is outside the maze boundary
    is_collision, is_outside = self.check_collision_and_boundaries(ant_obs) # This function needs to be defined
    if is_collision or is_outside:
        wall_penalty = -1  # Large penalty for hitting a wall or going out of bounds
    
    # Combine rewards
    reward = goal_reward + wall_penalty
    
    # Detail the reward components
    reward_components = {
        'goal_reward': goal_reward,
        'wall_penalty': wall_penalty
    }
    
    return np.float64(reward), reward_components
```

Task 5 Original Task
Effective Maze Solver

Task 5 Original Task Description
Maintain a goal_distance of 0.45, effectively reaching the target in the closed maze using the skills developed from all previous tasks.

```python
def compute_effective_maze_solver_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    ant_obs = self.ant_env._get_obs()
    distance_to_goal = self.goal_distance(ant_obs)

    # Define temperature parameters
    goal_temp = 0.1
    
    # Compute the absolute difference from the target distance
    distance_diff = np.abs(distance_to_goal - 0.45)

    # Reward for being at the optimal distance to the goal
    optimal_distance_reward = -np.tanh(distance_diff / goal_temp)
    
    # Combine rewards
    reward = optimal_distance_reward
    
    # Detail the reward components
    reward_components = {
        'optimal_distance_reward': optimal_distance_reward
    }
    
    return np.float64(reward), reward_components
```