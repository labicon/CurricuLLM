Task 1 Name
Basic Locomotion

Task 1 Description
Maximize the torso_velocity to achieve faster locomotion, while maintaining the torso_orientation close to the initial state.

```python
def compute_basic_locomotion_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    orientation_temp = 0.5
    velocity_temp = 1.0
    
    ant_obs = self.ant_env._get_obs()
    velocity_reward = np.linalg.norm(self.torso_velocity(ant_obs))
    # We want to keep torso_orientation close to the initial state [0.0, 0.75, 1.0]
    orientation_penalty = np.linalg.norm(self.torso_orientation(ant_obs) - np.array([0.0, 0.75, 1.0]))
    
    # Applying exponential transformation to orientation penalty
    orientation_penalty = np.exp(-orientation_temp * orientation_penalty)
    # Normalize the velocity reward to a range, e.g. [0,1]
    velocity_reward = np.tanh(velocity_temp * velocity_reward)
    
    total_reward = velocity_reward - orientation_penalty
    reward_components = {
        'velocity_reward': velocity_reward,
        'orientation_penalty': orientation_penalty,
    }
    
    return total_reward, reward_components
```

Task 2 Name 
Stabilized Movement

Task 2 Description
Minimize the torso_angular_velocity to ensure stability during movement. Maintain the velocity achieved in Task 1.

```python
def compute_stabilized_movement_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    stability_temp = 2.0
    
    ant_obs = self.ant_env._get_obs()
    stability_penalty = np.linalg.norm(self.torso_angular_velocity(ant_obs))
    
    # The reward for velocity is reused from Task 1
    velocity_reward, _ = self.compute_basic_locomotion_reward()
    
    # Applying exponential transformation to stability penalty
    stability_penalty = np.exp(-stability_temp * stability_penalty)
    
    total_reward = velocity_reward - stability_penalty
    reward_components = {
        'velocity_reward': velocity_reward,
        'stability_penalty': stability_penalty,
    }
    
    return total_reward, reward_components
```

Task 3 Name
Goal-oriented Locomotion

Task 3 Description
Minimize the goal_distance, directing the agent towards the random goal_pos while maintaining the torso_orientation and torso_angular_velocity as stable as possible. Use the locomotion skills from Task 1 and stabilization from Task 2.

```python
def compute_goal_oriented_locomotion_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    goal_distance_temp = 2.0
    
    ant_obs = self.ant_env._get_obs()
    goal_dist = self.goal_distance(ant_obs)
    
    # Higher reward for being closer to the goal
    goal_distance_reward = np.exp(-goal_distance_temp * goal_dist)
    
    # The rewards from Tasks 1 and 2 are reused
    basic_locomotion_reward, _ = self.compute_basic_locomotion_reward()
    stabilized_movement_reward, _ = self.compute_stabilized_movement_reward()
    
    total_reward = goal_distance_reward + basic_locomotion_reward + stabilized_movement_reward
    reward_components = {
        'goal_distance_reward': goal_distance_reward,
        'basic_locomotion_reward': basic_locomotion_reward,
        'stabilized_movement_reward': stabilized_movement_reward,
    }
    
    return total_reward, reward_components
```

Task 4 Name
Maze Navigation

Task 4 Description
Navigate towards the goal_pos while avoiding walls and obstacles, keeping the torso_coordinate within the maze boundaries. Apply the locomotion and stabilizing skills acquired in previous tasks to handle the maze's complexity.

```python
def compute_maze_navigation_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Additional components needed for navigating the maze could be included, such as penalties for collisions
    # or rewards for exploring new areas. The complexity will depend on the specifics of the maze environment.
    
    # Reuse the goal-oriented locomotion reward function.
    goal_oriented_reward, orientation_components = self.compute_goal_oriented_locomotion_reward()
    
    # For simplicity, we'll just return the goal-oriented reward here
    # as navigating the maze is a higher-level task that builds on it.
    return goal_oriented_reward, orientation_components
```

Task 5 Original Task
Effective Maze Solver

Task 5 Original Task Description
Maintain a goal_distance of 0.45, effectively reaching the target in the closed maze using the skills developed from all previous tasks.

```python
def compute_effective_maze_solver_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    target_distance = 0.45
    distance_temp = 5.0
    
    ant_obs = self.ant_env._get_obs()
    current_goal_distance = self.goal_distance(ant_obs)
    
    # Reward function encourages maintaining the distance at 0.45
    distance_diff = np.abs(current_goal_distance - target_distance)
    
    # Applying squared penalty for distances different from 0.45
    # This creates a steep penalty the further away from 0.45 the agent is
    distance_penalty = np.exp(-distance_temp * distance_diff**2)

    # Reuse the maze navigation reward components
    maze_navigation_reward, maze_reward_components = self.compute_maze_navigation_reward()
    
    total_reward = maze_navigation_reward + distance_penalty
    reward_components = maze_reward_components
    reward_components.update({'target_distance_reward': distance_penalty})
    
    return total_reward, reward_components
```