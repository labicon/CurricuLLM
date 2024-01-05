Task 1 Name
Basic Locomotion

```python
def compute_basic_locomotion_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    orientation_temp = 1.0  # Temperature parameter to scale the orientation reward
    velocity_temp = 1.0     # Temperature parameter to scale the velocity reward
    
    ant_obs = self._get_obs()
    
    # Get the torso orientation and velocity from the environment
    torso_orientation = self.torso_orientation(ant_obs)
    torso_velocity = self.torso_velocity(ant_obs)
    
    # Initial orientation for comparison
    initial_orientation = np.array([0.0, 0.75, 1.0])
    
    # Calculate how much the torso orientation deviates from the initial state
    orientation_deviation = np.linalg.norm(torso_orientation - initial_orientation)
    
    # Reward for being close to the initial orientation
    orientation_reward = np.exp(-orientation_deviation**2 / orientation_temp)
    
    # Normalize the forward velocity to be between 0 and 1, assuming a known max velocity
    max_velocity = 10.0  # This is an arbitrary value and should be replaced by the actual max velocity if known
    normalized_velocity = min(np.linalg.norm(torso_velocity) / max_velocity, 1.0)
    
    # Calculate the reward for moving fast forward
    velocity_reward = np.exp(normalized_velocity / velocity_temp)
    
    # Combine rewards with respective weights
    total_reward = 0.5 * orientation_reward + 0.5 * velocity_reward
    
    reward_components = {
        "orientation_reward": orientation_reward,
        "velocity_reward": velocity_reward
    }
    
    return total_reward, reward_components
```

Task 2 Name
Stabilized Movement

```python
def compute_stabilized_movement_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    stability_temp = 1.0   # Temperature parameter to scale the stability reward
    
    ant_obs = self._get_obs()
    
    # Get the torso angular velocity from the environment
    torso_angular_velocity = self.torso_angular_velocity(ant_obs)
    
    # Calculate the magnitude of torso angular velocity for stability
    stability_deviation = np.linalg.norm(torso_angular_velocity)
    
    # Reward for being stable (having low angular velocity)
    stability_reward = np.exp(-stability_deviation**2 / stability_temp)
    
    # Get the basic locomotion reward to maintain forward velocity
    velocity_reward, _ = self.compute_basic_locomotion_reward()
    
    # Combine rewards with respective weights
    total_reward = 0.5 * stability_reward + 0.5 * velocity_reward
    
    reward_components = {
        "stability_reward": stability_reward,
        "velocity_reward": velocity_reward
    }
    
    return total_reward, reward_components
```

Task 3 Name
Goal-oriented Locomotion

```python
def compute_goal_oriented_locomotion_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    orientation_temp = 1.0  # Temperature parameter to scale the orientation reward
    goal_distance_temp = 1.0 # Temperature parameter to scale the goal distance reward
    
    ant_obs = self._get_obs()
    
    # Get the goal distance from the environment
    goal_distance = self.goal_distance(ant_obs)
    
    # Reward for being closer to the goal
    goal_distance_reward = np.exp(-goal_distance**2 / goal_distance_temp)
    
    # Get the stabilized movement reward to maintain stability
    stability_reward, _ = self.compute_stabilized_movement_reward()
    
    # Combine rewards with respective weights
    total_reward = 0.5 * goal_distance_reward + 0.5 * stability_reward
    
    reward_components = {
        "goal_distance_reward": goal_distance_reward,
        "stability_reward": stability_reward
    }
    
    return total_reward, reward_components
```

Task 4 Name
Maze Navigation

```python
def compute_maze_navigation_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Since the environment and mechanics of maze navigation (like avoiding walls) are not elaborated
    # The reward components for avoiding walls and staying within boundaries need to be defined based on those mechanics.
    # Assuming those components exist, they would be integrated here along with goal-distance and stabilization.
    
    ant_obs = self._get_obs()
    
    # Integrate goal-oriented locomotion reward as a base
    total_reward, reward_components = self.compute_goal_oriented_locomotion_reward()
    
    # Add reward components for maze navigation specifics, e.g., penalties for hitting walls or rewards for correct direction
    
    # Incorporate additional rewards and penalties here
    
    return total_reward, reward_components
```

Task 5 Original Task
Effective Maze Solver

```python
def original_task_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    desired_goal_distance = 0.45  # Assuming the task is to stay close to this distance from the goal
    
    ant_obs = self._get_obs()
    
    # Calculate the current distance to the goal
    current_distance = self.goal_distance(ant_obs)
    
    # Reward for being close to the desired goal distance
    goal_reward = np.exp(-abs(current_distance-desired_goal_distance))
    
    # Integrate maze navigation reward as a base
    total_reward, reward_components = self.compute_maze_navigation_reward()
    
    # Weight the total reward with the goal reward
    total_reward = 0.7 * total_reward + 0.3 * goal_reward
    
    # Update the reward components
    reward_components["goal_distance_reward"] = goal_reward
    
    return total_reward, reward_components
```