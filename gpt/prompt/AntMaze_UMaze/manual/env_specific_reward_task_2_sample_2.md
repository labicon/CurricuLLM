Task 3 Name
Goal Orientation

Task 3 Description
The task is to guide the ant robot to orient itself so that it is facing towards the goal location to facilitate its movement towards the target in the maze.

```python
def goal_orientation_reward(self, ant_obs) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Weighting parameter for the importance of goal orientation in the reward
    goal_orientation_weight = 1.0  # This could be tuned based on task requirements

    # Compute goal orientation. For this task, we consider only the yaw orientation
    # as it determines the facing direction of the ant in the 2D plane of the maze.
    goal_direction = self.goal_pos() - self.torso_coordinate(ant_obs)[:2]
    goal_direction /= np.linalg.norm(goal_direction)  # Normalize to get unit vector
    current_orientation = self.torso_orientation(ant_obs)[1:]  # Ignore w component
    current_orientation /= np.linalg.norm(current_orientation)  # Normalize
    
    # Compute the dot product to measure the alignment between current orientation and goal direction
    dot_product = np.dot(goal_direction, current_orientation)
    
    # Use arccos to get the angle, and then normalize it between 0 to 1 using np.tanh
    angle_to_goal = np.arccos(np.clip(dot_product, -1.0, 1.0))
    goal_orientation_reward = np.tanh(np.pi - angle_to_goal)  # We want to minimize the angle
    
    # Apply weighting
    weighted_goal_orientation_reward = goal_orientation_weight * goal_orientation_reward

    # Our total reward is based solely on goal orientation for this task
    total_reward = weighted_goal_orientation_reward
    
    # Reward components as a dictionary
    reward_components = {
        'goal_orientation_reward': weighted_goal_orientation_reward
    }
    
    return total_reward, reward_components
```

Task 3 
Goal Distance

Task 3 Description
Minimize the distance between the ant's current position and the goal location. This incentivizes the ant to move closer to the goal in the maze.

```python
def goal_distance_reward(self, ant_obs) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Weighting parameter for the importance of goal distance in the reward
    goal_distance_weight = 1.0  # This could be tuned based on task requirements

    # Calculate distance to the goal from the ant's current position
    distance_to_goal = self.goal_distance(ant_obs)

    # We want to minimize this distance, so we use the negation of the tanh function
    goal_distance_reward = -np.tanh(distance_to_goal)
    
    # Apply weighting
    weighted_goal_distance_reward = goal_distance_weight * goal_distance_reward
    
    # Our total reward is based solely on goal distance for this task
    total_reward = weighted_goal_distance_reward
    
    # Reward components as a dictionary
    reward_components = {
        'goal_distance_reward': weighted_goal_distance_reward
    }
    
    return total_reward, reward_components
```