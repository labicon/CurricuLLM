Task 3 Name
Goal Orientation

Task 3 Description
The task "Goal Orientation" aims to direct the ant towards maintaining a specific orientation with respect to the goal. Here, we minimize the L2-norm difference between the ant's current orientation and a desired goal orientation. This encourages the ant to align itself with a specified target orientation.

```python
def goal_orientation_reward(self, ant_obs) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Desired goal orientation - this could be a fixed value or dynamic based on some strategy.
    # Given the starting state of orientation is [1.0, 0.0, 0.0, 0.0], a desired orientation to reach the goal might be similar.
    desired_goal_orientation = np.array([1.0, 0.0, 0.0, 0.0])
    
    # Extract current orientation of the ant's torso
    current_orientation = self.torso_orientation(ant_obs)
    
    # Compute the L2 norm difference between current and desired orientation
    orientation_difference = np.linalg.norm(current_orientation - desired_goal_orientation)

    # Reward for matching the orientation.
    # A weight factor can be used to scale the importance of this task.
    orientation_weight = 1.0
    orientation_reward = -orientation_weight * orientation_difference
    
    # Total reward is just the orientation reward in this case
    total_reward = orientation_reward

    # Dictionary to store individual reward components
    reward_components = {
        'orientation_reward': orientation_reward
    }
    
    return total_reward, reward_components
```

Task 4 Name
Minimize Goal Distance

Task 4 Description
The task "Minimize Goal Distance" focuses on reducing the distance between the current position of the ant's torso and the goal position. The reward function encourages the ant to move towards the goal as efficiently as possible, effectively solving the navigation problem in the maze.

```python
def minimize_goal_distance_reward(self, ant_obs) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Calculate the current distance to the goal
    current_goal_distance = self.goal_distance(ant_obs)
    
    # Reward for being close to the goal.
    # A weight factor can be used to scale the importance of minimizing the distance.
    goal_distance_weight = 1.0
    goal_distance_reward = -goal_distance_weight * current_goal_distance
    
    # Total reward is the negative of the goal distance to encourage minimization
    total_reward = goal_distance_reward

    # Dictionary to store individual reward components
    reward_components = {
        'goal_distance_reward': goal_distance_reward
    }
    
    return total_reward, reward_components
```