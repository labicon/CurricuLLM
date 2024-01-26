Task 3 Name
Goal Orientation

Task 3 Description
The ant robot should orient itself to face and move towards the specified goal within the maze to minimize the distance between its current position and the goal.

```python
def goal_orientation_reward(self, ant_obs) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Weight factors for each reward component
    orientation_weight = 0.3
    distance_weight = 1.0
    
    # Calculate the torso orientation vector and goal direction vector
    torso_orientation = self.torso_orientation(ant_obs)
    goal_position = self.goal_pos()
    torso_position = self.torso_coordinate(ant_obs)
    direction_to_goal = goal_position - torso_position[:2]  # Only consider x and y for 2D plane
    
    # Since the initial orientation is [1, 0, 0, 0], we consider the forward direction to be along the x-axis
    # We assume that torso_orientation provides sufficient information to get the forward vector
    # If torso_orientation is a quaternion, this part might require conversion from quaternion to directional vector
    forward_vector = torso_orientation[:2]  # Only consider x and y for 2D plane
    
    # Normalize vectors to compare directions only
    direction_to_goal = direction_to_goal / np.linalg.norm(direction_to_goal) if np.linalg.norm(direction_to_goal) > 0 else direction_to_goal
    forward_vector = forward_vector / np.linalg.norm(forward_vector) if np.linalg.norm(forward_vector) > 0 else forward_vector
    
    # The dot product of two normalized vectors gives the cosine of the angle between them
    orientation_similarity = np.dot(forward_vector, direction_to_goal)  # Larger values are better as it indicates facing towards the goal
    
    # Transform the goal orientation similarity to a reward component using tanh to bound it
    orientation_reward = np.tanh(orientation_similarity)
    
    # Compute the distance to the goal
    goal_distance = self.goal_distance(ant_obs)

    # Transform the distance to goal into a reward component, penalizing larger distances
    distance_reward = -np.linalg.norm(goal_distance)

    # Combine the reward components with their respective weights
    total_reward = (orientation_weight * orientation_reward) + (distance_weight * distance_reward)
    
    # Create a dictionary of individual reward components for better understanding of the different reward contributions
    reward_components = {
        'orientation_reward': orientation_weight * orientation_reward,
        'distance_reward': distance_weight * distance_reward
    }

    return total_reward, reward_components
```