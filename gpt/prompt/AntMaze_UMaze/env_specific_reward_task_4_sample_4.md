Task 5 Name
Original Task

Task 5 Description
The reward function encourages the ant to minimize the distance between its current position and the goal position to approximately 0.45 units.

```python
def original_task_reward(self, ant_obs) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Define the goal distance value to be achieved
    goal_distance_value = 0.45
    
    # Obtain the current distance to the goal from the ant's position
    current_goal_distance = self.goal_distance(ant_obs)

    # Calculate the proximity of the current goal distance to the desired goal distance
    # Use an L2 norm to penalize the abs difference between current and desired goal distance
    distance_to_goal_reward_component = -np.linalg.norm(current_goal_distance - goal_distance_value)
    
    # Weighting parameter for the distance to goal reward component
    distance_to_goal_weight = 1.0

    # Compute the total reward by combining the reward components
    reward = distance_to_goal_weight * distance_to_goal_reward_component
    
    # Create a dictionary of each individual reward component
    reward_components = {
        'distance_to_goal_reward': distance_to_goal_reward_component,
    }

    return reward, reward_components
```