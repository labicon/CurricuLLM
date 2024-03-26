Task 5 Name
Original Task

Task 5 Description
The objective is to maintain `goal_distance` as close as possible to the target value of `0.45`, where `goal_distance` is the Euclidean distance between the ant's torso and the goal position in the maze.

```python
def original_task_reward(self, ant_obs) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Define the target value for goal_distance
    target_goal_distance = 0.45
    
    # Retrieve the current goal distance
    current_goal_distance = self.goal_distance(ant_obs)
    
    # Define a weighting parameter for the goal distance reward component
    goal_distance_weight = 1.0
    
    # Compute the L2 norm difference between current and target goal distances
    goal_distance_error = np.linalg.norm(current_goal_distance - target_goal_distance)
  
    # Convert the error into a reward, where a smaller distance error gives a higher reward
    # Since the function should return a positive reward for closer distances, we use a negative sign.
    goal_distance_reward = -goal_distance_weight * goal_distance_error

    # Total reward is just the goal distance reward in this case
    reward = goal_distance_reward
    
    # Create a dictionary of individual reward components
    reward_components = {
        "goal_distance_reward": goal_distance_reward
    }
    
    return reward, reward_components
```