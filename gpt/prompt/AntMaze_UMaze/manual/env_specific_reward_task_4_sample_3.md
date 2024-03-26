Task 5 Name
Original Task

Task 5 Description
The goal is to maintain the position of the ant robot such that the distance to the goal remains constant at a value of 0.45 units away from the target position. This involves the ant performing actions that precisely position it at a set distance from the desired goal.

```python
def original_task_reward(self, ant_obs) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Define the specific goal distance that we want the ant to maintain
    desired_goal_distance = 0.45
    
    # Calculate the current distance to the goal
    current_goal_distance = self.goal_distance(ant_obs)
    
    # The reward is the negative L2 norm between the current and desired goal distance,
    # this enforces the agent to minimize the difference
    distance_error = np.linalg.norm(current_goal_distance - desired_goal_distance)
    
    # We want the distance_error to be zero, so we will reward the agent more the smaller the error is
    # A tanh transformation is used to smoothly penalize the agent as it deviates from the desired distance.
    # This ensures that the reward is bounded and has nice properties for optimization.
    distance_reward_component = -np.tanh(distance_error)
    
    # Define a weighting parameter for the distance error component
    distance_error_weight = 1.0
    
    # Calculate the total reward by applying the weighting to the distance reward component
    reward = distance_error_weight * distance_reward_component
    
    # Store reward information in a dictionary for debugging and analysis
    reward_info = {
        "distance_reward_component": distance_reward_component,
        "distance_error_weight": distance_error_weight
    }
    
    return reward, reward_info
```