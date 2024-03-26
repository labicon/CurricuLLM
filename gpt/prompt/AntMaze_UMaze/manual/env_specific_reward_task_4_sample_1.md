Task 5 Name
Original Task

Task 5 Description
The reward function should encourage the ant robot to maintain a goal distance of 0.45 units from the target position within the closed maze.

```python
def original_task_reward(self, ant_obs: np.ndarray) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Constants for reward terms
    goal_distance_target = 0.45
    distance_weight = 1.0
    
    # Compute the distance from the goal
    distance = self.goal_distance(ant_obs)
    
    # Reward component for maintaining the desired goal distance (0.45)
    distance_reward = -distance_weight * np.linalg.norm(distance - goal_distance_target)
    
    # Total reward is the sum of individual components
    reward = distance_reward
    
    # Populate reward components as a dictionary for debugging or analysis
    reward_components = {
        'distance_reward': distance_reward,
    }
    
    return reward, reward_components
```