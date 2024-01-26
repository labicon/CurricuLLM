Task 5 Name
Original Task

Task 5 Description
The goal of the reward function for the original task in the AntMazeEnv environment is to design a reward function that encourages the ant to maintain the goal_distance at a value of 0.45. The reward should penalize deviations from this desired distance.

```python
def original_task_reward(self, ant_obs) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Initialize reward dictionary.
    reward_components: Dict[str, np.float64] = {}
    
    # Define the weight for the goal_distance component.
    goal_distance_weight: np.float64 = 1.0
    
    # Calculate the distance to the goal position.
    goal_distance = self.goal_distance(ant_obs)
    
    # Use L2 norm to create a reward component based on the goal_distance.
    # The further from 0.45, the more negative the reward will be.
    goal_distance_reward: np.float64 = -goal_distance_weight * np.linalg.norm(goal_distance - 0.45)
    
    # Store the reward components in the dictionary.
    reward_components["goal_distance_reward"] = goal_distance_reward
    
    # Compute the total reward.
    reward = np.sum(list(reward_components.values()))
    
    return reward, reward_components
```