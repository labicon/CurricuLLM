Task 3 Name
Goal Orientation

Task 3 Description
Minimize the distance between the ant's torso and the goal position within the maze environment, based on the provided observation space.

```python
def goal_orientation_reward(self, ant_obs) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Weights for reward components
    goal_distance_weight = 1.0  # You can adjust weights to balance reward components

    # Calculate the distance to the goal from the torso position
    distance_to_goal = self.goal_distance(ant_obs)
    
    # Utilize negative L2 norm to incentivize minimizing the distance to the goal
    # Closer to zero is better, so we'll take the negative to encourage smaller values
    reward_distance_to_goal = -goal_distance_weight * np.linalg.norm(distance_to_goal)

    # Total reward is a combination of components
    reward = reward_distance_to_goal
    
    # Include individual reward components for debugging and analysis
    reward_components = {
        'distance_to_goal': reward_distance_to_goal
    }
    
    return reward, reward_components
```