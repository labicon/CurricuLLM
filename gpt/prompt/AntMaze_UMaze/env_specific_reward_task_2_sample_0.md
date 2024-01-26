Task 3 Name
Goal Orientation

Task 3 Description
The objective of this task is to achieve an orientation that minimizes the distance to the goal position. The reward encourages the ant to align its torso orientation to face the direction of the goal. Hence, this task aims at guiding the ant's movements to be more goal-directed by considering its orientation in relation to the target goal within the maze.

```python
def compute_goal_orientation_reward(self, ant_obs,) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Define weight for orientation reward component
    orientation_weight = 0.5
    
    # Calculate the distance to the goal
    distance_to_goal = self.goal_distance(ant_obs)
    
    # Compute the orientation reward component using tanh to encourage the ant to face towards the goal
    orientation_reward = -np.tanh(distance_to_goal)

    # Multiply by weight
    orientation_reward *= orientation_weight

    # Sum up the total reward
    total_reward = orientation_reward
    
    # Create a dictionary for individual reward components
    reward_components = {
        'orientation_reward': orientation_reward
    }
    
    return total_reward, reward_components
```