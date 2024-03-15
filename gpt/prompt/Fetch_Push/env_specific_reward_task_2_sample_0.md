Task 3 Name: Reduce Distance to Goal

Task 3 Description: Minimize [distance between Block xyz position and Desired goal position in xyz coordinate]. This task focuses on the core objective of the original task - moving the block towards the goal. By minimizing the distance between the block and the goal, the agent learns to push the block in the correct direction.

```python
def reduce_distance_to_goal_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    block_position = self.block_position()
    goal_position = self.goal_position()
    distance_to_goal = np.linalg.norm(block_position - goal_position)
    
    # Set a weighting factor to scale the importance of this component
    distance_weight = 1.0
    
    # Minimize the distance using -tanh to ensure the reward increases as distance decreases
    reward = -np.tanh(distance_weight * distance_to_goal)
    
    # It's helpful to break down the reward components for debugging and understanding the contributions 
    reward_components = {
        "distance_to_goal": -distance_to_goal  # Negative to indicate minimization
    }
    
    return reward, reward_components
```