Task 3 Name: Reduce Distance to Goal

Task 3 Description: Minimize [distance between Block xyz position and Desired goal position in xyz coordinate]. This task focuses on the core objective of the original task - moving the block towards the goal. By minimizing the distance between the block and the goal, the agent learns to push the block in the correct direction.

```python
def reduce_distance_to_goal_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Fetch the block and goal positions
    block_pos = self.block_position()
    goal_pos = self.goal_position()
    
    # Calculate the L2 distance between the block position and the goal position
    distance_to_goal = np.linalg.norm(block_pos - goal_pos)
    
    # Minimizing the distance to goal, hence the negative sign
    # Utilize a transformation function to smooth the reward as the distance decreases
    reward = -np.tanh(distance_to_goal)
    
    # Detailed reward components for potential debugging/analysis
    reward_components = {
        'distance_to_goal': distance_to_goal,
        'raw_reward': -distance_to_goal,
        'transformed_reward': reward
    }

    return reward, reward_components
```