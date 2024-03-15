Task 3 Name: Reduce Distance to Goal

Task 3 Description: Minimize the distance between the block's xyz position and the desired goal position in xyz coordinates. This task focuses on the core objective of the original task - moving the block towards the goal. By minimizing the distance between the block and the goal, the agent learns to push the block in the correct direction.

```python
def reduce_distance_to_goal_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    block_pos = self.block_position()
    goal_pos = self.goal_position()
    
    # Compute L2 norm which is the Euclidean distance between block position and goal position
    distance_to_goal = np.linalg.norm(goal_pos - block_pos)
    
    # The smaller the distance, the higher the reward. Using negative sign to convert minimization to maximization problem
    reward = -distance_to_goal

    reward_components = {
        "distance_to_goal": distance_to_goal,  # Keep track of the distance to goal component
    }
    
    return reward, reward_components
```