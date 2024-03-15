Task 3 Name: Reduce Distance to Goal

Task 3 Description: Minimize [distance between Block xyz position and Desired goal position in xyz coordinate]. This task focuses on the core objective of the original task - moving the block towards the goal. By minimizing the distance between the block and the goal, the agent learns to push the block in the correct direction.

```python
def reduce_distance_to_goal_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Fetch Block and Goal positions
    block_pos = self.block_position()
    goal_pos = self.goal_position()
    
    # Compute the L2 norm (euclidean distance) between the block and goal positions
    dist_to_goal = np.linalg.norm(block_pos - goal_pos)
    
    # We want to minimize this distance, so we use negative of the tanh of distance
    # This will give higher reward as the distance decreases
    reduction_in_distance_weight = 1.0  # weight constant to adjust importance
    reward = -reduction_in_distance_weight * np.tanh(dist_to_goal)
    
    # Individual components of the reward for monitoring and debugging
    reward_components = {
        "distance_to_goal": dist_to_goal,
        "weighted_distance_reward": reward
    }

    return reward, reward_components
```