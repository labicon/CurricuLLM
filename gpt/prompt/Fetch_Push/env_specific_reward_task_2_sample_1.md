Task 3 Name: Reduce Distance to Goal
Task 3 Description: Minimize [distance between Block xyz position and Desired goal position in xyz coordinate]. This task focuses on the core objective of the original task - moving the block towards the goal. By minimizing the distance between the block and the goal, the agent learns to push the block in the correct direction.

```python
def reduce_distance_to_goal_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    block_pos = self.block_position()
    goal_pos = self.goal_position()

    # Compute the L2 norm (Euclidean distance) between block position and goal position
    distance_to_goal = np.linalg.norm(block_pos - goal_pos)

    # Since we want to minimize the distance, use negative distance as reward
    reward = -distance_to_goal

    # For detailed analysis, track distance to goal as a separate component
    reward_components = {"distance_to_goal": distance_to_goal}

    return reward, reward_components
```