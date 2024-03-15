Task 4 Name: Move Block to Target Position

Task 4 Description: The agent must manipulate a block to a target position on the table by pushing it with a gripper. The task is considered successful if the block's position is within 0.05 units of the target position. This requires precise control over the manipulator's movement and velocity, as well as strategic planning to push the block towards the goal efficiently.

```python
def original_task_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Constants for weighting different components of the reward function
    distance_weight = 1.0
    velocity_weight = 0.1
    
    # Fetch the positions and velocities from the environment
    block_pos = self.block_position()
    goal_pos = self.goal_position()
    block_vel = np.linalg.norm(self.block_relative_linear_velocity())
    grip_vel = np.linalg.norm(self.end_effector_linear_velocity())
    
    # Calculate the distance to the goal (Euclidean distance in xy-plane only, as z is constant)
    distance_to_goal = np.linalg.norm(block_pos[:2] - goal_pos[:2])
    # We want the block to stay still at the goal, not just reach it and continue moving
    velocity_penalty = block_vel
    
    # Reward components
    reward_distance_to_goal = -distance_weight * distance_to_goal  # Negative reward for being away from the goal
    reward_velocity_penalty = -velocity_weight * velocity_penalty  # Negative reward for block’s velocity
    
    # Total reward summation
    reward = reward_distance_to_goal + reward_velocity_penalty
    
    # Dictionary for debugging or analysis
    rewards = {
        "distance_to_goal": reward_distance_to_goal,
        "velocity_penalty": reward_velocity_penalty
    }
    
    return reward, rewards
```