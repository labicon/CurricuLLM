Task 4 Name: Move Block to Target Position

Task 4 Description: The agent must manipulate a block to a target position on the table by pushing it with a gripper. The task is considered successful if the block's position is within 0.05 units of the target position. This requires precise control over the manipulator's movement and velocity, as well as strategic planning to push the block towards the goal efficiently.

```python
def original_task_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Define weight parameters
    distance_weight = 1.0
    velocity_weight = 0.1
    
    # Get positions and velocities
    block_pos = self.block_position()
    goal_pos = self.goal_position()
    block_vel = self.block_relative_linear_velocity()
    ee_vel = self.end_effector_linear_velocity()
    
    # Compute distance to goal (ignore z axis)
    distance_to_goal = np.linalg.norm(block_pos[:2] - goal_pos[:2])
    
    # Compute reward components
    distance_reward = -distance_weight * distance_to_goal  # Reward for moving block closer to goal
    
    # Also, we want to control the velocity at which the block is pushed towards the goal
    # Reward for keeping a slow and steady pace in the direction of the goal
    block_velocity_reward = -velocity_weight * np.linalg.norm(block_vel)
    
    # Encourage the end effector to move at a reasonable speed to ensure precise control
    ee_velocity_reward = -velocity_weight * np.linalg.norm(ee_vel)
    
    # Compute total reward
    total_reward = distance_reward + block_velocity_reward + ee_velocity_reward
    
    # Reward components dictionary
    reward_components = {
        "distance_reward": distance_reward,
        "block_velocity_reward": block_velocity_reward,
        "ee_velocity_reward": ee_velocity_reward
    }
    
    return total_reward, reward_components
```