Task 4 Name: Move Block to Target Position

Task 4 Description: The agent must manipulate a block to a target position on the table by pushing it with a gripper. The task is considered successful if the block's position is within 0.05 units of the target position. This requires precise control over the manipulator's movement and velocity, as well as strategic planning to push the block towards the goal efficiently.

```python
def original_task_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Extract necessary observations
    block_pos = self.block_position()
    goal_pos = self.goal_position()
    ee_pos = self.end_effector_position()
    block_vel = self.block_relative_linear_velocity()
    ee_vel = self.end_effector_linear_velocity()
    
    # Constants
    block_to_goal_distance_weight = 1.0
    ee_to_block_distance_weight = 0.5
    block_velocity_weight = 0.3
    ee_velocity_weight = 0.2
    
    # Calculating components of the reward function
    block_to_goal_distance = np.linalg.norm(block_pos - goal_pos)
    ee_to_block_distance = np.linalg.norm(ee_pos - block_pos)
    block_velocity_bonus = np.linalg.norm(block_vel)
    ee_velocity_penalty = np.linalg.norm(ee_vel)
    
    # Reward calculations
    block_to_goal_reward = -block_to_goal_distance_weight * block_to_goal_distance
    ee_to_block_reward = -ee_to_block_distance_weight * ee_to_block_distance
    block_velocity_reward = block_velocity_weight * np.tanh(block_velocity_bonus)
    ee_velocity_reward = -ee_velocity_weight * np.tanh(ee_velocity_penalty)
    
    # Total reward
    total_reward = block_to_goal_reward + ee_to_block_reward + block_velocity_reward + ee_velocity_reward
    
    # Reward components dictionary
    reward_components = {
        'block_to_goal_distance': block_to_goal_reward,
        'ee_to_block_distance': ee_to_block_reward,
        'block_velocity_bonus': block_velocity_reward,
        'ee_velocity_penalty': ee_velocity_reward
    }
    
    return np.float64(total_reward), reward_components
```