Task 4 Name: Move Block to Target Position

Task 4 Description: The agent must manipulate a block to a target position on the table by pushing it with a gripper. The task is considered successful if the block's position is within 0.05 units of the target position. This requires precise control over the manipulator's movement and velocity, as well as strategic planning to push the block towards the goal efficiently.

```python
def move_block_to_target_position_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Extract necessary observations
    block_pos = self.block_position()
    goal_pos = self.goal_position()
    block_vel = self.block_relative_linear_velocity()
    end_effector_pos = self.end_effector_position()
    end_effector_vel = self.end_effector_linear_velocity()
    
    # Weights for the different components of the reward
    distance_to_goal_weight = 10.0  # Encourages block to move closer to the goal
    block_stability_weight = -5.0  # Discourages the block from moving too fast
    manipulator_efficiency_weight = -2.0  # Discourages excessive movement of the manipulator
    
    # Reward components
    distance_to_goal = np.linalg.norm(block_pos[:2] - goal_pos[:2])  # Consider only x, y
    block_stability = np.linalg.norm(block_vel)
    manipulator_efficiency = np.linalg.norm(end_effector_vel)
    
    # Compose final reward
    total_reward = (distance_to_goal_weight * -np.tanh(distance_to_goal) +
                    block_stability_weight * -np.tanh(block_stability) +
                    manipulator_efficiency_weight * -np.tanh(manipulator_efficiency))
    
    # Components for debugging or analysis
    components = {
        "distance_to_goal": -np.tanh(distance_to_goal),
        "block_stability": -np.tanh(block_stability),
        "manipulator_efficiency": -np.tanh(manipulator_efficiency)
    }
    
    return total_reward, components
```