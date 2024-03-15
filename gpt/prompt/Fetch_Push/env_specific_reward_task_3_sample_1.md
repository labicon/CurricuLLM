Task 4 Name: Move Block to Target Position

Task 4 Description: The agent must manipulate a block to a target position on the table by pushing it with a gripper. The task is considered successful if the block's position is within 0.05 units of the target position. This requires precise control over the manipulator's movement and velocity, as well as strategic planning to push the block towards the goal efficiently.

```python
def original_task_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Reward components
    distance_weight = 0.1  # Weight for the distance component
    velocity_weight = 0.05  # Weight for the block's relative velocity component
    end_effector_weight = 0.05  # Weight for the end effector's velocity component

    # Fetch required information from environment
    block_pos = self.block_position()
    goal_pos = self.goal_position()
    block_vel = self.block_relative_linear_velocity()
    end_effector_vel = self.end_effector_linear_velocity()

    # Compute the distance from block to goal
    distance_to_goal = np.linalg.norm(block_pos - goal_pos)

    # Compute L2 squared norm of the block's relative linear velocity to encourage movement
    block_velocity_penalty = -np.linalg.norm(block_vel)**2
    
    # Compute L2 squared norm of the end effector's linear velocity to reduce unnecessary movement
    end_effector_velocity_penalty = -np.linalg.norm(end_effector_vel)**2
    
    # Combine components to calculate total reward
    reward_components = {
        "distance_to_goal": -distance_weight * distance_to_goal,
        "block_velocity_penalty": velocity_weight * block_velocity_penalty,
        "end_effector_velocity_penalty": end_effector_weight * end_effector_velocity_penalty,
    }

    # Total reward is the sum of individual components
    total_reward = sum(reward_components.values())

    return total_reward, reward_components
```