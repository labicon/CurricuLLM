Task 2 Name: Match End Effector Velocity with Block

Task 2 Description: Make [End effector linear velocity in xyz direction] as [Block linear velocity in xyz direction relative to end effector]. Ensuring the end effector moves at the same velocity as the block (initially zero) will teach the agent the precision needed in controlling its speed and direction, important for gentle and accurate pushes.

```python
def match_end_effector_velocity_with_block(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    end_effector_velocity = self.end_effector_linear_velocity()
    block_relative_velocity = self.block_relative_linear_velocity()
    
    # Calculate the velocity difference in xyz direction
    velocity_difference = end_effector_velocity - block_relative_velocity
    
    # Goal is to minimize the difference hence we use the L2 norm
    velocity_match_reward = -np.linalg.norm(velocity_difference)
    
    reward = velocity_match_reward
    reward_components = {
        'velocity_match_reward': velocity_match_reward
    }

    return reward, reward_components
```