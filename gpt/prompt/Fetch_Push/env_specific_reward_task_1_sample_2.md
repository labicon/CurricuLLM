Task 2 Name: Match End Effector Velocity with Block

Description: Make [End effector linear velocity in xyz direction] as [Block linear velocity in xyz direction relative to end effector]

Reason: Ensuring the end effector moves at the same velocity as the block (initially zero) will teach the agent the precision needed in controlling its speed and direction, important for gentle and accurate pushes.

```python
def match_end_effector_velocity_with_block(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    end_effector_velocity = self.end_effector_linear_velocity()
    block_relative_velocity = self.block_relative_linear_velocity()
    # Calculating the difference in velocity between the end effector and the block
    velocity_diff = end_effector_velocity - block_relative_velocity
    # Using L2 norm to measure the "distance" of this difference to zero. The goal is to minimize this difference.
    velocity_matching_error = np.linalg.norm(velocity_diff)
    # Reward is negatively proportional to the error (we want to minimize the error)
    reward_velocity_matching = -velocity_matching_error
    # Assigning a weight to this component of the reward to scale its effect
    weight_velocity_matching = 1.0  # Adjust this weight to prioritize this task properly
    total_reward = weight_velocity_matching * reward_velocity_matching
    
    reward_components = {
        "velocity_matching_error": velocity_matching_error,
        "reward_velocity_matching": reward_velocity_matching
    }

    return total_reward, reward_components
```