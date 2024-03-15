Task 2
Name: Match End Effector Velocity with Block
Description: Make [End effector linear velocity in xyz direction] as [Block linear velocity in xyz direction relative to end effector].

```python
def match_end_effector_velocity_with_block(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Observing the End effector and block velocity
    end_effector_velocity = self.end_effector_linear_velocity()
    block_velocity_relative_to_end_effector = self.block_relative_linear_velocity()

    # Computing the difference in velocities to minimize it
    velocity_difference = np.linalg.norm(end_effector_velocity - block_velocity_relative_to_end_effector)

    # Reward is higher when the difference is smaller. We can use negative velocity_difference for this, but using -np.tanh() to constrain the value between -1 and 0
    reward_velocity_match = -np.tanh(velocity_difference)
    
    # Detailed rewards for inspection purposes
    detailed_rewards = {
        "velocity_match": reward_velocity_match,
    }

    total_reward = reward_velocity_match
    return total_reward, detailed_rewards
```