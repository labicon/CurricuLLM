Task 2 Name: Match End Effector Velocity with Block
Task 2 Description: Make [End effector linear velocity in xyz direction] as [Block linear velocity in xyz direction relative to end effector]. Ensuring the end effector moves at the same velocity as the block (initially zero) will teach the agent the precision needed in controlling its speed and direction, important for gentle and accurate pushes.

```python
def match_end_effector_with_block_velocity(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Get the current end effector and block velocities
    end_effector_velocity = self.end_effector_linear_velocity()
    block_velocity = self.block_relative_linear_velocity()

    # Calculate the difference in velocities between the end effector and the block
    velocity_difference = end_effector_velocity - block_velocity

    # Define a weight for the velocity matching component
    velocity_weight = 1.0

    # Use the L2 norm of the velocity difference as a measure of reward. The smaller the difference, the higher the reward.
    # We negate the result as we want to minimize the difference.
    velocity_matching_reward = -np.linalg.norm(velocity_difference) * velocity_weight

    # Total reward is just the velocity matching component in this case
    reward = velocity_matching_reward

    # Components dictionary
    reward_components = {
        "velocity_matching_reward": velocity_matching_reward,
    }

    return reward, reward_components
```