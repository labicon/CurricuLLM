Task 1 Name: Align End Effector with Block
Task 1 Description: Make [End effector xyz position] as [Block xyz position]. This task will help the agent learn how to maneuver the end effector to the exact position of the block, which is a critical skill for pushing the block effectively towards the goal.

```python
def align_end_effector_with_block(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    block_pos = self.block_position()
    end_effector_pos = self.end_effector_position()

    # Calculate the distance between end effector and block
    distance = np.linalg.norm(end_effector_pos - block_pos)

    # Reward for minimizing the distance
    # Introduce a weighting parameter for the distance reward
    distance_weight = 1.0

    # Use negative exponentially scaled distance to encourage proximity
    reward = -distance_weight * np.exp(distance)

    # components dictionary for debugging or analysis
    components = {
        'distance': distance,
        'reward': reward
    }

    return reward, components
```