Task 1 Name: Align End Effector with Block

Task 1 Description: Make the end effector's xyz position the same as the block's xyz position. This task is fundamental for the robot to learn how to precisely maneuver its end effector to the exact position of the block, which is crucial for effectively pushing the block towards the goal.

```python
def align_end_effector_with_block_reward(self) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Fetch the current position of the end effector and the block
    end_effector_pos = self.end_effector_position()
    block_pos = self.block_position()
    
    # Calculate the L2 norm (euclidean distance) between the end effector and the block
    distance = np.linalg.norm(end_effector_pos - block_pos)
    
    # Use the negative distance as the reward to encourage reducing the distance
    # Between the end effector and the block
    reward = -distance
    
    return reward, {"distance_to_block": distance}
```