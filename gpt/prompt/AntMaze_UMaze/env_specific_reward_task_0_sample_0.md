Task 1 Name
Basic Locomotion

Task 1 Description
Encourage the ant to maximize its forward velocity to achieve locomotion. We aim to increase the speed of the ant's torso in the direction of the maze's target goal. 

```python
def basic_locomotion_reward(self, ant_obs) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Get the torso velocity in the direction of movement (x-axis)
    torso_vel = self.torso_velocity(ant_obs)
    forward_velocity = torso_vel[0]  # Assuming that x-axis is the forward direction

    # Reward component based on the forward velocity of the torso (to encourage forward movement)
    velocity_reward_weight = 1.0
    velocity_reward = np.tanh(forward_velocity)
    
    reward = velocity_reward_weight * velocity_reward
    
    # Dictionary of individual components with their corresponding rewards
    reward_components = {
        'velocity_reward': velocity_reward
    }
    
    return reward, reward_components
```