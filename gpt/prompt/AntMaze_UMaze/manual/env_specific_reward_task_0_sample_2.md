Task 1 Name
Basic Locomotion

Task 1 Description
The objective of the task is to maximize the velocity of the torso of the ant to enable it to move quickly through the environment.

```python
def basic_locomotion_reward(self, ant_obs: np.ndarray) -> Tuple[np.float64, Dict[str, np.float64]]:
    # reward components
    reward_velocity = 0.0
    rewards_dict = {}

    # parameters
    velocity_weight = 1.0  # Adjust this parameter to tune the importance of velocity in the reward function

    # calculate the torso velocity magnitude
    torso_vel = self.torso_velocity(ant_obs)
    torso_vel_magnitude = np.linalg.norm(torso_vel)

    # we use tanh to bound the velocity in [-1,1] range for reward shaping purposes
    reward_velocity = np.tanh(torso_vel_magnitude)
    
    # total reward
    reward = velocity_weight * reward_velocity

    # populate rewards dictionary
    rewards_dict['velocity'] = reward_velocity

    return reward, rewards_dict
```

This reward function captures the essence of the 'Basic Locomotion' task. By using the hyperbolic tangent of the velocity magnitude, we ensure that the reward scales well within the range of [-1, 1] without experiencing too much saturation. The `velocity_weight` parameter can be adjusted to give more or less importance to the torso velocity when calculating the total reward.