Task 1 Name
Basic Locomotion

Task 1 Description
The objective is to maximize the torso velocity of the ant robot in order to enhance its locomotion capabilities. Higher torso velocity implies better and faster movement through the environment.

```python
def basic_locomotion_reward(self, ant_obs: np.ndarray) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Define the weight for the reward terms
    velocity_weight = 1.0

    # Retrieve the torso's velocity
    torso_velocity_vector = self.torso_velocity(ant_obs)
    # Use the norm of the velocity for the reward
    torso_velocity = np.linalg.norm(torso_velocity_vector)  

    # Use tanh to keep the velocities in a reasonable range
    reward_velocity = np.tanh(torso_velocity)

    # Weight the velocity component of the reward
    total_reward = velocity_weight * reward_velocity

    # Return the total reward and the dictionary of individual reward components
    reward_components = {'velocity_reward': reward_velocity}

    return total_reward, reward_components
```

This reward function encourages the agent (the ant) to maximize its torso velocity. It uses the hyperbolic tangent function to keep the velocity values within a reasonable range and allows the reward to stay stable even when the velocity gets high. The velocity is weighted to alter its contribution to the total reward.