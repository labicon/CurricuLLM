Task 1 Name
Basic Locomotion

Task 1 Description
The objective is to maximize the torso velocity of the ant robot within the given 3D environment, encouraging it to move as fast as possible without any other specific directional requirement.

```python
def basic_locomotion_reward(self, ant_obs: np.ndarray) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Component weightings
    velocity_weight = 1.0

    # Get the torso velocity vector
    torso_vel = self.torso_velocity(ant_obs)

    # Compute the magnitude of the torso velocity which we want to maximize
    torso_vel_magnitude = np.linalg.norm(torso_vel)

    # Use a transformation function that grows as the velocity magnitude increases
    # Tangent hyperbolic function bounds the reward between -1 and 1 for stability
    reward_velocity = velocity_weight * np.tanh(torso_vel_magnitude)

    # Total reward is the sum of the individual reward components
    reward = reward_velocity

    # Dictionary for individual reward components, if required for debugging or analysis
    reward_components = {
        "reward_velocity": reward_velocity
    }

    return reward, reward_components
```