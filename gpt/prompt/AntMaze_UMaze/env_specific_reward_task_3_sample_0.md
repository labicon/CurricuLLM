Task 4 Name
Navigation and Turning

Task 4 Description
Maximize the ant robot's forward position (x-coordinate of torso_coordinate) while adjusting the torso's orientation to face multiple random goals (goal_pos), without actually reaching the goals.

```python
def navigation_turning_reward(self, ant_obs) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Collect necessary observations
    x_coordinate = self.torso_coordinate(ant_obs)[0]
    goal_direction = self.goal_pos() - self.torso_coordinate(ant_obs)[:2]
    goal_angle = np.arctan2(goal_direction[1], goal_direction[0])

    # Components of the reward
    forward_reward_weight = 1.0
    turning_reward_weight = 0.5

    # Calculate the reward for forward movement along the x-axis
    forward_reward = np.tanh(x_coordinate)
    # Apply an L2 norm to the difference in angle to face the goal
    current_orientation = self.torso_orientation(ant_obs)
    current_heading_angle = np.arctan2(current_orientation[1], current_orientation[0])
    turning_reward = -np.tanh(np.linalg.norm(current_heading_angle - goal_angle))

    # Aggregate the reward with their respective weights
    reward = (forward_reward_weight * forward_reward) + (turning_reward_weight * turning_reward)
    reward_components = {
        "forward_reward": forward_reward,
        "turning_reward": turning_reward
    }

    return reward, reward_components
```

This reward function drives the ant to increase its x-coordinate, which means moving forward in the maze, while also turning to face the intended random goal directions. The use of the hyperbolic tangent function helps keep rewards bounded within a sensible range for each respective task, and weights allow the user to balance how much each component contributes to the total reward.