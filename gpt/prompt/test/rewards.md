To devise reward functions for Task 2 where the goal is to maintain vertical jumping and landing without progressing in the x-direction, we need to utilize the given observation details. Specifically, the reward functions will focus on the z height of the torso (`observation[1]`), the change in x position (`observation[0]` before and `next_observation[0]` after), as well as the torques applied as actions and whether the hopper is in a healthy state.

Sample 1
```python
from typing import List, Tuple, Dict
import numpy as np

def compute_reward(observation: List, action: List, next_observation: List) -> Tuple[np.float64, Dict[str, np.float64]]:
    z_height = next_observation[1]
    change_in_x = abs(next_observation[0] - observation[0])
    healthy = 1 if next_observation[1] >= 0.7 else 0  # Using 0.7 as the threshold for healthy z height
    
    # Reward Components
    vertical_reward = np.clip(z_height, 0, 2)  # Encouraging vertical movement
    horizontal_penalty = -change_in_x  # Discouraging horizontal movement
    health_reward = 100 * healthy  # Big reward for being in a healthy state

    # Summing up components for the total reward
    reward = vertical_reward + horizontal_penalty + health_reward
    
    # Reward Components Dictionary
    reward_components = {
        'vertical_reward': vertical_reward,
        'horizontal_penalty': horizontal_penalty,
        'health_reward': health_reward
    }
    
    return np.float64(reward), reward_components
```

Sample 2
```python
def compute_reward(observation, action, next_observation) -> Tuple[np.float64, Dict[str, np.float64]]:
    z_height_reward_temperature = 0.2
    angle_reward_temperature = 0.2
    action_cost_temperature = 0.05
    stillness_reward_temperature = 0.1

    # Extract relevant observation components
    z_height = next_observation[1]
    torso_angle = next_observation[2]

    # Calculate reward components
    upright_posture_reward = np.exp(-z_height_reward_temperature * np.abs(1.25 - z_height)) * np.exp(-angle_reward_temperature * np.abs(0.0 - torso_angle))
    action_cost = -action_cost_temperature * np.sum(np.square(action))
    stillness_penalty = -stillness_reward_temperature * np.abs(observation[0] - next_observation[0])

    # Combine reward components
    total_reward = upright_posture_reward + action_cost + stillness_penalty

    # Individual reward components dict
    rewards = {
        'upright_posture_reward': upright_posture_reward,
        'action_cost': action_cost,
        'stillness_penalty': stillness_penalty
    }

    return total_reward, rewards
```

Sample 3
```python
def compute_reward(observation: List, action: List, next_observation: List) -> Tuple[np.float64, Dict[str, np.float64]]:
    z_height_reward = np.log1p(next_observation[1])  # Logarithmic reward for height to encourage gentle landings
    horizontal_penalty = -np.abs(observation[0])  # Linear penalty for horizontal position
    
    healthy = 1 if observation[2] >= -0.2 and observation[2] <= 0.2 else 0  # threshold for healthy angle
    angle_health_reward = 20 * healthy  # Reward for maintaining a healthy angle
    
    # Reward based on the use of minimal control
    control_cost = -np.sum(np.square(action))
    
    # Summing up components for the total reward
    reward = z_height_reward + horizontal_penalty + angle_health_reward + control_cost
    
    # Reward Components Dictionary
    reward_components = {
        'z_height_reward': z_height_reward,
        'horizontal_penalty': horizontal_penalty,
        'angle_health_reward': angle_health_reward,
        'control_cost': control_cost
    }

    return np.float64(reward), reward_components
```

Sample 4
```python
def compute_reward(observation, action, next_observation) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Parameters for binary rewards and costs
    action_cost_temperature = 0.1

    # Calculate binary rewards
    z_height = next_observation[1]
    torso_angle = next_observation[2]
    upright_posture_reward = float(0.7 < z_height < 2.0 and -0.2 < torso_angle < 0.2)
    action_cost = -action_cost_temperature * np.sum(np.square(action))

    # Total reward
    total_reward = upright_posture_reward + action_cost

    # Individual reward components dict
    rewards = {
        'upright_posture_reward': upright_posture_reward,
        'action_cost': action_cost
    }

    return total_reward, rewards
```

Sample 5
```python
def compute_reward(observation, action, next_observation) -> Tuple[np.float64, Dict[str, np.float64]]:
    posture_temperature = 0.05
    stillness_temperature = 0.03
    action_cost_temperature = 0.02

    # Extract variables
    z_height = next_observation[1]
    torso_angle = next_observation[2]
    x_velocity = next_observation[7]

    # Reward for maintaining posture
    posture_incentive = np.exp(-posture_temperature * np.abs(z_height - 1.25)) * np.exp(-posture_temperature * np.abs(torso_angle))

    # Incentive for minimal movement
    stillness_incentive = np.exp(-stillness_temperature * np.abs(x_velocity))

    # Cost of action
    action_cost = -action_cost_temperature * np.sum(np.square(action))

    # Overall reward
    total_reward = posture_incentive + stillness_incentive + action_cost

    # Breakdown of reward components
    rewards = {
        'posture_incentive': posture_incentive,
        'stillness_incentive': stillness_incentive,
        'action_cost': action_cost
    }

    return total_reward, rewards
```

