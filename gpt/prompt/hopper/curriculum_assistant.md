To solve the hops in the forward direction, we can break down the sequence of tasks into a curriculum to gradually teach the hopper agent how to perform effectively. Here is a list of tasks, with corresponding reward functions, that will help the agent to learn the main task effectively.

### Task 1: Balance on One Foot
**Task 1 Description:** Teach the hopper to balance on its foot without taking any steps. The agent should strive to maintain a stable, upright position without falling.

```python
def compute_reward(observation, action, next_observation) -> Tuple[np.float64, Dict[str, np.float64]]:
    z_height_after = next_observation[1]
    
    # Reward for maintaining a certain height range
    z_healthy = 1 if 0.8 <= z_height_after <= 1.2 else 0
    healthy_reward_temp = 5.0  # temperature parameter for normalizing healthy reward
    healthy_reward = np.exp(z_healthy / healthy_reward_temp)
    
    control_cost = 0.1 * np.sum(np.square(action))
    
    reward = healthy_reward - control_cost
    
    return reward, {'healthy_reward': healthy_reward, 'control_cost': control_cost}
```

### Task 2: Simple Hop
**Task 2 Description:** Now that the agent can balance on one foot, teach it to perform a small hop while maintaining balance.

```python
def compute_reward(observation, action, next_observation) -> Tuple[np.float64, Dict[str, np.float64]]:
    z_height_before = observation[1]
    z_height_after = next_observation[1]
    z_velocity = (z_height_after - z_height_before) / 0.05  # Assuming a timestep of 0.05

    healthy_reward = 1 if observation[2] > -0.15 and observation[2] < 0.15 else 0

    # Reward for hopping
    hop_reward_temp = 2.0  # temperature parameter for normalizing hop reward
    hop_reward = np.exp(max(z_velocity, 0) / hop_reward_temp)
    
    control_cost = 0.1 * np.sum(np.square(action))
    
    reward = healthy_reward + hop_reward - control_cost
    
    return reward, {'healthy_reward': healthy_reward, 'hop_reward': hop_reward, 'control_cost': control_cost}
```

### Task 3: Forward Movement (No hopping)
**Task 3 Description:** Before combining hopping with forward motion, ensure that the agent can lean forward and move without hopping.

```python
def compute_reward(observation, action, next_observation) -> Tuple[np.float64, Dict[str, np.float64]]:
    x_position_before = observation[0]
    x_position_after = next_observation[0]
    x_velocity = (x_position_after - x_position_before) / 0.05  # Assuming a timestep of 0.05

    # Encourage forward movement while maintaining balance
    forward_reward_temp = 1.0  # temperature parameter for normalizing forward reward
    forward_reward = np.exp(x_velocity / forward_reward_temp)
    
    healthy_reward = 1 if observation[2] > -0.15 and observation[2] < 0.15 else 0
    
    control_cost = 0.1 * np.sum(np.square(action))
    
    reward = healthy_reward + forward_reward - control_cost
    
    return reward, {'healthy_reward': healthy_reward, 'forward_reward': forward_reward, 'control_cost': control_cost}
```

### Task 4: Hopping Forward
**Task 4 Description:** Combine the skills learned from the previous tasks to perform hops that progress in the forward direction.

```python
def compute_reward(observation, action, next_observation) -> Tuple[np.float64, Dict[str, np.float64]]:
    x_position_before = observation[0]
    x_position_after = next_observation[0]
    z_height_before = observation[1]
    z_height_after = next_observation[1]
    
    x_velocity = (x_position_after - x_position_before) / 0.05
    z_velocity = (z_height_after - z_height_before) / 0.05
    
    # Reward for moving forwards
    forward_reward_temp = 1.0
    forward_reward = np.exp(x_velocity / forward_reward_temp)
    
    # Reward for hopping
    hop_reward_temp = 2.0
    hop_reward = np.exp(max(z_velocity, 0) / hop_reward_temp)
    
    healthy_reward = 1 if observation[2] > -0.15 and observation[2] < 0.15 else 0
    
    control_cost = 0.1 * np.sum(np.square(action))
    
    reward = healthy_reward + forward_reward + hop_reward - control_cost
    
    return reward, {'healthy_reward': healthy_reward, 'forward_reward': forward_reward, 'hop_reward': hop_reward, 'control_cost': control_cost}
```

By proceeding through these tasks incrementally, the agent can learn to master each component of the main task—balancing, hopping, moving forward, and combining hopping with forward motion. Each component reinforces the agent’s ability to maintain stability while achieving the goal, and the reward functions provide clear signals for improvement.