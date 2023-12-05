Task 1 Name
Learn to Stand

Task 1 Description
The first task for the agent is to learn how to balance and stand upright for an extended period of time without taking any steps. The agent should apply minimal torques to maintain an upright position (z height approximately 1.25) and keep the angles of the torso, thigh, leg, and foot close to zero to avoid falling over. Success is evaluated based on how long the agent can stand without exceeding the min and max angle constraints and maintaining a healthy z range.

```python
def compute_reward_standing(observation, action, next_observation) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Initialize reward and components dictionary
    reward = 0
    reward_components = {}
    
    # Define temperature parameters for normalizing rewards
    angle_temp = 0.1
    z_temp = 0.5
    control_temp = 0.01
    
    # Extract important variables from the observation
    z_height = next_observation[1]
    angle_of_torso = next_observation[2]
    
    # Control cost for applying torques
    control_cost = 0.1 * np.sum(np.square(action))
    reward_components['control_cost'] = -control_temp * control_cost

    # Reward for maintaining a healthy z and minimal torso angle
    desired_z_height = 1.25
    z_reward = -np.abs(z_height - desired_z_height)
    angle_reward = -np.abs(angle_of_torso)

    # Normalize the z and angle rewards
    reward_components['z_height'] = z_temp * z_reward
    reward_components['angle_of_torso'] = angle_temp * angle_reward

    # Calculate total reward
    reward = sum(reward_components.values())

    return reward, reward_components
```

Task 2 Name
Learn to Squat and Rise

Task 2 Description
The second task is to learn a controlled squat and rise motion by flexing and extending the thigh, leg, and foot joints without hopping or moving horizontally. The agent must bend the joints to lower its center of mass and then extend them to rise back to the standing position. Evaluation is based on maintaining balance and smoothness of the squat and rise motion within the healthy range of states and without exceeding the boundaries for angles or z coordinates.

```python
def compute_reward_squat_rise(observation, action, next_observation) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Initialize reward and components dictionary
    reward = 0
    reward_components = {}
    
    # Define temperature parameters for normalizing rewards
    z_vel_temp = 0.05
    control_temp = 0.01
    joint_angles_temp = 0.1
    
    # Extract important variables from the observation
    z_vel = (next_observation[1] - observation[1]) / 0.01  # Assuming known timestep dt = 0.01
    
    # Control cost for applying torques
    control_cost = 0.1 * np.sum(np.square(action))
    reward_components['control_cost'] = -control_temp * control_cost

    # Reward for smoothness - small vertical velocity during squat and rise
    z_vel_reward = -np.square(z_vel)
    reward_components['z_velocity'] = z_vel_temp * z_vel_reward

    # Reward for maintaining balance - joint angles should be close to zero
    joint_angles = np.abs(next_observation[3:6])
    joint_angle_reward = -np.sum(joint_angles)
    reward_components['joint_angles'] = joint_angles_temp * joint_angle_reward

    # Calculate total reward
    reward = sum(reward_components.values())

    return reward, reward_components
```

Task 3 Name
Learn to Hop in Place

Task 3 Description
Once the agent has mastered standing and squat-rising, it now has to learn how to hop vertically without forward movement. This practice will help the agent develop a sense of how to generate lift and use torques effectively. Success is evaluated by the vertical height achieved and the ability to land back on the foot, maintaining balance, and readiness for subsequent hops.

```python
def compute_reward_hop_in_place(observation, action, next_observation) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Initialize reward and components dictionary
    reward = 0
    reward_components = {}
    
    # Define temperature parameters for normalizing rewards
    z_height_temp = 0.5
    control_temp = 0.01
    stationary_x_vel_temp = 0.1
    
    # Extract important variables from observations
    z_height = next_observation[1]
    x_velocity = (next_observation[0] - observation[0]) / 0.01  # Assuming known timestep dt = 0.01

    # Control cost for applying torques
    control_cost = 0.1 * np.sum(np.square(action))
    reward_components['control_cost'] = -control_temp * control_cost

    # Reward height achieved during hop
    reward_components['z_height'] = z_height_temp * z_height

    # Penalty for horizontal movement, encourage in-place hopping
    reward_components['x_velocity'] = -stationary_x_vel_temp * np.square(x_velocity)

    # Calculate total reward
    reward = sum(reward_components.values())

    return reward, reward_components
```

Task 4 Original Task
Learn to Hop Forward

Task 4 Original Task Description
Building on the previous tasks, the original task is to apply torques to make consecutive hops that move the hopper in the forward (right) direction. The agent has to find the right timing and magnitude of torques to apply to the thigh, leg, and foot joints to propel itself forward while also maintaining an upright, healthy posture. It requires coordination of balance and propulsion learned in previous tasks. Success is evaluated based on the forward distance covered (x_velocity), energy efficiency (control_cost), and the agent's ability to stay within the healthy state.

```python
def compute_reward_hop_forward(observation, action, next_observation) -> Tuple[np.float64, Dict[str, np.float64]]:
    # Initialize reward and components dictionary
    reward = 0
    reward_components = {}
    
    # Extract important variables from observations
    x_position_before = observation[0]
    x_position_after = next_observation[0]
    x_velocity = (x_position_after - x_position_before) / 0.01  # Assuming known timestep dt = 0.01
    
    # Control cost for applying torques
    control_cost = 0.1 * np.sum(np.square(action))
    reward_components['control_cost'] = -control_cost

    # Check if the hopper is in a healthy state to provide a bonus reward
    is_healthy = HopperEnv.is_healthy(next_observation)
    reward_components['healthy_reward'] = is_healthy

    # Reward for forward movement
    reward_components['x_velocity'] = x_velocity

    # Calculate total reward
    reward = sum(reward_components.values())

    return reward, reward_components
```