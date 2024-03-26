def compute_reward_curriculum(self):
    ant_obs = self.get_ant_obs()
    torso_vel = self.torso_velocity(ant_obs)
    
    # Reward components
    velocity_reward_weight = 1.0
    
    # Reward calculations
    velocity_reward = np.tanh(np.linalg.norm(torso_vel)) # Maximize torso velocity
    
    # Total reward calculation
    total_reward = velocity_reward_weight * velocity_reward
    
    # Reward dictionary
    reward_dict = {
        'velocity_reward': velocity_reward
    }
    
    return total_reward, reward_dict