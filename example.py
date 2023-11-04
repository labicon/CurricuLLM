import gymnasium as gym
import Curriculum

env = gym.make("Curriculum/HalfCheetah-v4", task="enhanced_stability")

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    print(reward)
    print(info)

    if terminated or truncated:
        observation, info = env.reset()
env.close()