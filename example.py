import gymnasium as gym

env = gym.make("ALE/MontezumaRevenge-v5", render_mode='human')

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = 0
    observation, reward, terminated, truncated, info = env.step(action)

    ram = env.unwrapped.ale.getRAM()
    print(ram)

    if terminated or truncated:
        observation, info = env.reset()
env.close()