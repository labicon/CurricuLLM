from gymnasium.envs.registration import register

register(
     id="Curriculum/HalfCheetah-v4",
     entry_point="envs.curriculum_halfcheetah:HalfCheetahEnv",
     max_episode_steps=1000,
)