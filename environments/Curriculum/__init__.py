from gymnasium.envs.registration import register

register(
     id="Curriculum/HalfCheetah-v4",
     entry_point="Curriculum.envs:HalfCheetahEnv",
     max_episode_steps=1000,
)

register(
     id="Curriculum/HalfCheetah_Feedback-v4",
     entry_point="Curriculum.envs:HalfCheetahEnvFeedback",
     max_episode_steps=1000,
)

register(
     id="Curriculum/Hopper-v5",
     entry_point="Curriculum.envs:HopperEnv",
     max_episode_steps=1000,
)