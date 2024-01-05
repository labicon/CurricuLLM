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

register(
     id="Curriculum/AntMaze_UMaze-v0",
     entry_point="Curriculum.envs.ant_maze_v0:AntMazeEnv",
     max_episode_steps=700,
)

register(
     id="Curriculum/AntMaze_UMaze-v1",
     entry_point="Curriculum.envs.ant_maze_v1:AntMazeEnv",
     max_episode_steps=700,
)

register(
     id="Curriculum/AntMaze_UMaze-v2",
     entry_point="Curriculum.envs.ant_maze_v2:AntMazeEnv",
     max_episode_steps=700,
)

register(
     id="Curriculum/AntMaze_UMaze-v3",
     entry_point="Curriculum.envs.ant_maze_v3:AntMazeEnv",
     max_episode_steps=700,
)

register(
     id="Curriculum/AntMaze_UMaze-v4",
     entry_point="Curriculum.envs.ant_maze_v4:AntMazeEnv",
     max_episode_steps=700,
)