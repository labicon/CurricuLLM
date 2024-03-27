from gymnasium.envs.registration import register

register(
     id="Curriculum/AntMaze_UMaze-v0",
     entry_point="Curriculum.envs.AntMaze_UMaze_v0:AntMazeEnv",
     max_episode_steps=700,
)

register(
     id="Curriculum/AntMaze_UMaze-v1",
     entry_point="Curriculum.envs.AntMaze_UMaze_v1:AntMazeEnv",
     max_episode_steps=700,
)

register(
     id="Curriculum/AntMaze_UMaze-v2",
     entry_point="Curriculum.envs.AntMaze_UMaze_v2:AntMazeEnv",
     max_episode_steps=700,
)

register(
     id="Curriculum/AntMaze_UMaze-v3",
     entry_point="Curriculum.envs.AntMaze_UMaze_v3:AntMazeEnv",
     max_episode_steps=700,
)

register(
     id="Curriculum/AntMaze_UMaze-v4",
     entry_point="Curriculum.envs.AntMaze_UMaze_v4:AntMazeEnv",
     max_episode_steps=700,
)

register(
     id="Curriculum/Fetch_Push-v0",
     entry_point="Curriculum.envs.Fetch_Push_v0:MujocoFetchPushEnv",
     max_episode_steps=50,
)

register(
     id="Curriculum/Fetch_Push-v1",
     entry_point="Curriculum.envs.Fetch_Push_v1:MujocoFetchPushEnv",
     max_episode_steps=50,
)

register(
     id="Curriculum/Fetch_Push-v2",
     entry_point="Curriculum.envs.Fetch_Push_v2:MujocoFetchPushEnv",
     max_episode_steps=50,
)

register(
     id="Curriculum/Fetch_Push-v3",
     entry_point="Curriculum.envs.Fetch_Push_v3:MujocoFetchPushEnv",
     max_episode_steps=50,
)

register(
     id="Curriculum/Fetch_Push-v4",
     entry_point="Curriculum.envs.Fetch_Push_v4:MujocoFetchPushEnv",
     max_episode_steps=50,
)