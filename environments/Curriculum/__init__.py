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
     id="Curriculum/FetchPush-v0",
     entry_point="Curriculum.envs.FetchPush_v0:MujocoFetchPushEnv",
     max_episode_steps=50,
)

register(
     id="Curriculum/FetchPush-v1",
     entry_point="Curriculum.envs.FetchPush_v1:MujocoFetchPushEnv",
     max_episode_steps=50,
)

register(
     id="Curriculum/FetchPush-v2",
     entry_point="Curriculum.envs.FetchPush_v2:MujocoFetchPushEnv",
     max_episode_steps=50,
)

register(
     id="Curriculum/FetchPush-v3",
     entry_point="Curriculum.envs.FetchPush_v3:MujocoFetchPushEnv",
     max_episode_steps=50,
)

register(
     id="Curriculum/FetchPush-v4",
     entry_point="Curriculum.envs.FetchPush_v4:MujocoFetchPushEnv",
     max_episode_steps=50,
)

register(
     id=f"AdroitHandRelocate-v1",
     entry_point="Curriculum.envs.AdrointHandRelocate_v0:AdroitHandRelocateEnv",
     max_episode_steps=200,
)

register(
     id=f"AdroitHandRelocate-v1",
     entry_point="Curriculum.envs.AdrointHandRelocate_v1:AdroitHandRelocateEnv",
     max_episode_steps=200,
)

register(
     id=f"AdroitHandRelocate-v2",
     entry_point="Curriculum.envs.AdrointHandRelocate_v2:AdroitHandRelocateEnv",
     max_episode_steps=200,
)