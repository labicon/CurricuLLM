from gymnasium.envs.registration import register

register(
     id="Curriculum/Walker2d-v0",
     entry_point="Curriculum.envs.Walker2d_v0:Walker2dEnv",
     max_episode_steps=1000,
)

register(
     id="Curriculum/Walker2d-v1",
     entry_point="Curriculum.envs.Walker2d_v1:Walker2dEnv",
     max_episode_steps=1000,
)

register(
     id="Curriculum/Walker2d-v2",
     entry_point="Curriculum.envs.Walker2d_v2:Walker2dEnv",
     max_episode_steps=1000,
)

register(
     id="Curriculum/AntMaze_UMaze",
     entry_point="Curriculum.envs.AntMaze_UMaze:AntMazeEnv",
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
     id="Curriculum/FetchPickAndPlace-v0",
     entry_point="Curriculum.envs.FetchPickAndPlace_v0:MujocoFetchPickAndPlaceEnv",
     max_episode_steps=50,
)

register(
     id="Curriculum/FetchPickAndPlace-v1",
     entry_point="Curriculum.envs.FetchPickAndPlace_v1:MujocoFetchPickAndPlaceEnv",
     max_episode_steps=50,
)

register(
     id="Curriculum/FetchPickAndPlace-v2",
     entry_point="Curriculum.envs.FetchPickAndPlace_v2:MujocoFetchPickAndPlaceEnv",
     max_episode_steps=50,
)

register(
     id=f"Curriculum/AdroitHandRelocate-v0",
     entry_point="Curriculum.envs.AdroitHandRelocate_v0:AdroitHandRelocateEnv",
     max_episode_steps=200,
)

register(
     id=f"Curriculum/AdroitHandRelocate-v1",
     entry_point="Curriculum.envs.AdroitHandRelocate_v1:AdroitHandRelocateEnv",
     max_episode_steps=200,
)

register(
     id=f"Curriculum/AdroitHandRelocate-v2",
     entry_point="Curriculum.envs.AdroitHandRelocate_v2:AdroitHandRelocateEnv",
     max_episode_steps=200,
)