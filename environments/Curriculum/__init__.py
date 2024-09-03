from gymnasium.envs.registration import register

# register(
#      id="Curriculum/Walker2d-v0",
#      entry_point="Curriculum.envs.Walker2d_v0:Walker2dEnv",
#      max_episode_steps=1000,
# )

# register(
#      id="Curriculum/Walker2d-v1",
#      entry_point="Curriculum.envs.Walker2d_v1:Walker2dEnv",
#      max_episode_steps=1000,
# )

# register(
#      id="Curriculum/Walker2d-v2",
#      entry_point="Curriculum.envs.Walker2d_v2:Walker2dEnv",
#      max_episode_steps=1000,
# )

register(
     id="Curriculum/AntMaze_UMaze",
     entry_point="Curriculum.envs.AntMaze_UMaze:AntMazeEnv",
     max_episode_steps=700,
)

register(
     id="Curriculum/AntMaze_UMaze_play",
     entry_point="Curriculum.envs.AntMaze_UMaze_play:AntMazeEnv",
     max_episode_steps=700,
)

register(
     id="Curriculum/FetchPush",
     entry_point="Curriculum.envs.FetchPush:MujocoFetchPushEnv",
     max_episode_steps=50,
)

# register(
#      id="Curriculum/FetchPush-v0",
#      entry_point="Curriculum.envs.FetchPush_v0:MujocoFetchPushEnv",
#      max_episode_steps=50,
# )

# register(
#      id="Curriculum/FetchPush-v1",
#      entry_point="Curriculum.envs.FetchPush_v1:MujocoFetchPushEnv",
#      max_episode_steps=50,
# )

# register(
#      id="Curriculum/FetchPush-v2",
#      entry_point="Curriculum.envs.FetchPush_v2:MujocoFetchPushEnv",
#      max_episode_steps=50,
# )

register(
     id="Curriculum/FetchPickAndPlace",
     entry_point="Curriculum.envs.FetchPickAndPlace:MujocoFetchPickAndPlaceEnv",
     max_episode_steps=50,
)

register(
     id="Curriculum/FetchSlide",
     entry_point="Curriculum.envs.FetchSlide:MujocoFetchSlideEnv",
     max_episode_steps=50,
)