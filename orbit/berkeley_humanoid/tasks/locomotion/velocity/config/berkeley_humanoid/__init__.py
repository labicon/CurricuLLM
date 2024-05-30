import gymnasium as gym

from . import agents, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Velocity-Rough-Berkeley-Humanoid-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.BerkeleyHumanoidRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BerkeleyHumanoidRoughPPORunnerCfg,
    },
)

gym.register(
    id="Velocity-Rough-Berkeley-Humanoid-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.BerkeleyHumanoidRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BerkeleyHumanoidRoughPPORunnerCfg,
    },
)
