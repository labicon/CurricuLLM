import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Velocity-Flat-Berkeley-Humanoid-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.BerkeleyHumanoidFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BerkeleyHumanoidFlatPPORunnerCfg,
    },
)

gym.register(
    id="Velocity-Flat-Berkeley-Humanoid-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.BerkeleyHumanoidFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BerkeleyHumanoidFlatPPORunnerCfg,
    },
)

gym.register(
    id="Velocity-Rough-Berkeley-Humanoid-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.BerkeleyHumanoidRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BerkeleyHumanoidRoughPPORunnerCfg,
    },
)

gym.register(
    id="Velocity-Rough-Berkeley-Humanoid-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.BerkeleyHumanoidRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BerkeleyHumanoidRoughPPORunnerCfg,
    },
)
