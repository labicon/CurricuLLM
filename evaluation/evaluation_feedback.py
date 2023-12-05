import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped

from utils.vec_monitor import CurriculumVecMonitor


def curriculum_evaluate_policy_feedback(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0] or is_vecenv_wrapped(env, CurriculumVecMonitor)

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards_main = []
    episode_rewards_task = []
    episode_rewards_dict = []
    episode_lengths = []
    episode_success = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards_main = np.zeros(n_envs)
    current_rewards_task = np.zeros(n_envs)
    current_rewards_dict = [None for _ in range(n_envs)]
    current_lengths = np.zeros(n_envs, dtype="int")
    current_success = np.zeros(n_envs, dtype="int")

    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards_main += np.array([info["reward_main"] for info in infos])
        current_rewards_task += np.array([info["reward_task"] for info in infos])
        current_success += np.array([info["success"] for info in infos]).astype(int)

        for env_idx, info in enumerate(infos):
            if current_rewards_dict[env_idx] is None:
                current_rewards_dict[env_idx] = {key: 0 for key in info["reward_dict"].keys()}
            current_rewards_dict[env_idx] = {key: current_rewards_dict[env_idx][key] + info["reward_dict"][key] for key in info["reward_dict"].keys()}

        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        raise NotImplementedError("Evaluation with Monitor wrapper is not supported yet.")
                    else:
                        episode_rewards_main.append(current_rewards_main[i])
                        episode_rewards_task.append(current_rewards_task[i])
                        episode_rewards_dict.append(current_rewards_dict[i])
                        episode_lengths.append(current_lengths[i])
                        episode_success.append(current_success[i]/current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards_main[i] = 0
                    current_rewards_task[i] = 0
                    current_rewards_dict[i] = None
                    current_success[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    # episode_ variable is a list of float, int(for success), or dictionary(for reward_dict)
    mean_reward_main = np.mean(episode_rewards_main)
    std_reward_main = np.std(episode_rewards_main)
    mean_reward_task = np.mean(episode_rewards_task)
    std_reward_task = np.std(episode_rewards_task)
    if reward_threshold is not None:
        assert mean_reward_task > reward_threshold, "Mean reward task below threshold: " f"{mean_reward_task:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards_main, episode_rewards_task, episode_rewards_dict, episode_lengths, episode_success
    return mean_reward_main, std_reward_main, mean_reward_task, std_reward_task
