import numpy as np

def analyze_trajectory_ant(obs_trajectory, goal_trajectory):
    # obs_trajectory: list of observations
    # Get list of torso_coord, torso_orientation, torso_velocity, torso_angular_velocity, goal_pos, goal_distance
    torso_coord = []
    torso_orientation = []
    torso_velocity = []
    torso_angular_velocity = []
    goal_pos = []
    goal_distance = []

    for obs, goal in zip(obs_trajectory, goal_trajectory):
        torso_coord.append(obs[0:3])
        torso_orientation.append(obs[3:7])
        torso_velocity.append(obs[15:17])
        torso_angular_velocity.append(obs[18:21])
        goal_pos.append(goal)
        goal_distance.append(np.linalg.norm(obs[0:2] - goal))

    # change to np array
    torso_coord = np.array(torso_coord)
    torso_orientation = np.array(torso_orientation)
    torso_velocity = np.abs(np.array(torso_velocity))
    torso_angular_velocity = np.array(torso_angular_velocity)
    goal_pos = np.array(goal_pos)
    goal_distance = np.array(goal_distance)

    # Calculate mean and std of each variable
    statistics = {}
    statistics["torso_coord_mean"] = np.mean(torso_coord, axis=0).round(2)
    statistics["torso_coord_std"] = np.std(torso_coord, axis=0).round(2)
    statistics["torso_orientation_mean"] = np.mean(torso_orientation, axis=0).round(2)
    statistics["torso_orientation_std"] = np.std(torso_orientation, axis=0).round(2)
    statistics["torso_velocity_mean"] = np.mean(torso_velocity, axis=0).round(2)
    statistics["torso_velocity_std"] = np.std(torso_velocity, axis=0).round(2)
    statistics["torso_angular_velocity_mean"] = np.mean(torso_angular_velocity, axis=0).round(2)
    statistics["torso_angular_velocity_std"] = np.std(torso_angular_velocity, axis=0).round(2)
    statistics["goal_pos_mean"] = np.mean(goal_pos, axis=0).round(2)
    statistics["goal_pos_std"] = np.std(goal_pos, axis=0).round(2)
    statistics["goal_distance_mean"] = np.mean(goal_distance, axis=0).round(2)
    statistics["goal_distance_std"] = np.std(goal_distance, axis=0).round(2)

    return statistics

def analyze_trajectory_fetch(obs_trajectory, goal_trajectory):
    # obs_trajectory: list of observations
    # Get list of end effector position, block position, relative block linear velocity, end effector velocity, goal_pos, gosl_distance

    end_effector_pos = []
    block_pos = []
    gripper_distance = []
    block_relative_velocity = []
    end_effector_velocity = []
    goal_pos = []
    goal_distance = []

    for obs, goal in zip(obs_trajectory, goal_trajectory):
        end_effector_pos.append(obs[0:3])
        block_pos.append(obs[3:6])
        gripper_distance.append(abs(obs[9] - obs[10]))
        block_relative_velocity.append(obs[15:18])
        end_effector_velocity.append(obs[20:23])
        goal_pos.append(goal)
        goal_distance.append(np.linalg.norm(obs[3:6] - goal))

    # change to np array
    end_effector_pos = np.array(end_effector_pos)
    block_pos = np.array(block_pos)
    gripper_distance = np.array(gripper_distance)
    block_velocity = np.array(block_relative_velocity) + np.array(end_effector_velocity)
    end_effector_velocity = np.array(end_effector_velocity)
    goal_pos = np.array(goal_pos)
    goal_distance = np.array(goal_distance)

    # Calculate mean and std of each variable
    statistics = {}
    statistics["end_effector_pos_mean"] = np.mean(end_effector_pos, axis=0).round(2)
    statistics["end_effector_pos_std"] = np.std(end_effector_pos, axis=0).round(2)
    statistics["block_pos_mean"] = np.mean(block_pos, axis=0).round(2)
    statistics["block_pos_std"] = np.std(block_pos, axis=0).round(2)
    statistics["gripper_distance_mean"] = np.mean(gripper_distance, axis=0).round(2)
    statistics["gripper_distance_std"] = np.std(gripper_distance, axis=0).round(2)
    statistics["block_velocity_mean"] = np.mean(block_velocity, axis=0)
    statistics["block_velocity_std"] = np.std(block_velocity, axis=0)
    statistics["end_effector_velocity_mean"] = np.mean(end_effector_velocity, axis=0)
    statistics["end_effector_velocity_std"] = np.std(end_effector_velocity, axis=0)
    statistics["goal_pos_mean"] = np.mean(goal_pos, axis=0).round(2)
    statistics["goal_pos_std"] = np.std(goal_pos, axis=0).round(2)
    statistics["goal_distance_mean"] = np.mean(goal_distance, axis=0).round(2)
    statistics["goal_distance_std"] = np.std(goal_distance, axis=0).round(2)

    return statistics