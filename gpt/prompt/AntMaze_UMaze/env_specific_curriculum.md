Task 1 Name
Basic Locomotion

Task 1 Description
Maximize the torso_velocity to achieve faster locomotion, while maintaining the torso_orientation close to the initial state.

Task 2 Name
Stabilized Movement

Task 2 Description
Minimize the torso_angular_velocity to ensure stability during movement. Maintain the velocity achieved in Task 1.

Task 3 Name
Goal-oriented Locomotion

Task 3 Description
Minimize the goal_distance, directing the agent towards the random goal_pos while maintaining the torso_orientation and torso_angular_velocity as stable as possible. Use the locomotion skills from Task 1 and stabilization from Task 2.

Task 4 Name
Maze Navigation

Task 4 Description
Navigate towards the goal_pos while avoiding walls and obstacles, keeping the torso_coordinate within the maze boundaries. Apply the locomotion and stabilizing skills acquired in previous tasks to handle the maze's complexity.

Task 5 Original Task
Effective Maze Solver

Task 5 Original Task Description
Maintain a goal_distance of 0.45, effectively reaching the target in the closed maze using the skills developed from all previous tasks.