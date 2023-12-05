Task 1 Name
Understanding the Basics of Balance

Task 1 Description
The agent should learn to maintain the hopper in an upright position with zero velocity (not falling over) for an extended period of time. The starting position is set with the hopper's torso standing vertically with a slight randomized initial disturbance in position and velocity. The reward is positively proportional to the time maintaining balance without exceeding a predefined maximum angle deviation from vertical.

Task 2 Name
Stationary Hopping

Task 2 Description
The agent must learn to perform a single hop and land back to the starting position without tilting or falling over. The hopper must remain in place (x-coordinate should not change significantly) while only moving in the z direction. The reward is given for successfully performing a hop and returning to the balance position within a small margin of error for x-position and angles of body parts. 

Task 3 Name
Controlled Hopping

Task 3 Description
This task involves performing multiple hops in the same place, with the goal of achieving a rhythmic and consistent hopping pattern. The agent is tasked to control the applied torques to ensure that the hopper takes off and lands in the same x-coordinate while maintaining a healthy posture described by the is_healthy function. Rewards are provided for each successful hop and maintaining the upright position.

Task 4 Name
Directional Hopping

Task 4 Description
Now, the agent should learn to hop forward by applying torques that push the hopper in the right direction while still maintaining balance. The reward function should encourage forward motion with minimal sideways drift. The sequence of tasks should lead to developing an efficient gait that combines balance, power, and direction.

Task 5 Name
Efficient Forward Hopping

Task 5 Description
Here, the agent needs to optimize the hopping sequence for efficient forward travel. The agent must achieve higher forward velocity by improving the coordination of torques applied. The reward function should now heavily penalize inefficient energy usage and sideways movement, and provide bonuses for speed, as well as maintaining a healthy state as per the is_healthy function.

Task 6 Name
Maximize Distance within Time Limit

Task 6 Description
The goal for the agent here is to maximize the horizontal distance covered in a fixed time limit with a series of hops. The agent must apply the learnings from previous tasks to achieve the greatest distance possible before time runs out. The reward is based on the forward distance covered minus the control costs and the healthy state reward. There are penalties for falling over before the time limit.

Task n Original Task
Task n Original Task Description
The agent should use the torques at its joints to achieve hops that move in the forward (right) direction while maximizing the cumulative reward. This involves complex coordination of all the learnings from previous tasks to maintain balance, control, and efficiency, minimizing control costs and maintaining a healthy torso angle and height throughout the hopping sequence. The performance of the hopper is evaluated based on the distance covered in the forward direction, subtracting the costs of control and ensuring a healthy state is maintained across the entire task.