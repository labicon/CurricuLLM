Task 1 Name
Standing Balance

Task 1 Description
Maintain a healthy_reward of 1, minimize control_cost, maintain a z_coordinate_of_the_torso value of greater than 0.7, and minimize angular_velocity_of_the_torso_angle; this will teach the hopper to balance on its foot without falling over or using excessive actions.

Task 2 Name
Smooth Hopping

Task 2 Description
Maintain a healthy_reward of 1, minimize control_cost, maintain a z_velocity_of_the_torso that is not too high to indicate smooth hops, maintain an angle_of_the_torso within (-0.2, 0.2), and maximize x_velocity_of_the_torso; the goal is to encourage the agent to perform smooth, efficient hops forward without risking the health constraint or incurring high action penalties.

Task 3 Name
Torso Alignment

Task 3 Description
Maintain a healthy_reward of 1, minimize control_cost, maintain an angle_of_the_torso within (-0.2, 0.2), and maintain low angular_velocity_of_the_torso_angle; this task focuses on keeping the torso aligned properly during movements to align the agent with the requirement of the healthy_reward.

Task 4 Name
Coordinated Joint Movements

Task 4 Description
Maintain a healthy_reward of 1, minimize control_cost, maintain smooth angular velocities for the thigh, leg, and foot joints (minimizing angular_velocity_of_the_thigh_joint, angular_velocity_of_the_leg_joint, and angular_velocity_of_the_foot_joint); this task will train the agent to coordinate its joints in a manner that contributes to efficient movement while maintaining balance and healthiness.

Final Task Original Task
Original Task Description

Minimize control_cost, maintain a healthy_reward of 1, and maximize x_coordinate_of_the_torso. The goal here is to synthesize all the learned tasks - balancing, smooth hopping, torso alignment, and coordinated joint movements - to achieve efficient hops in the forward direction while maintaining health constraints and minimizing energy expenditure.