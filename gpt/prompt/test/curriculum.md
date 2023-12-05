Task 1 Name
Learn to Stand

Task 1 Description
Before hopping, the agent must learn to balance and stand upright. Starting from a variety of positions within the initial state space, the goal is to achieve and maintain an upright torso with minimal movement. The agent should apply torques to all joints to prevent the hopper from falling over, prioritizing minimal energy expenditure. Success is measured by maintaining a z height of the torso above a threshold (such as 1.2 times the initial z height) and keeping the torso angle within a limited range for a duration without falling or exceeding energy thresholds.

Task 2 Name
Static Balance on One Foot

Task 2 Description
The agent needs to learn how to balance statically on its foot. This task involves controlling the torques on the three hinges without any hopping, such that the hopper maintains an upright position with zero velocity while withstanding perturbations (small external forces applied to the body parts). Success is measured by the hopper's ability to re-stabilize after perturbations without falling, maintaining a z height above a preset threshold and torso angle within a limited range for a certain duration.

Task 3 Name
Small Controlled Hops

Task 3 Description
In this task, the agent progresses from standing to performing small vertical hops while trying to minimize forward or backward movement. The agent should apply controlled torques that lift the foot off the ground by a small amount and then cushion the landing to achieve a soft touch-down. The performance is measured by the consistency of the hop height, the smoothness of each landing, and the ability to maintain overall balance without tilting or falling.

Task 4 Name
Forward Hops with Balance

Task 4 Description
The agent now combines balance with small forward hops. Here, the goal is to hop forward, landing each time without losing balance and falling over. The agent must maintain a forward trajectory, controlling its body orientation and applying torques to regulate both vertical and horizontal motion. Performance is evaluated based on the forward distance covered, the consistency of hop lengths, the time spent airborne, and the ability to remain steady upon landing.

Task 5 Name
Maximize Forward Velocity

Task 5 Description
Building on the previous task, the current goal is to maximize forward velocity while hopping. The agent should apply torques to the hinges to achieve the greatest possible forward speed through a sequence of hops without compromising its stability. The performance is evaluated on the increase in the x velocity of the torso as calculated in the reward function, while also managing control costs and maintaining the health criteria set by the environment.

Task 6 Original Task
Efficient and Healthy Hopping

Task 6 Original Task Description
The original task is to combine the skills learned from the earlier tasks to perform efficient and healthy hops in the forward direction. The agent must apply torques on the hinges connecting the torso, thigh, leg, and foot to achieve maximum forward momentum. The hops need to be smooth, controlled, and efficient, optimizing for forward speed while minimizing energy use and avoiding unhealthy states (falling, tipping over, or reaching unsafe angles). Hopping performance will be assessed using the reward function provided in the environment code, taking into account x velocity, healthiness of the hopper, and control costs.