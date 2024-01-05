#!/bin/bash

for i in {1..5}
do
   xvfb-run -s "-screen 0 1400x900x24" python3 visualize_policy.py stabilized_movement $i
done

for i in {1..5}
do
   xvfb-run -s "-screen 0 1400x900x24" python3 visualize_policy.py goal_oriented_locomotion $i
done
