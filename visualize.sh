#!/bin/bash

for i in {0..3}
do
   xvfb-run -s "-screen 0 1400x900x24" python3 visualize_policy.py goal_orientation $i
done
