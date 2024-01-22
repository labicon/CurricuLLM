#!/bin/bash

for i in {0..4}
do
   xvfb-run -s "-screen 0 1400x900x24" python3 visualize_policy.py stabilized_movement $i
done
