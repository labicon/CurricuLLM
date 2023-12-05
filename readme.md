# Install Custom Environments

Before you run curriculum training, please edit your environment code and install edited custom environments.  

`pip install -e environments`

## Making video in Google Cloud

If you want to make video in Google Cloud, you should use virtual screen.

```
xvfb-run -s "-screen 0 1400x900x24" python3 visualize_policy.py
```