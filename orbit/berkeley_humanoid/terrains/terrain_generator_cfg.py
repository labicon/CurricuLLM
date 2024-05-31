"""Configuration for custom terrains."""

import omni.isaac.orbit.terrains as terrain_gen
from omni.isaac.orbit.terrains.terrain_generator_cfg import TerrainGeneratorCfg


TRIMESH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.5, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.5, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)