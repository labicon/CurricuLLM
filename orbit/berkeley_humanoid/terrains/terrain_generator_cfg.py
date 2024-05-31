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
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=1.0,
        ),
    },
)