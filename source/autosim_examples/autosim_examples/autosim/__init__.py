from autosim import register_pipeline

register_pipeline(
    id="AutoSimPipeline-FrankaCubeLift-v0",
    entry_point=f"{__name__}.pipelines.franka_lift_cube:FrankaCubeLiftPipeline",
    cfg_entry_point=f"{__name__}.pipelines.franka_lift_cube:FrankaCubeLiftPipelineCfg",
)
