from gymnasium.envs.registration import register, registry

if "VirtualTB-v0" not in registry:
    register(
        id="VirtualTB-v0",
        entry_point="virtualTB.envs:VirtualTB",
    )
