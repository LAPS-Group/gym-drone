from gym.envs.registration import register

register(
    id='DroneCardinal-v0',
    entry_point='gym_drone.envs:DroneCardinalDirectionsEnv'
)

register(
    id='TurnShort-v0',
    entry_point='gym_drone.envs:TurnShortEnv'
)

register(
    id='TurnShort-v1',
    entry_point='gym_drone.envs:TurnShortEnvV1'
)
