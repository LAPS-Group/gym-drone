from gym.envs.registration import register

register(
    id=’DroneCardinal-v0',
    entry_point=’gym_drone.envs:DroneCardinalDirectionsEnv’
)