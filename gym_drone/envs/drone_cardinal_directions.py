import gym
from gym import spaces
from gym.utils import seeding 
import numpy as np

def lattice_path_length(a, b):
    a_x, a_y = a
    b_x, b_y = b
    return abs(a_x - b_x) + abs(a_y - b_y)

def create_grid(shape):
    grid = np.zeros(shape)
    for row in range(len(grid)):
        for column in range(len(grid[row])):
            grid[row][column] = lattice_path_length((0, 0), (row, column))
    

class DroneCardinalDirectionsEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    #grid 
    
    def __init__(self, rows, columns):
        self._rows = rows
        self._columns = columns
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(rows),
            spaces.Discrete(columns),
            spaces.Discrete(rows),
            spaces.Discrete(columns)))
        self.seed()
        #spaces.Box for a n-dimensional box of values
        #use that later on when map is part of the
        #state itself.
        #Gym soccer also uses spaces.Box with shape=1 to
        #process floats as action_space
        
        self._grid = create_grid((rows, columns))
        
    def _step(self, action):
        """
        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        assert self.action_space.contains(action)
        direction = ACTION_LOOKUP[action]
        self._drone_pos += direction
        drone_x, drone_y = self._drone_pos
        reward = 0
        if drone_x < 0 or drone_y < 0 
           or drone_x > self._columns or drone_y > self._rows:
            reward -= 1
            self._drone_pos -= direction
            drone_x, drone_y = self._drone_pos
        reward -= self._grid[drone_x][drone_y]
        
        episode_over = False
        if self._drone_pos == self._goal_pos:
            reward = 0
            episode_over = True
        
        ob = (self._drone_pos, self._goal_pos)
        
        return ob, reward, episode_over, {}
        
    def _reset(self):
        self._drone_pos = 
        self._goal_pos = 
        pass
        
    def _render(self, mode='human', close=False):
        pass

ACTION_LOOKUP = {
    0 : (1, 0),
    1 : (0, 1),
    2 : (-1,0),
    3 : (0,-1)
}