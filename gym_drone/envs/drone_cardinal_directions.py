import gym
from gym import spaces
from gym.utils import seeding 
import numpy as np
import matplotlib.pyplot as plt

def lattice_path_length(a, b):
    a_x, a_y = a
    b_x, b_y = b
    return abs(a_x - b_x) + abs(a_y - b_y)

def create_grid(shape):
    grid = np.zeros(shape)
    for row in range(len(grid)):
        for column in range(len(grid[row])):
            grid[row][column] = lattice_path_length((0, 0), (row, column))
    return grid

def pick_random_point(np_random, shape):
    rows, columns = shape
    return (np_random.choice(rows), np_random.choice(columns))

class DroneCardinalDirectionsEnv(gym.Env):
    metadata = {'render.modes': ['human', 'notebook']}
    
    def __init__(self, **kwargs):
        rows = 8
        columns = 8
        if 'rows' in kwargs:
            rows = kwargs.get('rows')
        if 'columns' in kwargs:
            columns = kwargs.get('columns')
        self._shape = (rows, columns)
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
        
        self._drone_pos = None
        self._goal_pos = None
        self._grid = create_grid((rows, columns))
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def step(self, action):
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
        old_drone_pos = self._drone_pos
        self._drone_pos = tuple(map(sum, zip(self._drone_pos, direction)))
        drone_y, drone_x = self._drone_pos
        reward = 0
        if (drone_x < 0 or drone_y < 0 or
                drone_x >= self._columns or drone_y >= self._rows):
            reward -= 1
            self._drone_pos = old_drone_pos
            drone_y, drone_x = self._drone_pos
        else:
            self._path.append(self._drone_pos)
        reward -= self._grid[drone_y][drone_x]
        
        episode_over = False
        if self._drone_pos == self._goal_pos:
            reward = 0
            episode_over = True
        
        return self._get_obs(), reward, episode_over, {}

    def _get_obs(self):
        return (self._drone_pos[0], self._drone_pos[1],
                self._goal_pos[0], self._goal_pos[1])
        
    def reset(self):
        self._drone_pos = pick_random_point(self.np_random, self._shape)
        self._goal_pos = self._drone_pos
        while self._goal_pos == self._drone_pos:
            self._goal_pos = pick_random_point(self.np_random, self._shape)
        self._path = [self._drone_pos]
        return self._get_obs(), 0, False, {}
        
    def render(self, mode='notebook'):
        if mode == 'notebook':
            plt.imshow(self._grid)
            plt.axis("off")
            y, x = ([] for i in range(2))
            for node_y, node_x in self._path:
                x.append(node_x)
                y.append(node_y)

            plt.plot(x, y, 'k')
            start_row, start_column = self._path[0]
            goal_row, goal_column = self._goal_pos
            plt.scatter(x=[start_column, goal_column], y=[start_row, goal_row],
                        c='r', s=40, zorder=3)
            plt.show()

        elif mode == 'rgb_array':
            output = np.zeros((self._shape[0], self._shape[1], 3))

            # Set red channel to height
            output[:, :, 0] = self._grid

            # Scale red channel to be approximately 0 to 255
            max_value = np.amax(self._grid)
            output[:, :, 0] /= max_value

            # Set green pixel at start, blue pixel at goal
            output[self._drone_pos[0] - 1, self._drone_pos[1] - 1, 1] = 255
            output[self._goal_pos[0] - 1, self._goal_pos[1] - 1, 2] = 255

            return output
        
        return None

ACTION_LOOKUP = {
    0 : (1, 0),
    1 : (0, 1),
    2 : (-1, 0),
    3 : (0, -1)
}
