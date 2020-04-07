import gym
from gym import spaces
from gym.utils import seeding 
import numpy as np
import matplotlib.pyplot as plt

class Path(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, point):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        else:
            for i in range(self.capacity - 1):
                self.memory[i] = self.memory[i + 1]
        self.memory[self.position] = point
        self.position = min(self.position + 1, self.capacity - 1)

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
        self._memory_capacity = 10
        self._ax = plt
        if 'rows' in kwargs:
            rows = kwargs.get('rows')
        if 'columns' in kwargs:
            columns = kwargs.get('columns')
        if 'memory_capacity' in kwargs:
            self._memory_capacity = kwargs.get('memory_capacity')
        if 'ax' in kwargs:
            self._ax = kwargs.get('ax')

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

        # Check if action brings drone outside grid,
        # if so cancel movement and punish reward
        reward = 0
        if (drone_x < 0 or drone_y < 0 or
                drone_x >= self._columns or drone_y >= self._rows):
            self._drone_pos = old_drone_pos
            drone_y, drone_x = self._drone_pos
            reward -= 1
        # If the drone is still inside the grid, add
        # the current node to the list of points to
        # be rendered when rendering path.
        else:
            self._path.push(self._drone_pos)

        # Reward is at any point (except goal) the negative
        # of the height at each point.
        reward -= self._grid[drone_y][drone_x]

        # Check if goal is reached, which ends the episode
        episode_over = False
        if self._drone_pos == self._goal_pos:
            reward = 0
            episode_over = True

        return self._get_obs(), reward, episode_over, {}

    def _get_obs(self):
        return np.array([self._drone_pos[0], self._drone_pos[1],
                         self._goal_pos[0], self._goal_pos[1]])

    def reset(self):
        self._drone_pos = pick_random_point(self.np_random, self._shape)
        self._goal_pos = self._drone_pos
        while self._goal_pos == self._drone_pos:
            self._goal_pos = pick_random_point(self.np_random, self._shape)
        # self._path = [self._drone_pos]
        self._path = Path(self._memory_capacity)
        self._path.push(self._drone_pos)
        return self._get_obs()

    def render(self, mode='notebook'):
        if mode == 'notebook':
            self._ax.cla()
            self._ax.imshow(self._grid)
            self._ax.axis("off")
            y, x = ([] for i in range(2))
            for node_y, node_x in self._path.memory:
                x.append(node_x)
                y.append(node_y)

            self._ax.plot(x, y, 'k')
            start_row, start_column = self._path.memory[len(
                self._path.memory) - 1]
            goal_row, goal_column = self._goal_pos
            return self._ax.scatter(x=[start_column, goal_column],
                                    y=[start_row, goal_row],
                                    c='r', s=40, zorder=3)

        elif mode == 'rgb_array':
            output = np.zeros((self._shape[0], self._shape[1], 3), dtype=np.uint8)

            # Set red channel to height
            output[:, :, 0] = self._grid

            # Scale red channel to be approximately 0 to 255
            max_value = np.amax(self._grid)
            min_value = np.amin(self._grid)
            # output[:, :, 0] /= max_value
            # output[:, :, 0] *= 255
            
            # Linear scale, in which every number is moved down
            # by the lowest, so that lowest point lies at 0, and
            # every point is linearly scaled by the factor needed
            # to move the moved highest point to 255. The result is
            # lowest point is at 0 and highest is at 255, and every
            # other is somewhere inbetween
            def scale_linear(highest, lowest, number):
                return (number - lowest) * (255/(highest - lowest))

            for y, row in enumerate(output[:, :, 0]):
                for x, column in enumerate(row):
                    output[y, x, 0] = scale_linear(max_value, min_value, column)

            # Set green pixel at start, blue pixel at goal
            output[self._drone_pos[0] - 1, self._drone_pos[1] - 1, 1] = 255
            output[self._goal_pos[0] - 1, self._goal_pos[1] - 1, 2] = 255

            return output

        return None


ACTION_LOOKUP = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1)
}
