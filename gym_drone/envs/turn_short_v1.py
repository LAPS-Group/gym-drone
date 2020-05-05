from turn_short import TurnShort
import math
import gym
from gym import spaces
from gym.utils import seeding 
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000
from os import listdir
from os.path import isfile, join


def action_to_position(action, grid_shape, subsampling):
    action_index = (action // subsampling, action % subsampling)
    return (math.floor((grid_shape[0] / subsampling) * action_index[0]),
            math.floor((grid_shape[1] / subsampling) * action_index[1]))

def lattice_path_length(a, b):
    a_x, a_y = a
    b_x, b_y = b
    return abs(a_x - b_x) + abs(a_y - b_y)

def get_heightmap(training_data, shape, np_random):
    # return training_data[0:shape[0], 0:shape[1]]
    rows, columns = shape
    data_rows, data_columns = training_data.shape
    start_position = (np_random.choice(data_rows - rows),
                      np_random.choice(data_columns - columns))
    return training_data[start_position[0]:start_position[0] + rows,
                         start_position[1]:start_position[1] + columns]

def pick_random_point(np_random, shape):
    rows, columns = shape
    return (np_random.choice(rows), np_random.choice(columns))


class TurnShortEnvV1(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, **kwargs):
        # the number of files to use from the training_data_dir. By default 0,
        # which will load every file in the folder into memory.
        self._training_samples = 0
        # shape of heightmap grid
        self._shape = (40, 40)
        # number of actions in each direction, resulting in a subampling**2
        # action space.
        self._subsampling = 40
        # # number of steps in each episode, where each step is a turn
        self._steps = 2
        # the turn rate of the drone
        self._turn_rate = 4
        # the matplotlib ax to plot on, otherwise will plot to the general
        # one by default. Useful if you want the render as a subplot.
        self._ax = plt
        if 'training_data_dir' in kwargs:
            self._training_data_dir = kwargs.get('training_data_dir')
            if self._training_data_dir[-1] != '/':
                self._training_data_dir += '/'
        if 'training_samples' in kwargs:
            self._training_samples = kwargs.get('training_samples')
        if 'shape' in kwargs:
            self._shape = kwargs.get('shape')
        if 'subsampling' in kwargs:
            self._subsampling = kwargs.get('subsampling')
        if 'steps' in kwargs:
            self._steps = kwargs.get('steps')
        if 'turn_rate' in kwargs:
            self._turn_rate = kwargs.get('turn_rate')
        if 'ax' in kwargs:
            self._ax = kwargs.get('ax')

        n_rows, n_columns = self._shape
        self.action_space = spaces.Discrete(self._subsampling ** 2)
        self.observation_space = spaces.Tuple((
            # last point
            spaces.Discrete(n_rows),
            spaces.Discrete(n_columns),
            # current point
            spaces.Discrete(n_rows),
            spaces.Discrete(n_columns),
            # goal point
            spaces.Discrete(n_rows),
            spaces.Discrete(n_columns)))
        self.seed()

        self._last_drone_pos = (None, None)
        self._drone_pos = (None, None)
        self._goal_pos = (None, None)
        self._step = 0
        self._heightmap = np.zeros(self._shape)

        self._training_data = []
        self._load_training_data(number=self._training_samples)

        self.reset()

    def _load_training_data(self, number=0):
        file_names = [f for f in listdir(
            self._training_data_dir) if isfile(join(self._training_data_dir, f))]
        if number != 0:
            file_names = file_names[:number]
        else:
            number = len(file_names)
        rows, columns = self._shape
        self._training_data = np.zeros(
            (number, rows, columns), dtype=np.uint8)
        for count, file_name in enumerate(file_names):
            with Image.open(self._training_data_dir + file_name) as img:
                pixels = img.load()
                width, height = img.size
                if count == 0:
                    self._training_data = np.zeros((number, width, height))

                grid = np.zeros((height, width))
                for y in range(height):
                    for x in range(width):
                        grid[y][x] = pixels[x, y]
                # self._training_data.append(grid)
                self._training_data[count, :, :] = grid

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if self._step < self._steps - 1:
            new_position = action_to_position(action,
                                              self._shape,
                                              self._subsampling)
            new_position = (int(new_position[0]),
                            int(new_position[1]))
        else:
            new_position = self._goal_pos

        # if first step, only two points. therefore consider only the straight
        # line from the start point to the new point
        if self._step == 0:
            traversible = True
            total_height, points = TurnShort._line_sampled_height(self._drone_pos,
                                                                  new_position,
                                                                  self._heightmap)
        # otherwise, consider the turn from the last position, turning before
        # current position, headed towards new_position
        else:
            # check if path is flyable considering the turnrate
            traversible = TurnShort.flight_possible(self._last_drone_pos,
                                                    self._drone_pos,
                                                    new_position,
                                                    self._turn_rate)
            if traversible:
                total_height, points = TurnShort.get_turn_height(self._last_drone_pos,
                                                                 self._drone_pos,
                                                                 new_position,
                                                                 self._turn_rate,
                                                                 self._heightmap)
        # the turn can theoretically end up outside the space of the grid, in
        # that case it will return None. Check for this, and if it is the case
        # reward is zero.
        if traversible and total_height is None:
            traversible = False

        if not traversible:
            return self._get_obs(), 0, True, {}

        # reward such that there's a positive reward for making the turn, but
        # less reward the higher the total altitude of the point.
        reward = 100 / (total_height + 1)

        self._last_drone_pos = self._drone_pos
        self._drone_pos = new_position
        self._step += 1
        episode_over = self._step >= self._steps
        state = self._get_obs()
        return state, reward, episode_over, {'total_height': total_height,
                                             'points_traversed': points}

    def _get_obs(self):
        return (self._last_drone_pos[0],
                self._last_drone_pos[1],
                self._drone_pos[0],
                self._drone_pos[1],
                self._goal_pos[0],
                self._goal_pos[1])

    def reset(self):
        self._last_drone_pos = (None, None)
        self._drone_pos = pick_random_point(self.np_random, self._shape)
        self._goal_pos = self._drone_pos
        while self._goal_pos == self._drone_pos \
                or lattice_path_length(self._goal_pos, self._drone_pos) < self._shape[0] - 10:
            self._goal_pos = pick_random_point(self.np_random, self._shape)
            self._drone_pos = pick_random_point(self.np_random, self._shape)
        self._step = 0
        # self._path = Path(self._memory_capacity)
        # self._path.push(self._drone_pos)
        min_height = 0
        max_height = 0
        while min_height == max_height:
            self._heightmap = get_heightmap(
                self._training_data[self.np_random.choice(len(self._training_data) - 1)],
                self._shape,
                self.np_random)
            min_height = np.amin(self._heightmap)
            max_height = np.amax(self._heightmap)
        return self._get_obs()

    def render(self, mode='human'):
        pass
