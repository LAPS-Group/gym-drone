import gym
from gym import spaces
from gym.utils import seeding
import math
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000
from os import listdir
from os.path import isfile, join


def get_heightmap(training_data, shape, np_random):
    # return training_data[0:shape[0], 0:shape[1]]
    rows, columns = shape
    data_rows, data_columns = training_data.shape
    if data_rows - rows == 0 or data_columns - columns == 0:
        start_position = (0, 0)
    else:
        start_position = (np_random.choice(data_rows - rows),
                          np_random.choice(data_columns - columns))
    return training_data[start_position[0]:start_position[0] + rows,
                         start_position[1]:start_position[1] + columns]


class WaypointEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, **kwargs):
        # the number of files to use from the training_data_dir. By default 0,
        # which will load every file in the folder into memory.
        self._training_samples = 0
        if 'training_samples' in kwargs:
            self._training_samples = kwargs.get('training_samples')
        # the directory of training data. if none are specified the random 
        # values are instead used for each cell.
        self._training_data_dir = None
        if 'training_data_dir' in kwargs:
            self._training_data_dir = kwargs.get('training_data_dir')
            if self._training_data_dir[-1] != '/':
                self._training_data_dir += '/'
        # the diagonal size of the grid
        self._grid_size = 2
        if 'grid_size' in kwargs:
            self._grid_size = kwargs.get('grid_size')
        # number of different possible height_values
        self._height_values = 10
        if 'height_values' in kwargs:
            self._height_values = kwargs.get('height_values')
        # where to plot
        self._ax = plt
        if 'ax' in kwargs:
            self._ax = kwargs.get('ax')

        # which corners are the start and stop position
        self._oriention = 0
        self._start = (0, 0)
        self._stop = (0, 0)

        self._shape = (self._grid_size, self._grid_size)
        self._heightmap = np.zeros(self._shape)

        self._training_data = []
        self._load_training_data(self._training_samples)

        self.action_space = spaces.Discrete(self._grid_size)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(2),
             #spaces.MultiDiscrete([self._height_values]*(self._grid_size**2))))
             spaces.MultiDiscrete(np.zeros(self._shape) + self._height_values)))

        self.seed()
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
                image_type = img.mode
                if count == 0:
                    self._training_data = np.zeros((number, width, height))

                img_array = np.array(img)
                if img_array.ndim == 2:
                    self._training_data[count, :, :] = img_array
                if img_array.ndim == 3:
                    self._training_data[count, :, :] = img_array[:, :, 0] / 255

    def _get_obs(self):
        return (self._oriention, self._heightmap)

    def _number_to_index(self, number):
        # uses matrix indexing, (row, column)
        return (number % self._grid_size, number // self._grid_size)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        if self._oriention == 0:
            waypoint = (action - (self._grid_size - 1), action)
        else:
            waypoint = ((self._grid_size - 1) - action, action)

        travel_points = []
        # go to the correct column
        travel_points.append((0, action))
        # bottom of grid
        travel_points.append((self._grid_size - 1, action))
        # stop position
        travel_points.append(self._stop)

        y, x = self._start
        cells = 1
        cells_traversed = [self._start]
        total = self._heightmap[y][x]
        for position in travel_points:
            while (y, x) != position:
                if y != position[0]:
                    y += int(math.copysign(1, position[0] - y))
                elif x != position[1]:
                    x += int(math.copysign(1, position[1] - x))
                cells += 1
                cells_traversed.append((y, x))
                total += self._heightmap[y][x]

        self._cells = cells
        self._total_height = total

        state = self._get_obs()
        #reward = (1/(total + 1))/cells
        reward = -(total/cells) + 255
        episode_over = True
        return state, reward, episode_over, { 'cells': cells_traversed }

    def reset(self):
        # set random start and stop orientation
        # 0: start in upper left, stop in bottom right
        # 1: start in upper right, stop in bottom left
        self._oriention = self.observation_space[0].sample()
        if self._oriention == 0:
            self._start = (0, 0)
            self._stop = (self._grid_size - 1, self._grid_size - 1)
        else:
            self._start = (0, self._grid_size - 1)
            self._stop = (self._grid_size - 1, 0)

        # set the random heights of the grid
        if self._training_data_dir is not None:
            min_height = 0
            max_height = 0
            while min_height == max_height:
                self._heightmap = get_heightmap(
                    self._training_data[self.np_random.choice(
                        len(self._training_data) - 1)],
                    self._shape,
                    self.np_random)
                min_height = np.amin(self._heightmap)
                max_height = np.amax(self._heightmap)
        else:
            self._heightmap = self.observation_space[1].sample()
        return self._get_obs()

    def render(self, mode='human'):
        if mode == 'rgb_array':
            output = np.zeros(
                (self._shape[0], self._shape[0], 3), dtype=np.uint8)
            output[:, :, 0] = self._heightmap * (255 // (self._height_values - 1))
            output[self._start[0], self._start[1], 1] = 255
            output[self._stop[0], self._stop[1], 2] = 255
            return output
