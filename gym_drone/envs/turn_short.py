import gym
from gym import spaces
from gym.utils import seeding 
import math
import operator
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000
from os import listdir
from os.path import isfile, join


class TurnShort:

    def _vector_angle(a, b, c):
        BA = tuple(map(operator.sub, a, b))
        BC = tuple(map(operator.sub, c, b))
        BA_magnitude = np.linalg.norm(BA)
        BC_magnitude = np.linalg.norm(BC)

        # find angle between vectors
        dot_product = np.asarray(BA) @ np.asarray(BC)
        # angle in radians. Cap the value of angle between 0 and 1, because of
        # floating point errors, and acos not being defined outside that range.
        angle = math.acos(
            max(min(dot_product / (BA_magnitude * BC_magnitude), 1), -1))
        return angle

    def _line_sampled_height(start, stop, grid, stepsize=0.1):
        vector = tuple(map(operator.sub, stop, start))
        vector_magnitude = np.linalg.norm(vector)
        steps = int(vector_magnitude // stepsize)
        unit_vector = tuple(
            map(operator.truediv, vector, [vector_magnitude]*2))

        current_x = None
        current_y = None
        total = 0
        points = []
        for step in range(steps):
            current_pos = tuple(map(operator.mul,
                                    unit_vector,
                                    [step * stepsize]*2))
            current_pos = tuple(map(operator.add,
                                    current_pos,
                                    start))
            if math.floor(current_pos[0]) != current_x \
                    or math.floor(current_pos[1]) != current_y:
                current_x = math.floor(current_pos[0])
                current_y = math.floor(current_pos[1])
                points.append((current_x, current_y))
            total += grid[current_x][current_y]
        return total, points

    def _circle_sampled_height(start, stop, origo, grid, stepsize=0.1):
        grid_shape = grid.shape
        OA = tuple(map(operator.sub, origo, start))
        radius = np.linalg.norm(OA)

        # origo plus unit vector in x direction, to find absolute angles.
        O_i = tuple(map(operator.add, origo, (1, 0)))
        rad_start = TurnShort._vector_angle(start, origo, O_i)
        rad_stop = TurnShort._vector_angle(stop, origo, O_i)

        # find arclength, and steps needed to travel stepsize on each step.
        rad_total = abs(rad_start - rad_stop)
        arclength = rad_total * radius
        steps = int(arclength // stepsize)

        # NOTE: if the angle thing turns into a problem, look at turn_short and
        # do the same workaround.

        rad_steps = np.linspace(rad_start, rad_stop, num=steps)
        current_x = None
        current_y = None
        total = 0
        points = []
        for step in rad_steps:
            direction_vector = (math.cos(step), math.sin(step))
            direction_vector = tuple(map(operator.mul,
                                         [radius]*2,
                                         direction_vector))
            # the current point at which the height is sampled
            tangent_point = tuple(map(operator.add, origo, direction_vector))
            if math.floor(tangent_point[0]) != current_x \
                    or math.floor(tangent_point[1]) != current_y:
                current_x = math.floor(tangent_point[0])
                current_y = math.floor(tangent_point[1])
                points.append((current_x, current_y))
            if current_x < 0 or current_x >= grid_shape[1] \
                    or current_y < 0 or current_y >= grid_shape[0]:
                return None
            total += grid[current_x][current_y]
        return total, points

    # a, b, c, where a and c are start and stop points,
    # while b is the center point which determines the
    # angle that tangent ac
    def flight_possible(a, b, c, turnradius):
        BA = tuple(map(operator.sub, a, b))
        BC = tuple(map(operator.sub, c, b))
        BA_magnitude = np.linalg.norm(BA)
        BC_magnitude = np.linalg.norm(BC)

        # angle from tangent lines to circle center, divided by two as it is
        # in the center of the angle between the two vectors.
        angle = TurnShort._vector_angle(a, b, c) / 2

        # use the shortest of the two vectors to calculate the turn from, as if
        # you start turning as early as you can
        shortest = BA_magnitude
        if BC_magnitude < shortest:
            shortest = BC_magnitude

        # use tan to find radius of the turn circle
        #
        # angle divided by two because half the angle is the angle between one
        # of the lines and the center of the circle.
        radius_of_turn = shortest * math.tan(angle / 2)

        # if the radius required for the turn is greater less than the turn
        # radius possible by the drone, the turn is not flyable
        return radius_of_turn >= turnradius

    # a, b, c, where a and c are start and stop points,
    # while b is the center point which determines the
    # angle that tangent ac
    def shortest_turn(a, b, c, turnradius):
        # angle from tangent lines to circle center, divided by two as it is
        # in the center of the angle between the two vectors.
        relative_angle = TurnShort._vector_angle(a, b, c) / 2
        BO_magnitude = turnradius / math.sin(relative_angle)
        # the distance from b to the correct point where a and c will lie.
        tangent_distance = turnradius / math.tan(relative_angle)

        # B + i, to use for finding absolute angle.
        B_i = tuple(map(operator.add, b, (1, 0)))

        # find the angle to the center of the circle by taking the average of
        # the angle of the two vectors. _vector_angle finds the shortes angle
        # between the vectors, so if the angle is >180, it will instead be the
        # equivalent of -angle, so convert it back by doing 2pi - angle.
        BA_angle = TurnShort._vector_angle(B_i, b, a)
        if BA_angle < b[1]:
            BA_angle = 2*math.pi - BA_angle
        BC_angle = TurnShort._vector_angle(B_i, b, c)
        if BC_angle < b[1]:
            BC_angle = 2*math.pi - BC_angle

        # since we're adding the relative angle to the absolute angle later on,
        # check which one is the lowest, so that adding the angle to that will
        # end up inside the angle.
        # lowest_angle_vector = a
        # if BC_angle < BA_angle:
            # lowest_angle_vector = c

        absolute_angle = (BA_angle + BC_angle) / 2
        # absolute_angle = TurnShort._vector_angle(B_i, b, lowest_angle_vector)
        # if lowest_angle_vector[1] < b[1]:
            # absolute_angle = 2*math.pi - absolute_angle
        # absolute_angle += relative_angle

        direction_vector = (math.cos(absolute_angle), math.sin(absolute_angle))
        BO = tuple(map(operator.mul, [BO_magnitude]*2, direction_vector))

        # correct point a and c to be at the tangent position
        BA = tuple(map(operator.sub, a, b))
        BC = tuple(map(operator.sub, c, b))
        BA_magnitude = np.linalg.norm(BA)
        BC_magnitude = np.linalg.norm(BC)
        BA = tuple(
            map(operator.mul, [tangent_distance / BA_magnitude] * 2, BA))
        BC = tuple(
            map(operator.mul, [tangent_distance / BC_magnitude] * 2, BC))

        # make all vectors absolute, instead of relative to b
        O = tuple(map(operator.add, b, BO))
        A = tuple(map(operator.add, b, BA))
        C = tuple(map(operator.add, b, BC))

        return {
            "A": A,
            "C": C,
            "O": O
        }

    def get_turn_height(a, b, c, turn_rate, grid, stepsize=0.1):
        turn = TurnShort.shortest_turn(a, b, c, turn_rate)
        # a turn is composed of three segments: the line leading up to the turn,
        # the turn itself, and the line after the curve to the final point
        pre_turn_height, pre_points = TurnShort._line_sampled_height(
            a, turn['A'], grid)
        turn_height, turn_points = TurnShort._circle_sampled_height(turn['A'],
                                                                    turn['C'],
                                                                    turn['O'],
                                                                    grid)
        post_turn_height, post_points = TurnShort._line_sampled_height(
            turn['C'], c, grid)
        points = pre_points + turn_points + post_points
        # the turn can theoretically end up outside the space of the grid, in
        # that case it will return None.
        if turn_height is None:
            return None
        return pre_turn_height + turn_height + post_turn_height, points


def action_to_position(action, grid_shape, subsampling):
    action_index = (action // subsampling, action % subsampling)
    return ((grid_shape[0] / subsampling) * action_index[0],
            (grid_shape[1] / subsampling) * action_index[1])

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


class TurnShortEnv(gym.Env):

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
        # self._steps = 5
        # the turn rate of the drone
        self._turn_rate = 4
        # the matplotlib ax to plot on, otherwise will plot to the general
        # one by default. Useful if you want the render as a subplot.
        self._ax = plt
        if 'training_data_dir' in kwargs:
            self._training_data_dir = kwargs.get('training_data_dir')
            if self._training_data_dir[-1] != '/':
                self._training_data_dir+= '/'
        if 'training_samples' in kwargs:
            self._training_samples = kwargs.get('training_samples')
        if 'shape' in kwargs:
            self._shape = kwargs.get('shape')
        if 'subsampling' in kwargs:
            self._subsampling = kwargs.get('subsampling')
        # if 'steps' in kwargs:
            # self._steps = kwargs.get('steps')
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
        # self._step = None
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
        new_position = action_to_position(
            action, self._shape, self._subsampling)

        state = self._get_obs()
        if not _TurnShort.flight_possible(self._drone_pos,
                                          new_position,
                                          self._goal_pos,
                                          self._turn_rate):
            return state, 0, True, {}

        # TODO: move this to the _TurnShort class, under a function named 
        # get_turn_height or something.
        total_height, points = _TurnShort.get_turn_height(self._drone_pos,
                                                          new_position,
                                                          self._goal_pos,
                                                          self._turn_rate,
                                                          self._heightmap)
        # the turn can theoretically end up outside the space of the grid, in
        # that case it will return None. Check for this, and if it is the case
        # reward is zero.
        if total_height == None:
            return state, 0, True, {}
        
        # total height member set, as well as the points traversed list, to 
        # make experimenting with different ways to render the problem and
        # reward functions easier.
        self._total_height = total_height
        self._points_traversed = points
        print("test")

        # reward such that there's a positive reward for making the turn, but
        # less reward the higher the total altitude of the point.
        reward = 100 / env._total_height
        episode_over = True

        return state, reward, episode_over, { 'points_traversed': points }

    def _get_obs(self):
        return (self._last_drone_pos[0],
                self._last_drone_pos[1],
                self._drone_pos[0],
                self._drone_pos[1],
                self._goal_pos[0],
                self._goal_pos[1])

    def reset(self):
        # self._last_drone_pos = None
        self._drone_pos = pick_random_point(self.np_random, self._shape)
        self._goal_pos = self._drone_pos
        while self._goal_pos == self._drone_pos \
                or lattice_path_length(self._goal_pos, self._drone_pos) < self._shape[0] - 10:
            self._goal_pos = pick_random_point(self.np_random, self._shape)
            self._drone_pos = pick_random_point(self.np_random, self._shape)
        # self._step = 0
        # self._path = Path(self._memory_capacity)
        # self._path.push(self._drone_pos)
        # TODO: again, temporary way of getting training data, remove this
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
