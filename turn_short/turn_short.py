import math
import operator
import numpy as np

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
                return None, []
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
        pre_turn_height, pre_points = TurnShort._line_sampled_height(a,
                                                                     turn['A'],
                                                                     grid)
        turn_height, turn_points = TurnShort._circle_sampled_height(turn['A'],
                                                                    turn['C'],
                                                                    turn['O'],
                                                                    grid)
        post_turn_height, post_points = TurnShort._line_sampled_height(
            turn['C'], c, grid)
        points = pre_points + turn_points + post_points
        # the turn can theoretically end up outside the space of the grid, in
        # that case it will return None.
        if turn_height is None or turn_points == []:
            return None, []
        return pre_turn_height + turn_height + post_turn_height, points
