# Neural Bug Algorithm Implementation for the UWaterloo course ECE 780 To 8
# Copyright (C) 2017 Andreas Stöckel
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import geometry

# Default radar angles with a slight asymmetry
DEFAULT_RADAR_ANGLES = np.linspace(-7 * np.pi / 9, 7.5 * np.pi / 9, 8)


class Sensors:
    """
    The sensors class defines the sensor system of an Agent, which provides a
    numerical representation of its current surroundings. Note that agent
    instances do not necessarly use all of the sensors provided.
    """

    def __init__(self,
                 goal,
                 x=0,
                 y=0,
                 theta=0,
                 radar_range=1.0,
                 radar_angles=DEFAULT_RADAR_ANGLES):
        # Copy all arguments not handled by the update function
        self.goal = goal
        self.radar_range = radar_range
        self.radar_angles = radar_angles

        # Setup the radar data structures
        self._radar = np.zeros(len(radar_angles))
        self.radar_vectors = np.array(
            list(map(lambda a: (np.cos(a), np.sin(a)), radar_angles)))

        # The hit_obstacle flag is manually updated in the agent instance
        self._hit_obstacle = False

        # Call update to populate all sensor readings with an initial state
        self.update(x, y, theta, None)

    def update(self, x, y, theta, environment):
        # Copy the location and orientation data
        self.x = x
        self.y = y
        self.theta = theta

        # Update the radar
        for i, angle in enumerate(self.radar_angles):
            dx, dy = np.cos(angle + self.theta), np.sin(angle + self.theta)
            if environment is None:
                self._radar[i] = self.radar_range
            else:
                _, self._radar[i], _ = environment.check_collision(
                    x, y, dx * self.radar_range, dy * self.radar_range)
        self._radar = np.minimum(self.radar_range, self._radar)

        # Check whether the goal is visible from the current location
        if environment is None:
            self._goal_visible = False
        else:
            ray_hit_obstacle, _, _ = environment.check_collision(
                x, y, self.goal[0] - x, self.goal[1] - y)
            self._goal_visible = not ray_hit_obstacle

        # Compass directions
        angle = np.arctan2(self.goal[1] - self.y, self.goal[0] - self.x)
        self._compass = np.array((np.cos(angle - self.theta),
                                  np.sin(angle - self.theta)))
        self._absolute_compass = np.array((np.cos(angle), np.sin(angle)))

        # Distance to goal
        self._distance_to_goal = np.hypot(self.goal[0] - self.x,
                                          self.goal[1] - self.y)

    def radar(self):
        return self._radar, self.radar_vectors

    def hit_obstacle(self):
        return self._hit_obstacle

    def distance_to_goal(self):
        return self._distance_to_goal

    def compass(self):
        return self._compass

    def absolute_compass(self):
        return self._absolute_compass

    def goal_visible(self):
        return self._goal_visible


class Motor:
    """
    The motor class describes the motor system input of the agent. The motor
    system is controlled by a 2D input vector, which defines both the speed of
    the movement and the direction of the movement.
    """

    def __init__(self):
        self.dtheta = 0
        self.vx = 0
        self.vy = 0

    def update(self, vx, vy):
        """
        This function should be called by the agent controller to update the
        current velocity vector of the robot. This vector updates the desired
        direction (relative to the current orientation of the robot), the actual
        position of the robot is updated by the environment and takes collisions
        into account.
        """
        self.vx = vx
        self.vy = vy


class TraceField:
    """
    Structure used to describe the per-agent class trace variables.
    """
    def __init__(self, id_, name, dtype="float"):
        self.id = id_
        self.name = name
        self.dtype = dtype


class Agent:
    """
    The Agent class represents a single agent in the simulation. It provides
    sensors and motors and functionality to update its current location in space
    based on an environment instance.
    """

    def __init__(self,
                 goal,
                 x=0,
                 y=0,
                 theta=0,
                 radius=0.15,
                 rotation_speed=4,
                 dt_coarse=1e-2,
                 dt_fine=1e-3,
                 seed=None,
                 sensors=None,
                 motor=None):
        """
        Initializes a new Agent instance.

        goal: goal location.
        x: initial x location.
        y: initial y location.
        theta: initial rotation in radians.
        radius: radius of the disc-shaped robot in meters.
        rotation_speed: rotation speed multiplier, allows to speed up/slow down
        the reaction of the agent to rotation commands compared to direction
        commands. A value of one corresponds to a maximum rotation speed of one
        full rotation per second.
        dt_coarse: simulator update timestep.
        dt_fine: internal dynamical system simulation timestep.
        seed: random number generator seed or None if a random seed should be
        used.
        sensors: a Sensors instance that should be used by the agent or None if
        a default Sensors instance should be used.
        motor: a Motor instance that should be used by the agent or None if a
        default Motor instance should be used.
        """
        self.goal = goal
        self.x = x
        self.y = y
        self.theta = theta
        self.t = 0
        self.radius = radius
        self.rotation_speed = rotation_speed
        self.dt_coarse = dt_coarse
        self.dt_fine = dt_fine
        self.seed = seed

        # Create the sensors instance if it wasn't specified in the constructor
        if sensors is None:
            sensors = Sensors(goal, x, y, theta)

        # Create the motor instance if it wasn't specified in the constructor
        if motor is None:
            motor = Motor()

        self.sensors = sensors
        self.motor = motor
        self._trajectory = []  # Recorded trajectory
        self._trace = [] # Recorded trace
        self.reached_goal = False

        self.init()

    def name(self):
        """
        Returns a human readable name of the agent class used in visualisation.
        """
        return self.__class__.__name__

    def color(self):
        return "#5c3566"

    def trajectory(self):
        """
        Returns the recorded robot trajectory. The trajectory is a 2D matrix
        where each row corresponds to a timestamp
        """
        return np.array(self._trajectory, dtype=np.float64)

    def trace(self):
        """
        Returns the recorded robot trace. Call trace_description for a
        description of the individual fields.
        """
        return np.array(self._trace, dtype=np.float64)

    def trace_description(self):
        """
        Returns an array describing each of the array fields returned by the
        "behave" method. Each array entry must be an instance of TraceField.
        """
        return []

    def init(self):
        """
        This method should be overriden by the actual Agent implementation.
        """
        pass

    def behave(self, sensors, motor):
        """
        This method should be overriden by the actual Agent implementations and
        provides the actual sensor-motor loop for the agent. The function may
        return an array of values, which are tracked and stored along with the
        trajectory. The variables should be described in the trace_description
        method.
        """
        pass

    def update(self, environment, dt, dt_fine=1e-3):
        """
        This function progresses the agent simulation a single time step dt.

        environment: Environment instances used to read the sensor information
        and to provide collision feedback.
        dt: simulation timestep in seconds. This is a macro timestep and
        describes the interval of the calls to the "update" function.
        dt_fine: internal simulation timestep then should be used when solving
        dynamical systems inside the update function.
        """

        # Set the current position in the sensors
        self.sensors.update(self.x, self.y, self.theta, environment)

        # Call the "behave" method for the desired behaviour
        trace = self.behave(self.sensors, self.motor)

        # Add the _trace element to the recorded trace
        if not trace is None:
            self._trace.append([self.t] + trace)

        # Fetch the current motor velocity vector
        vx, vy = self.motor.vx, self.motor.vy
        s = np.hypot(vy, vx) * dt

        # Update theta in a single step (this is only an approximation)
        if s > 1e-6:
            self.theta += np.arctan2(vy, vx) * self.rotation_speed * dt

        # Reset the sensors "hit_obstacle" flag
        self.sensors._hit_obstacle = False

        # Update the position, slide along obstacles
        dx, dy = np.cos(self.theta), np.sin(self.theta)
        it = 0
        while (it == 0) or (s > 1e-3 * dt):
            # Raycast for collisions into this direction, originating at both
            # the centre and the two points on the circle orthogonal to the
            # movement direction
            pnts = ((self.x, self.y), (self.x - dy * self.radius,
                                       self.y + dx * self.radius),
                    (self.x + dy * self.radius, self.y - dx * self.radius), )
            min_l, min_direction = np.inf, None
            for i, pnt in enumerate(pnts):
                has_collision, l, direction = environment.check_collision(
                    pnt[0], pnt[1],
                    dx * (s + self.radius), dy * (s + self.radius))
                if has_collision:
                    if i == 0:
                        l = min(s,
                                geometry.circle_line_collision(
                                    dx, dy, direction[0], direction[1],
                                    self.radius, l))
                    else:
                        l = min(s, l - self.radius)
                    if l < min_l:
                        min_l = l
                        min_direction = direction

            # If there has been no collision detected, just move into the
            # desired direction
            if min_direction is None:
                self.x += dx * s
                self.y += dy * s
                s = 0
            else:
                # We had a collision
                self.sensors._hit_obstacle = True

                # Move towards the obstacle, stay a small amount away,
                # proportional to the timestep
                min_advance = 0 if it < 1 else -0.1 * dt  # Avoid hangs
                self.x += max(min_advance, min_l - 0.1 * dt) * dx
                self.y += max(min_advance, min_l - 0.1 * dt) * dy
                s = s - abs(max(min_advance, min_l -
                                0.1 * dt))  # Decrease the remaining way

                # Slide along the surface of the obstacle by moving in the
                # direction of the obstacle outline and reducing the remaining
                # step size by the dot product of the original movement
                # direction and the surface direction
                proj = np.dot(min_direction, [dx, dy])
                s = s * np.abs(proj)
                dx, dy = min_direction * np.sign(proj)

            # Append the point to the trajectory
            self._trajectory.append((self.t, self.x, self.y, self.theta))

            # This is no longer the first iteration
            it = it + 1

        # Check whether we reached the goal
        if np.hypot(self.x - self.goal[0],
                    self.y - self.goal[1]) < self.radius:
            self.reached_goal = True

        # Update the time measure
        self.t += dt

