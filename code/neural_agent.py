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

#
# neural_agent.py
#
# Base class for agents controlled by a Nengo neural network. Implements the
# basic network including the input and output nodes, as well as the code
# actually stepping through the Nengo simulation.
#

import agent
import nengo
import numpy as np

DISTANCE_SCALE_FACTOR = 1 / 20


class NeuralAgent(agent.Agent):
    """
    The NeuralAgent class is the base class for neural network based agent
    implementations. It provides the basic Network instance and connects the
    robot simulation to the Nengo neural network simulator. Classes derrived
    from NeuralAgent should override the init method to setup the neural network
    that computes the Agent behaviour. The "behave" method is already
    implemented.
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
        Initializes a new NeuralAgent instance.

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

        # Generate a non-random seed for Nengo networks
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max)

        # Create a random number generator
        self.rng = np.random.RandomState(seed)
        self.seed_gen = lambda: self.rng.randint(np.iinfo(np.int32).max)

        # Rescale the "distance_to_goal" to a range between -1 and 1
        max_dist, min_dist = np.hypot(x - goal[0], y - goal[1]) + 0.5, 0
        rescale_d = lambda d: 2 * (d - min_dist) / (max_dist - min_dist) - 1.0

        # Create the sensors instance if it wasn't specified in the constructor
        if sensors is None:
            sensors = agent.Sensors(goal, x, y, theta)

        # Create the motor instance if it wasn't specified in the constructor
        if motor is None:
            motor = agent.Motor()

        # Build a Nengo network instance and add a Node ("I/O gateways") for
        # each sensor modality, as well as a Node for the motor system
        self.network = nengo.Network(seed=self.seed_gen())
        with self.network:
            # Instantiate all sensor nodes
            s = sensors  # Shorthand
            self.node_radar_depth = nengo.Node(
                lambda _: s.radar()[0], label="radar_depth")
            self.node_hit_obstacle = nengo.Node(
                lambda _: s.hit_obstacle() * 1.0, label="hit_obstacle")
            self.node_distance_to_goal = nengo.Node(
                lambda _: rescale_d(s.distance_to_goal()),
                label="distance_to_goal")
            self.node_compass = nengo.Node(
                lambda _: s.compass(), label="compass")
            self.node_absolute_compass = nengo.Node(
                lambda _: s.absolute_compass(), label="absolute_compass")
            self.node_goal_visible = nengo.Node(
                lambda _: s.goal_visible() * 1.0, label="goal_visible")

            # Instantiate the motor node
            def motor_fun(t, x):
                if t > 50e-3:
                    # Ignore the first 50ms of motor commands. Strange things
                    # may happen here.
                    motor.update(x[0], x[1])
                    return x
                return [0, 0]

            self.node_motor = nengo.Node(motor_fun, size_in=2, label="motor")

        # Trace objects
        self._trace_description = []
        self._probes = []

        # Call the parent constructor, will call the "init" function implemented
        # by child classes.
        super().__init__(goal, x, y, theta, radius, rotation_speed, dt_coarse,
                         dt_fine, seed, sensors, motor)

        # Build the simulator
        self.simulator = nengo.simulator.Simulator(
            self.network, dt=self.dt_fine, progress_bar=False)

    def trace_description(self):
        """
        Returns the trace description as assembled by child classes using the
        add_trace method.
        """
        return self._trace_description

    def behave(self, sensors, motor):
        """
        Implementation of the behaviour for neural network based agent
        instances. This method just runs the simulation for dt_coarse time.
        """
        self.simulator.run(self.dt_coarse)

        # Gather and return the trace data
        res = [0] * len(self._probes)
        for i, probe in enumerate(self._probes):
            res[i] = probe[1](self.simulator.data[probe[0]][-1, :])
        return res

    def add_trace(self, obj, id_, name, trafo=None, dtype=None, synapse=25e-3):
        """
        Traces the value of one of the given Nengo objects.
        """

        # Add an entry to the _trace_description array
        self._trace_description.append(agent.TraceField(id_, name, dtype))

        # Add a probe for the given object to the network
        if trafo is None:
            trafo = lambda x: x
        with self.network:
            self._probes.append((nengo.Probe(
                obj, sample_every=self.dt_coarse, synapse=synapse), trafo))

