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

from neural_agent import NeuralAgent
import common_neural

import nengo
import numpy as np


class Bug_0_neural(NeuralAgent):
    """
    Neural implementation of the Bug 0 algorithm. Tries to move into a straight
    line towards the goal. If in contact with an obstacle moves into clockwise
    direction with respect to the obstacle boundary until we can move into the
    direction of the obstacle.
    """

    def name(self):
        return "Bug 0 (Neural)"

    def color(self):
        return "#ce5c00"

    def init(self):
        with self.network:
            # Integrate the hit_obstacle signal with slight decay
            ens_hit_obstacle = nengo.Ensemble(
                n_neurons=50,
                dimensions=1,
                seed=self.seed_gen(),
                label="ens_hit_obstacle")
            nengo.Connection(
                ens_hit_obstacle,
                ens_hit_obstacle,
                transform=0.9,
                synapse=10e-3)
            nengo.Connection(
                self.node_hit_obstacle, ens_hit_obstacle, transform=0.25)

            # Periodically reset ens_hit_obstacle using an oscillator
            ens_oscillator = nengo.Ensemble(
                50, dimensions=2, radius=0.75, label="ens_oscillator")
            nengo.Connection(
                ens_oscillator,
                ens_oscillator,
                transform=2.0 * np.array([[1, 0.5], [-0.5, 1]]),
                synapse=0.1)  # Approximate period is 1.0s
            nengo.Connection(
                ens_oscillator,
                ens_hit_obstacle.neurons,
                function=lambda x: np.exp(-10 * np.linalg.norm(x - [0, 1])**2),
                transform=-1.5 * np.ones((ens_hit_obstacle.n_neurons, 1)))

            # Represent the compass direction in its own ensemble
            ens_compass = nengo.Ensemble(
                n_neurons=100, dimensions=2, label="ens_compass")
            nengo.Connection(
                self.node_compass, ens_compass, seed=self.seed_gen())

            # Calculate the follow obstacle gradient using the follow obstacle
            # subnetwork
            net_follow_obstacle = common_neural.make_follow_obstacle_network(
                self.seed_gen, self.sensors.radar_range,
                self.sensors.radar_vectors, self.radius)

            # Feed both the movement direction from ens_compass and the movement
            # direction from net_follow_obstacle into the motor output
            nengo.Connection(ens_compass, self.node_motor)
            nengo.Connection(self.node_radar_depth,
                             net_follow_obstacle.radar_depth)
            nengo.Connection(net_follow_obstacle.motor, self.node_motor)

            # Select between the "move to target" action and the
            # "follow obstacle" action by inhibiting either the
            basal_ganglia = nengo.networks.BasalGanglia(2)
            node_threshold = nengo.Node(lambda _: 0.5, label="threshold")
            nengo.Connection(ens_hit_obstacle, basal_ganglia.input[0])
            nengo.Connection(node_threshold, basal_ganglia.input[1])
            nengo.Connection(basal_ganglia.output[0], net_follow_obstacle.stop)
            nengo.Connection(
                basal_ganglia.output[1],
                ens_compass.neurons,
                transform=4 * np.ones((ens_compass.n_neurons, 1)),
                synapse=10e-3)

            # Add some traces
            self.add_trace(basal_ganglia.output, "state", "State",
                           lambda x: 0 if x[0] < x[1] else 1,
                           ["MOVE", "FOLLOW"])
            self.add_trace(ens_oscillator, "ens_oscillator",
                           "ens\\_oscillator",
                           lambda x: x[0])
            self.add_trace(ens_hit_obstacle, "ens_hit_obstacle",
                           "ens\\_hit\\_obstacle")

