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

class Bug_follow_neural(NeuralAgent):
    """
    Bug which solely shows the follow_obstacle behaviour.
    """

    def name(self):
        return "Follow (Neural)"

    def color(self):
        return "#9dc50c"

    def init(self):
        with self.network:
            follow_net = common_neural.make_follow_obstacle_network(
                self.seed_gen, self.sensors.radar_range,
                self.sensors.radar_vectors, self.radius)
            nengo.Connection(self.node_radar_depth, follow_net.radar_depth)
            nengo.Connection(follow_net.motor, self.node_motor)
