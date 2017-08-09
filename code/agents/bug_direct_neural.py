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

import nengo


class Bug_direct_neural(NeuralAgent):
    """
    This is the most primitive possible algorithm: just drive into the direction
    of the goal without any obstacle circumvention.
    """

    def name(self):
        return "Direct (Neural)"

    def init(self):
        with self.network:
            # Represent the compass direction in a spiking neuron ensemble
            self.ens_compass = nengo.Ensemble(
                n_neurons=50, dimensions=2, seed=self.seed_gen(),
                label="ens_compass")

            # Connect the input compass direction to the compass ensemble,
            # connect the compass ensemble to the motor
            nengo.Connection(self.node_compass, self.ens_compass)
            nengo.Connection(self.ens_compass, self.node_motor)

