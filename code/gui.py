#!/usr/bin/env python3

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
# gui_stub.py
#
# Tiny script which can be used to inspect a neural agent implementation in
# Nengo GUI
#

import simulation

seed = 3782
agent_name = "Bug_2_neural"
agent_class = simulation.classloader(agent_name)
agent = agent_class((0, 0), 0, 0, 0, seed=seed)

model = agent.network
print(agent_name + " has " + str(model.n_neurons) + " neurons in total.")
