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
# common.py
#
# Functions commonly used in the agent implementations.
#

import numpy as np

M_ROT = np.array([[np.cos(-0.34 * np.pi), -np.sin(-0.34 * np.pi)],
                [np.sin(-0.34 * np.pi), np.cos(-0.34 * np.pi)]])

def follow_obstacle_vector_weight(depth, min_dist, max_dist):
    return (1 - np.clip((depth - min_dist) / (max_dist - min_dist), 0, 1)) ** 2

def follow_obstacle(sensors, motor, min_dist, max_dist=1.0):
    depths, vectors = sensors.radar()
    weight = follow_obstacle_vector_weight(depths, min_dist, max_dist)
    vector = weight.T @ (vectors @ M_ROT)
    l = np.hypot(vector[0], vector[1])
    if l > 1.0:
        vector = vector / l
    motor.update(vector[0], vector[1])

def move_towards_goal(sensors, motor):
    # Move into the direction of the obstacle
    compass_direction = sensors.compass()
    motor.update(compass_direction[0], compass_direction[1])

