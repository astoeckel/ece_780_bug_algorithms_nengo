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
# environment.py
#
# Module containing the Environment class, which represents the current start,
# and goal location, as well as all obstacles.
#

import numpy as np
import geometry


class Environment:
    """
    The environment class represents the current start and goal location, as
    well as all obstacles in the environment. The environment class provides an
    interface for querying collisions.
    """

    def __init__(self, start, goal, obstacles, obstacle_point_radius=0.5):
        """
        Instantiates the Environment class.

        start: start location.
        goal: goal location.
        obstacles: list of polygons acting as obnstacles in the environment.
        obstacle_point_radius: radius used when subdividing the polygons for
        efficient collision querying.
        """
        self.start = start
        self.goal = goal
        self.cqe = geometry.CollisionQueryEngine(obstacles, obstacle_point_radius)

    def check_collision(self, x, y, dx, dy):
        """
        Sends a ray from location x, y with direction dx, dy through the
        environment. Returns a triple as results consisting of a boolean flag
        indicating whether there was a collision, a distance l that indicates
        how far we have to travel along dx, dy for a collision, as well as a
        vector describing the surface direction at the collision point.
        """
        return self.cqe.check_collision(x, y, dx, dy)

