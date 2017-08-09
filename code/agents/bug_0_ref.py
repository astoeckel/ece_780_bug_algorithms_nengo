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

import agent
import common

import numpy as np


class Bug_0_ref(agent.Agent):
    """
    Reference implementation of the Bug 0 algorithm: if not in contact with an
    obstacle moves in a straight line towards the goal. If in contact with an
    obstacle moves into clockwise direction with respect to the obstacle
    boundary until we can move into the direction of the obstacle. The latter
    behaviour is implemented by periodically trying to move into the obstacle
    direction.
    """

    def init(self):
        self.follow_obstacle = 0.0
        self.t_follow = 0.0

    def trace_description(self):
        return [
            agent.TraceField("state", "State",
                             ["MOVE", "FOLLOW"]),
            agent.TraceField("t_follow", "t\\_follow"),
            agent.TraceField("follow_obstacle", "follow\\_obstacle")
        ]

    def name(self):
        return "Bug 0 (Ref.)"

    def color(self):
        return "#a40000"

    def behave(self, sensors, motor):
        # State transition
        if sensors.goal_visible() or self.t_follow > 1.0:
            self.follow_obstacle *= 0.5
        elif sensors.hit_obstacle():
            self.follow_obstacle += 1.0 * self.dt_coarse
        else:
            self.follow_obstacle -= self.follow_obstacle * self.dt_coarse

        # Behaviour implementation:
        if self.follow_obstacle > 0.25:
            common.follow_obstacle(sensors, motor, self.radius)
            self.t_follow += self.dt_coarse  # Try moving towards the goal every 0.5s
        else:
            self.t_follow = 0
            common.move_towards_goal(sensors, motor)

        return [
            self.follow_obstacle > 0.25, self.t_follow, self.follow_obstacle
        ]

