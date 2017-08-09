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

STATE_START = 0
STATE_GO_TOWARDS_GOAL = 1
STATE_FOLLOW_OBSTACLE = 2


class Bug_2_ref(agent.Agent):
    """
    Reference implementation of the Bug 1 algorithm: if not in contact with an
    obstacle moves in a straight line towards the goal. If in contact with an
    obstacle, moves along the entire obstacle line and records the location with
    minimum distance to the
    """

    def init(self):
        self.d_hit = 0
        self.a_start = np.array([0, 0])
        self.state = STATE_START
        self.follow_obstacle = 0
        self.old_fac = 0

    def name(self):
        return "Bug 2 (Ref.)"

    def color(self):
        return "#204a87"

    def trace_description(self):
        return [agent.TraceField("state", "State", ["IDLE", "DIR\\_MEM", "MOVE", "DIST\\_MEM", "FOLLOW"])]

    def behave(self, sensors, motor):
        if self.state == STATE_START:
            self.a_start = sensors.absolute_compass()
            self.state = STATE_GO_TOWARDS_GOAL
        elif self.state == STATE_GO_TOWARDS_GOAL:
            common.move_towards_goal(sensors, motor)

            if sensors.hit_obstacle():
                self.follow_obstacle += 1.0 * self.dt_coarse
            else:
                self.follow_obstacle -= self.follow_obstacle * self.dt_coarse

            if self.follow_obstacle >= 0.25:
                # Switch to the follow_obstacle state
                self.d_hit = sensors.distance_to_goal()
                self.state = STATE_FOLLOW_OBSTACLE
                self.follow_obstacle = 0
        else:
            common.follow_obstacle(sensors, motor, self.radius)

            # Switch to the trace mode once we again reached to start
            # location
            if (np.abs(np.dot(self.a_start, sensors.absolute_compass())) >
                    0.999 and sensors.distance_to_goal() + 10e-2 < self.d_hit):
                self.state = STATE_GO_TOWARDS_GOAL

        return [4 if self.state == STATE_FOLLOW_OBSTACLE else self.state + 1]

