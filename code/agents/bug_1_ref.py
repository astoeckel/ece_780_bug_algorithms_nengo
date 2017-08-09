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

STATE_GO_TOWARDS_GOAL = 0
STATE_FOLLOW_OBSTACLE = 1
STATE_GO_MIN = 2

class Bug_1_ref(agent.Agent):
    """
    Reference implementation of the Bug 1 algorithm: if not in contact with an
    obstacle moves in a straight line towards the goal. If in contact with an
    obstacle, moves along the entire obstacle line and records the location with
    minimum distance to the goal.
    """

    def init(self):
        self.d_hit = 0
        self.a_hit = np.array([0, 0])
        self.d_min = 0
        self.t_traveled = 0
        self.state = STATE_GO_TOWARDS_GOAL
        self.follow_obstacle = 0

    def name(self):
        return "Bug 1 (Ref.)"

    def color(self):
        return "#4e9a06"

    def trace_description(self):
        return [
            agent.TraceField("state", "State",
                             ["MOVE", "FOLLOW", "MIN"]),
            agent.TraceField("follow_obstacle", "follow\\_obstacle"),
            agent.TraceField("d_hit", "d\\_hit"),
            agent.TraceField("d_min", "d\\_min")
        ]

    def behave(self, sensors, motor):
        if self.state == STATE_GO_TOWARDS_GOAL:
            common.move_towards_goal(sensors, motor)

            if sensors.hit_obstacle():
                self.follow_obstacle += 1.0 * self.dt_coarse
            else:
                self.follow_obstacle -= self.follow_obstacle * self.dt_coarse

            if self.follow_obstacle >= 0.25:
                # Switch to the follow_obstacle state
                self.d_hit = sensors.distance_to_goal()
                self.d_min = self.d_hit
                self.a_hit = sensors.absolute_compass()
                self.t_traveled = 0
                self.state = STATE_FOLLOW_OBSTACLE
                self.follow_obstacle = 0
        else:
            common.follow_obstacle(sensors, motor, self.radius)

            # Record the time we've been in this stage
            self.t_traveled += self.dt_coarse

            if self.state == STATE_FOLLOW_OBSTACLE:
                # Record the minimum distance
                self.d_min = min(self.d_min, sensors.distance_to_goal())

                # Switch to the trace mode once we reached the start location
                if self.t_traveled > 2:
                    if ((np.dot(self.a_hit, sensors.absolute_compass()) > 0.9) and
                            np.abs(self.d_hit - sensors.distance_to_goal()) <
                            1e-1):
                        self.state = STATE_GO_MIN
            if self.state == STATE_GO_MIN:
                if np.abs(self.d_min - sensors.distance_to_goal()) < 1e-1:
                    self.state = STATE_GO_TOWARDS_GOAL

        return [self.state, self.follow_obstacle, self.d_hit, self.d_min]

