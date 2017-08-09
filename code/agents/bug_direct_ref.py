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
# bug_direct_ref.py
#
# Reference implementation of a Bug just driving into the goal direction.
#

from agent import Agent
import common

class Bug_direct_ref(Agent):
    """
    This is the most primitive possible algorithm: just drive into the direction
    of the goal without any obstacle circumvention.
    """

    def name(self):
        return "Direct (Ref.)"

    def color(self):
        return "#AA305C"

    def behave(self, sensors, motor):
        common.move_towards_goal(sensors, motor)
