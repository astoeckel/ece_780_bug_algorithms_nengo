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

from agent import Agent
import common

class Bug_follow_ref(Agent):
    """
    Bug which solely shows the follow_obstacle behaviour.
    """

    def name(self):
        return "Follow (Ref.)"

    def color(self):
        return "#c4a000"

    def behave(self, sensors, motor):
        common.follow_obstacle(sensors, motor, self.radius)
