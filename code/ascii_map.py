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

import numpy as np
import itertools
from matplotlib import _cntr as cntr

import geometry

def parse_ascii_map(txt, scale=0.5, aspect=1.0, jitter=1e-3):
    """
    Helper function which turns an ASCII map into a description understood by
    the environment class. This description consists of a set of possible start
    and goal locations, as well as a list of polygons describing obstacles.

    txt: string containing an ASCII representation of a map. Space characters
    correspond to free space, '#' characters to obstacles, 'S' characters to a
    potential start location and 'G' characters to potential goal locations.
    scale: scale factor for the conversion from row/column indices to meters.
    aspect: aspect ratio correction -- all x-values are multiplied by this
    value.
    jitter: random, uniform noise between -jitter and jitter that is being added
    to all points to reduce the chance of unhandled degenerate cases in the
    computational geometry code.
    """

    # Split at newline characters, add a newline at the beginning and at the end
    txt = ('\n' + txt + '\n').split('\n')

    # Make sure each line in the environment description has the same length
    max_len = max(map(len, txt))
    txt = list(map(lambda s: " " + s + " " * (max_len - len(s)) + " ", txt))

    # Extract possible start and goal positions ("S" and "G")
    txt = list(map(lambda s: bytearray(s, 'ascii'), txt))
    env = np.fliplr(np.array(txt).T)
    start = np.array(list(zip(*np.where(env == ord('S')))), dtype=np.float)
    goal = np.array(list(zip(*np.where(env == ord('G')))), dtype=np.float)

    # Trace the contour line defined by the "#"s into a set of polygons
    env = 1.0 * (env == ord('#'))
    x, y = np.mgrid[:env.shape[0], :env.shape[1]]
    trace = cntr.Cntr(x, y, env).trace(0.4)

    # Simplify the outline coordinates using the corresponding Shapely method.
    # This merges many short line segments into longer line segments, improving
    # the performance of the intersection tests.
    segments = []
    for coords in trace[:len(trace) // 2]:
        segments.append(geometry.simplify_polygon(coords))

    # Aspect ratio correction and scaling
    sv = np.array([scale * aspect, scale])[None, :]
    if start.size > 0:
        start = start * sv
    if goal.size > 0:
        goal = goal * sv
    for segment in segments:
        segment *= sv

        # Apply some random jitter to all nodes
        if jitter > 0:
            segment += np.random.uniform(-jitter, jitter, segment.shape)

    return start, goal, segments


#
# Main Program. Parse and display some ASCII map.
#

test_map = """\
  S    ######################      G
       ######################
       ######################
                                ##   G
           ###########################
  S        ############################
                                    ###
           ##   ##                 ###
            ## ##      G         ###
             ###                ###
  S          ###             ####
              #          #######      G


            #########
            #########
            #########

        #####
        #####                G
        #####

        ###           S
        ###
        ###
"""

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys

    # If a command line argument is given, 
    if len(sys.argv) == 2:
        with open(sys.argv[1], 'r') as f:
            txt = f.read()
    else:
        txt = test_map

    start, goal, polygons = parse_ascii_map(txt)

    fig, ax = plt.subplots()
    ax.hold(True)
    for polygon in polygons:
        ax.add_artist(plt.Polygon(polygon, fill=False, color='k'))
        ax.plot(polygon[:, 0], polygon[:, 1], 'o', markersize=3, color='k', markerfacecolor='k')
    for s in start:
        ax.plot(s[0], s[1], "o", color="#204a87", markersize=7)
    for g in goal:
        ax.plot(g[0], g[1], "x", color="#cc0000", markersize=7)
    ax.set_aspect(1)
    plt.show()

