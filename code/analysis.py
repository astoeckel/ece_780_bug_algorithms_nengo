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
# analysis.py
#
# Reads data written by the simulation script. Provides helper functions shared
# with the visualisation script.
#


class Struct:
    """
    Helper class doing away with the annoying Python habit of distinguishing
    between objects and dictionaries. JavaScript FTW.

    https://stackoverflow.com/questions/1305532/convert-python-dict-to-object
    """

    def __init__(self, entries):
        self.__dict__.update(entries)


def process_simulation_files(files, need_shortest_paths=True):
    """
    Reads a set of simulation files and sorts them according to the map and the
    agent that is being used. Returns a structure containing the actual recorded
    data sorted by map and agent, as well as an auxiliary structure containing
    information about individual maps.
    """

    import os
    import pickle

    # Read the files, group them by map and agent
    maps = {}
    map_data = {}
    for filename in files:
        # Get the data mtime
        mtime = os.stat(filename).st_mtime

        # Demarshall the data
        with open(filename, "rb") as f:
            data = pickle.load(f)
        data = Struct(data)
        map_name = data.map_name
        agent_name = data.agent_class

        # Add the map to "maps" and "map_data"
        if not map_name in maps:
            maps[map_name] = {}
            map_data[map_name] = Struct({
                "start": {},
                "goal": {},
                "obstacles": data.obstacles,
                "mtime": 0,
                "duration": 0,
                "shortest_paths": {}
            })
        maps_entry = maps[map_name]
        map_data_entry = map_data[map_name]
        map_data_entry.mtime = max(map_data_entry.mtime, mtime)
        map_data_entry.duration = max(map_data_entry.duration, data.duration)

        # Sort by agent
        if not agent_name in maps_entry:
            maps_entry[agent_name] = []
        maps_entry[agent_name].append(data)

        # Add possible start and goal indices
        if not data.start_idx in map_data_entry.start:
            map_data_entry.start[data.start_idx] = data.start
        if not data.goal_idx in map_data_entry.goal:
            if not data.goal is None:
                map_data_entry.goal[data.goal_idx] = data.goal

    # Measure the shortest paths in the maps
    if need_shortest_paths:
        import shortest_path
        for map_data_entry in map_data.values():
            # Convert start and goal points to lists
            start_idcs, start_pnts = [], []
            goal_idcs, goal_pnts = [], []
            for i, s in map_data_entry.start.items():
                start_idcs.append(i)
                start_pnts.append(s)
            for i, g in map_data_entry.goal.items():
                goal_idcs.append(i)
                goal_pnts.append(g)

            # Calculate the shortest paths between all start and goal locations
            res = shortest_path.shortest_paths(start_pnts, goal_pnts,
                                 map_data_entry.obstacles)
            map_data_entry.shortest_paths = {idx: {} for idx in start_idcs}
            for i, res_per_start in enumerate(res):
                for j, path in enumerate(res_per_start):
                    map_data_entry.shortest_paths[start_idcs[i]][goal_idcs[
                        j]] = path

    # Sort the experiment data
    for agents in maps.values():
        for experiments in agents.values():
            experiments.sort(key=lambda x: (x.start_idx, x.goal_idx, x.seed))

    return maps, map_data
