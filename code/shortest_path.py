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
# shortest_path.py
#
# This module is used to construct a weighted visibility graph from a map and
# to compute the shortest path through the map from a set of start points to a
# set of goal points.
#

import numpy as np
import geometry

def construct_visibility_graph(node_locations, obstacles):
    """
    Constructs a weighted adjacency list which contains the distance between
    two nodes if the nodes are visible to each other, or zero if the nodes are
    not visible from each other.

    The resulting adjacency list consists of a list of tuples (i, w) for each
    node specifying the neighbouring nodes and the connection weight.

    node_locations: a N x 2 matrix, where N is the number of nodes. Each row
    specifies the location of a node in 2D space.
    obstacles: a list of polygons defining obstacles in space.
    """

    import sys

    assert node_locations.ndim == 2 and node_locations.shape[1] == 2, \
        "node_locations must be a 2D-matrix with two columns"
    N = node_locations.shape[0]

    # Create a collision query engine containing the obstacles
    cqe = geometry.CollisionQueryEngine(obstacles)

    # Iterate over all pairs of nodes and fill the result structure
    res = [{} for _ in range(N)]
    NN, cnt = N * (N - 1) / 2, 0
    for i in range(N):
        x0, y0 = node_locations[i]
        for j in range(i + 1, N):
            cnt = cnt + 1
            x1, y1 = node_locations[j]
            dx, dy = x1 - x0, y1 - y0
            w = np.hypot(dx, dy)
            ox, oy = 1e-6 * dx / w, 1e-6 * dy / w
            if cqe.point_in_polygon(x0 + ox, y0 + oy):
                continue
            has_collision, _, _ = cqe.check_collision(x0 + ox, y0 + oy,
                                                      dx - 2 * ox, dy - 2 * oy)
            if not has_collision:
                res[i][j] = w
                res[j][i] = w
        sys.stderr.write("Building visibility graph {:0.2f}%    \r".
                         format(cnt / NN * 100))
        sys.stderr.flush()
    print()
    return res


def dijkstra(start_idx, goal_idcs, weighted_adjacency_list):
    """
    Computes the shortest path between a start node and a list of goal nodes
    in a graph. The graph is specified by the weighted adjacency list. Returns
    a list for each index in goal_idcs specifying the shortest path as a list of
    indices starting with start_idx (if there is a path to the node). Note that
    this is not a particularly efficient implementation (does not use a heap).
    """

    # Make sure start_idx and goal_idcs are valud
    N = len(weighted_adjacency_list)
    assert start_idx < N, "start_idx out of bounds"

    # Make sure goal_idcs is converted to a list
    goal_idcs = np.atleast_1d(goal_idcs)
    assert goal_idcs.ndim == 1, "goal_idcs must be a one-dimensional list!"
    assert np.all(goal_idcs < N), "goal_idx out of bounds"

    # Forward-tracking -- calculate the minimum weight for each node in the
    # graph when starting at start_idx
    ws = np.ones(N) * np.inf
    ws[start_idx] = 0
    unvisited = np.ones(N, dtype=bool)
    parents = np.zeros(N, dtype=int)
    idcs = np.arange(N, dtype=int)
    while np.any(unvisited):
        i = idcs[unvisited][np.argmin(ws[unvisited])]
        for j, w in weighted_adjacency_list[i].items():
            if ws[i] + w < ws[j]:
                ws[j] = ws[i] + w
                parents[j] = i
        unvisited[i] = False

    # Back-tracking: extract the shortest path
    res = [[j] for j in goal_idcs]
    for k, goal_idx in enumerate(goal_idcs):
        if ws[goal_idx] == np.inf:
            res[k] = []
            continue
        i = goal_idx
        while i != start_idx:
            res[k].append(parents[i])
            i = parents[i]
        res[k] = res[k][::-1]  # Reverse the path
    return res


def shortest_paths(start,
                   goal,
                   obstacles,
                   subdivision_radius=0.5,
                   return_debug_info=False):
    """
    Calculates the shortest paths from each start location to each goal
    location. Returns a list for each start position, which contains a list for
    each goal position, which contains a list describing the path points.
    """

    # Simplify the obstacles, only the corner points are relevant.
    # Create the node locations and groups
    node_locations = [
        geometry.simplify_polygon(obstacle)
        for obstacle in obstacles
    ]
    offs_start = sum(map(len, node_locations))
    node_locations.extend(map(np.atleast_2d, start))
    offs_goal = sum(map(len, node_locations))
    node_locations.extend(map(np.atleast_2d, goal))
    node_groups = [[i] * len(x) for i, x in enumerate(node_locations)]

    node_locations = np.concatenate(node_locations, axis=0)
    node_groups = np.concatenate(node_groups, axis=0)

    # Calculate the visibility graph
    graph = construct_visibility_graph(node_locations, obstacles)

    # Add connections between the polygon points
    first = 0
    group = 0
    for i in range(1, len(node_locations)):
        if node_groups[i] == node_groups[i - 1]:
            w = np.linalg.norm(node_locations[i - 1] - node_locations[i])
            graph[i - 1][i] = w
            graph[i][i - 1] = w
        else:
            if first != i - 1:
                w = np.linalg.norm(node_locations[first] - node_locations[i -
                                                                          1])
                graph[first][i - 1] = w
                graph[i - 1][first] = w
            first = i

    # For each start location, find the shortest path to each of the goals
    res = [None for _ in start]
    for i, _ in enumerate(start):
        res[i] = dijkstra(offs_start + i,
                          list(range(offs_goal, offs_goal + len(goal))), graph)

    # Convert each path to a series of locations
    for paths in res:
        for i, path in enumerate(paths):
            paths[i] = node_locations[path]

    if return_debug_info:
        return res, node_locations, node_groups, graph
    return res


if __name__ == "__main__":
    import sys
    import ascii_map
    import matplotlib
    import matplotlib.pyplot as plt
    import cProfile

    start, goal, obstacles = ascii_map.parse_ascii_map(ascii_map.test_map)
    plt.plot(start[:, 0], start[:, 1], '+', color='k', markersize=10, zorder=4)
    plt.plot(goal[:, 0], goal[:, 1], 'x', color='k', markersize=10, zorder=4)
 
    res, node_locations, node_groups, graph = shortest_paths(\
        start, goal, obstacles, return_debug_info=True)

    cmap = matplotlib.cm.get_cmap('viridis')
    colours = np.array(
        list(map(cmap, np.linspace(
            0, 1, max(node_groups) + 1)))) * [0.75, 0.75, 0.75, 1.0]

    for i, neighbours in enumerate(graph):
        for j in neighbours.keys():
            plt.plot(
                node_locations[(i, j), 0],
                node_locations[(i, j), 1],
                '-',
                color=[0.75, 0.75, 0.75],
                linewidth=0.5, zorder=0)

    plt.scatter(
        node_locations[:, 0], node_locations[:, 1], color=colours[node_groups], zorder=2)

    for r in res:
        for s in r:
            plt.plot(s[:, 0], s[:, 1], '-', linewidth=2, zorder=3)

    plt.savefig('shortest_path_test.pdf')

    N = 200
    xs = np.random.uniform(-5, 5, (N, 2))
    start_idx = np.random.randint(N)
    goal_idcs = np.random.randint(N, size=5)

    weighted_adjacency_list = [{} for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            w = np.linalg.norm(xs[i] - xs[j])
            if (w < 1):
                weighted_adjacency_list[i][j] = w
                weighted_adjacency_list[j][i] = w

    paths = dijkstra(start_idx, goal_idcs, weighted_adjacency_list)

    fig, ax = plt.subplots()
    for i, neighbours in enumerate(weighted_adjacency_list):
        for j in neighbours.keys():
            ax.plot(xs[(i, j), 0], xs[(i, j), 1], color='k', linewidth=0.5)
    ax.plot(xs[:, 0], xs[:, 1], 'o')
    ax.plot(xs[start_idx, 0], xs[start_idx, 1], 'o', color='r', markersize=10)
    for path in paths:
        ax.plot(xs[path, 0], xs[path, 1], '-')
    plt.savefig('dijkstra_test.pdf')

    plt.show()
