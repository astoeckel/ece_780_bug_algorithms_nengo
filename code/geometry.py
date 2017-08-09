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
# geometry.py
#
# This helper module contains computational geometry primitives used in the
# implementation of the robot simulator.
#

import numpy as np
from scipy.spatial import cKDTree


def distance_point_to_line(p, p1, p2):
    """
    Calculate the shortest distance between a point and a line defined by two
    points.

    See https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    """
    x, y = p
    x1, y1 = p1
    x2, y2 = p2
    return (np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) /
            np.hypot(y2 - y1, x2 - x1))


def circle_line_collision(dx, dy, ox, oy, r, l):
    """
    Given a circle with radius r that moves along a line with direction vector
    dx, dy. At l * dx, l * dy the center of the circle intersects with another
    line with direction vector ox, oy. This function returns the maximum
    distance the circle can move along dx, dy before intersecting with the line.
    Note that dx, dy and ox, oy must be unit vectors.
    """

    # Calculate the cosine between the angles, we're only interested in the
    # smaller angle (hence the abs)
    cos_alpha = np.abs(dx * ox + dy * oy)

    # Distance we're allowed to travel is l - r / sin(alpha)
    return l - r / np.sqrt(1 - cos_alpha**2)


def simplify_polygon(P, tolerance=1e-3):
    """
    Simplifies a polygon by deleting points which do not significantly change
    the angle of the corresponding line segment. "Significantly" is defined by
    the "tolerance" parameter.
    """

    P = np.array(P)
    N = P.shape[0]
    assert (N > 2), "Polygon is degenerate"
    assert (P.shape[1] == 2), "Polygon must be planar"

    res = [P[0]]
    i = 0
    while i < N:
        p1 = P[i]
        j = i + 2
        while j <= N:
            p2 = P[j % N]
            err = 0
            for k in range(i + 1, j):
                err += distance_point_to_line(P[k], p1, p2)
            if err > tolerance:
                res.append(P[j - 1])
                break
            j = j + 1
        i = j - 1

    return np.array(res)


def subdivide_polygon(P, max_dist=1):
    """
    Subdivides each line segment of the Polygon in such a way that no two
    control points are further apart than max_dist. As a second return value
    returns a list assigning each additional point its original point index.
    """
    P = np.array(P)
    N = P.shape[0]
    assert (N > 1), "Polygon is degenerate"
    assert (P.shape[1] == 2), "Polygon must be planar"

    res = [P[0]]
    idcs = [0]
    for i in range(N):
        j = (i + 1) % N
        p1 = P[i]
        p2 = P[j]
        dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
        n = int(dist / max_dist)
        x = (p2 - p1) / (n + 1)
        for k in range(n):
            res.append(p1 + x * (k + 1))
            idcs.append(i)
        if j > 0:
            res.append(p2)
            idcs.append(j)

    return np.array(res), idcs


def polyline_segment_length(P):
    """
    Given an array if N points returns an array of N lengths, where each entry
    corresponds to the distance between point n and n - 1 (zero for the first
    point).
    """
    PS = np.roll(P, 1, axis=0)
    PS[0] = P[0]
    return np.linalg.norm(P - PS, axis=1)


def polyline_length(P):
    """
    Calculates the total length of the given polyline.
    """
    return np.sum(polyline_segment_length(P))


def polyline_distance_to_point(P, x):
    """
    Calculates the distance each point in a polyline P to a reference point x.
    """
    return np.linalg.norm(np.atleast_2d(P) - np.atleast_1d(x)[None, :], axis=1)


def subdivide_ray(p, d, l, max_dist):
    """
    Builds a polyline representing a ray originating at point p with direction
    vector d, length l and a maximum distance between points of max_dist.
    """
    return p + np.linspace(0, 1, np.ceil(l / max_dist) + 1)[:, None] * d * l


def line_segment_intersection(p, d, l, p3, p4):
    """
    Checks whether two line segment with points (p, p + d * l) and (p3, p4)
    intersect, where d is a unit-length vector and l is a scalar. The weird
    calling convention is due to this function being used in the collision
    detection code.
    """

    # Calculate num, den from equation 4.2
    num = (p4[0] - p3[0]) * (p[1] - p3[1]) - (p4[1] - p3[1]) * (p[0] - p3[0])
    den = (p4[1] - p3[1]) * (d[0]) - (p4[0] - p3[0]) * (d[1])

    # Check for special cases
    if num == 0 and den == 0:
        # Lines are coincident. Check whether they actually intersect by
        # projecting p1, p2, p3, p4 on the common direction vector v
        alpha1 = 0
        alpha2 = l
        alpha3 = np.dot(p3 - p, d)
        alpha4 = np.dot(p4 - p, d)

        if alpha2 < alpha1:
            alpha1, alpha2 = alpha2, alpha1
        if alpha4 < alpha3:
            alpha3, alpha4 = alpha4, alpha3

        i1 = max(alpha1, alpha3)
        i2 = min(alpha2, alpha4)
        if i2 < i1:
            return np.inf
        else:
            return i1 * l
    if num != 0 and den == 0:
        return np.inf

    # Make sure sa, sb is in [0, 1]
    sa = num / den
    if p4[0] - p3[0] != 0:
        sb = ((p[0] - p3[0]) + sa * d[0]) / (p4[0] - p3[0])
    else:
        sb = ((p[1] - p3[1]) + sa * d[1]) / (p4[1] - p3[1])
    if sa < 0 or sa > l or sb < 0 or sb > 1:
        return np.inf

    # Return sa
    return sa


class CollisionQueryEngine:
    def __init__(self, obstacles, radius=0.25):
        """
        Constructor of the CollisionQueryEngine class. Builds the KDTree index
        for the given list of polygonal obstacles
        """

        # Subdivide the obstacles up to the given resolution and build an index
        # mapping from all polygon points to the polygon index and original
        # polygon vertex
        self.radius = radius
        self.obstacles = list(map(np.atleast_2d, obstacles))
        self.point_index = []
        self.points = []
        for i, obstacle in enumerate(self.obstacles):
            pnts, idcs = subdivide_polygon(obstacle, self.radius)
            self.points.extend(pnts)
            self.point_index.extend(zip([i] * len(pnts), idcs))
        self.points = np.atleast_2d(self.points)
        self.point_index = np.array(self.point_index, dtype=np.int32)

        # Build a KDTree for the points
        self.tree = None
        if self.points.size > 0:
            self.tree = cKDTree(self.points)

            # Used in the "point in polygon" calculation
            self.bounds_min_x = np.min(self.points[:, 0]) - 3
            self.bounds_min_y = np.min(self.points[:, 1]) - 5

    def check_collision(self, x, y, dx, dy):
        """
        Sends a ray (dx, dy) from the point (x, y) and checks for collision with
        any of the polygons.
        """

        # Special case if there are no obstacle points
        if self.tree is None:
            return False, np.inf, None

        # Subdivide the ray to the desired resolution
        p = np.array((x, y))
        l = np.hypot(dx, dy)
        d = np.array((dx, dy)) / l
        ray = subdivide_ray(p, d, l, self.radius)

        # March along the ray and find near obstacles
        ray_idcs = self.tree.query_ball_point(ray, self.radius)
        min_dist = np.inf
        cur_dir = None
        visited = set()
        for idcs in ray_idcs:
            for idx in idcs:
                # Fetch the polygon and vertex belonging to the candidate point
                polygon_idx, vertex_idx = self.point_index[idx]
                if (polygon_idx, vertex_idx) in visited:
                    continue
                visited.add((polygon_idx, vertex_idx))
                P = self.obstacles[polygon_idx]
                N = len(P)

                # Fetch the two line segments belonging to the polygon vertex
                p21 = P[vertex_idx]
                p22a = P[(N + vertex_idx - 1) % N]
                p22b = P[(vertex_idx + 1) % N]

                # Perform line segment intersections between the test ray and
                # the two segments, test whether the intersection with the ray
                # is closer than any other intersection so far
                dA = line_segment_intersection(p, d, l, p21, p22a)
                dB = line_segment_intersection(p, d, l, p21, p22b)
                if dA < min_dist:
                    min_dist = dA
                    cur_dir = p21 - p22a
                if dB < min_dist:
                    min_dist = dB
                    cur_dir = p21 - p22b

            # If we’ve found an intersetction we can abort here. We are marching
            # along the ray. Closer obstacles cannot be found.
            if min_dist < np.inf:
                break

        # Normalise the direction vector
        if not cur_dir is None:
            cur_dir = cur_dir / np.linalg.norm(cur_dir)

        return min_dist < np.inf, min_dist, cur_dir

    def point_in_polygon(self, x, y):
        """
        Returns true if the given point is inside of any of the polygons.
        """

        import matplotlib.pyplot as plt

        # Special case if there are no obstacle points
        if self.tree is None:
            return False

        # Create the test ray
        p = np.array((x, y))
        d = np.array((self.bounds_min_x - x, self.bounds_min_y - y))
        l = np.hypot(d[0], d[1])
        d = d / l
        ray = subdivide_ray(p, d, l, self.radius)

        # Number of intersections
        count = 0

        # March along the ray and find near obstacles
        ray_idcs = self.tree.query_ball_point(ray, self.radius)
        visited = set()
        distances, resolution = set(), 1e6
        for idcs in ray_idcs:
            for idx in idcs:
                # Fetch the polygon and vertex belonging to the candidate point
                polygon_idx, vertex_idx = self.point_index[idx]
                if (polygon_idx, vertex_idx) in visited:
                    continue
                visited.add((polygon_idx, vertex_idx))
                P = self.obstacles[polygon_idx]
                N = len(P)

                # Fetch the two line segments belonging to the polygon vertex
                p21 = P[vertex_idx]
                p22a = P[(N + vertex_idx - 1) % N]
                p22b = P[(vertex_idx + 1) % N]

                # Perform line segment intersections between the test ray and
                # the two segments, test whether the intersection with the ray
                # is closer than any other intersection so far
                dA = line_segment_intersection(p, d, l, p21, p22a)
                dB = line_segment_intersection(p, d, l, p21, p22b)
                if dA < np.inf:
                    k = int(dA * resolution)
                    if not k in distances:
                        distances.add(k)
                        count += 1
                if dB < np.inf:
                    k = int(dB * resolution)
                    if not k in distances:
                        distances.add(k)
                        count += 1

        # The point is in a polygon if there is an odd number of intersections
        return count % 2 == 1


if __name__ == "__main__":
    assert distance_point_to_line([0, 0], [-1, 1], [1, 1]) == 1
    assert distance_point_to_line([0, 1], [-1, 1], [1, 1]) == 0

    assert np.isclose(circle_line_collision(0, 1, 1, 0, 0.1, 10), 9.9)
    assert np.isclose(circle_line_collision(0, 1, -1, 0, 0.1, 10), 9.9)
    assert np.isclose(circle_line_collision(0, -1, 1, 0, 0.1, 10), 9.9)
    assert np.isclose(circle_line_collision(0, -1, -1, 0, 0.1, 10), 9.9)

    assert np.isclose(circle_line_collision(1, 0, 0, 1, 0.1, 10), 9.9)
    assert np.isclose(circle_line_collision(1, 0, 0, -1, 0.1, 10), 9.9)
    assert np.isclose(circle_line_collision(-1, 0, 0, 1, 0.1, 10), 9.9)
    assert np.isclose(circle_line_collision(-1, 0, 0, -1, 0.1, 10), 9.9)

    os2 = 1 / np.sqrt(2)
    assert np.isclose(
        circle_line_collision(os2, os2, 0, 1, 0.1, 1), 1 - 0.1 * np.sqrt(2))
    assert np.isclose(
        circle_line_collision(os2, os2, 0, -1, 0.1, 1), 1 - 0.1 * np.sqrt(2))
    assert np.isclose(
        circle_line_collision(os2, os2, 1, 0, 0.1, 1), 1 - 0.1 * np.sqrt(2))
    assert np.isclose(
        circle_line_collision(os2, os2, -1, 0, 0.1, 1), 1 - 0.1 * np.sqrt(2))

    P1 = np.array([[0, 0], [0, 1], [1, 0]])
    assert np.all(simplify_polygon(P1) == P1)

    P2 = np.array([[0, 0], [0, 0.5], [0, 1], [1, 0], [0.75, 0], [0.5, 0]])
    assert np.all(simplify_polygon(P2) == P1)

    P3 = subdivide_polygon(P1, 0.1)[0]  #
    assert np.all(simplify_polygon(P3) == P1)

    e1 = CollisionQueryEngine([[(1, -1), (1, 1), (-1, 1), (-1, -1)]])

    has_collision, dist, direction = e1.check_collision(-2, 0, 10, 0)
    assert has_collision == True
    assert dist == 1.0
    assert np.all(direction == [0, 1])

    has_collision, dist, direction = e1.check_collision(-2, 0, 1, 0)
    assert has_collision == True
    assert dist == 1.0
    assert np.all(direction == [0, 1])

    has_collision, dist, direction = e1.check_collision(-2, 0, 0.9, 0)
    assert has_collision == False
    assert dist == np.inf
    assert direction == None

    has_collision, dist, direction = e1.check_collision(-2, 0, 2, 2)
    assert has_collision == True
    assert np.isclose(dist, np.sqrt(2))

    has_collision, dist, direction = e1.check_collision(-1, 0, 2, 2)
    assert has_collision == True
    assert dist == 0

    has_collision, dist, direction = e1.check_collision(-2, 0, 2, 3)
    assert has_collision == False

    has_collision, dist, direction = e1.check_collision(-2, 0, 3, 2)
    assert has_collision == True

    eps = 1e-9
    assert e1.point_in_polygon(0, 0) == True
    assert e1.point_in_polygon(2, 2) == False
    assert e1.point_in_polygon(-2, -2) == False
    assert e1.point_in_polygon(-1 + eps, 0) == True
    assert e1.point_in_polygon(0, 1 - eps) == True
    assert e1.point_in_polygon(0, -1 + eps) == True

    assert np.all(polyline_segment_length([[1], [2], [3]]) == [0, 1, 1])
    assert np.all(polyline_segment_length([[1, 2], [2, 2], [3, 2]]) == [0, 1, 1])
    assert polyline_length([[1], [2], [3]]) == 2

    #
    # Visualise the CollisionQueryEngine
    #

    import matplotlib.pyplot as plt
    import ascii_map
    _, _, obstacles = ascii_map.parse_ascii_map(ascii_map.test_map, scale=1.0)
    e2 = CollisionQueryEngine(obstacles)

    fig, ax = plt.subplots()

    for obstacle in obstacles:
        ax.add_artist(
            plt.Polygon(obstacle, linewidth=1, fill=False, color='g'))

    for _ in range(1000):
        x, y = np.random.uniform(0, 40, 2)
        dx, dy = np.random.uniform(-5, 5, 2)

        ax.plot([x, x + dx], [y, y + dy], linewidth=0.5, alpha=0.2, color='k')
        has_collision, dist, direction = e2.check_collision(x, y, dx, dy)
        if has_collision:
            scale = 1 / np.hypot(dx, dy)
            ox, oy = x + dx * dist * scale, y + dy * dist * scale
            ax.plot([ox], [oy], 'o', markersize=2, color='k')
            ax.plot([x, ox], [y, oy], linewidth=0.5, color='b')
            ax.plot(
                [ox, ox + direction[0]], [oy, oy + direction[1]],
                linewidth=1,
                color='r')

    plt.savefig("geometry_test.pdf", format="pdf", bbox_inches="tight")

