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
"""
This program takes a set of *.pcl files as produced by simulation.py and draws
the map and robot trajectory for each run.
"""

import numpy as np
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

import geometry


def rgba(c):
    return np.array(matplotlib.colors.to_rgba(c))


def hexcolor(c):
    return ''.join("{:02X}".format(int(v * 255)) for v in rgba(c))


def plot_experiments(name,
                     map_data,
                     agent_runs,
                     max_t=np.inf,
                     plot_end_location=True,
                     plot_shortest_path=True):
    fig, ax = plt.subplots(figsize=(5.25, 5.25))

    # Draw the map
    start, goal, obstacles = map_data.start, map_data.goal, map_data.obstacles
    min_x, min_y = np.inf, np.inf
    max_x, max_y = -np.inf, -np.inf

    # Plot all obstacles
    for obstacle in obstacles:
        # Extend the map boundaries
        obstacle = np.array(obstacle)
        min_x = min(np.min(obstacle[:, 0]), min_x)
        min_y = min(np.min(obstacle[:, 1]), min_y)
        max_x = max(np.max(obstacle[:, 0]), max_x)
        max_y = max(np.max(obstacle[:, 1]), max_y)

        # Plot the obstacle
        ax.add_artist(plt.Polygon(obstacle, fill=False, color='k', zorder=0))

    # Plot all start and end points
    for s in start.values():
        min_x, min_y = min(s[0], min_x), min(s[1], min_y)
        max_x, max_y = max(s[0], max_x), max(s[1], max_y)
        ax.plot(s[0], s[1], "x", color="k", markersize=7, zorder=3)
    for g in goal.values():
        min_x, min_y = min(g[0], min_x), min(g[1], min_y)
        max_x, max_y = max(g[0], max_x), max(g[1], max_y)
        ax.plot(g[0], g[1], "+", color="k", markersize=7, zorder=3)

    # Plot shortest paths
    if plot_shortest_path:
        for path_per_start in map_data.shortest_paths.values():
            for path in path_per_start.values():
                ax.plot(
                    path[:, 0],
                    path[:, 1],
                    linewidth=0.5,
                    linestyle=(0, (0.25, 0.25)),
                    color=[0.5, 0.5, 0.5],
                    zorder=-1)

    # Iterate over all agents and experiments
    for i, key in enumerate(agent_runs.keys()):
        n = len(agent_runs[key])
        vs = np.linspace(1.0, 0.5, n)
        for j, data in enumerate(agent_runs[key]):
            colour = rgba(data.color) * vs[j] + np.array([1.0, 1.0, 1.0, 1.0
                                                          ]) * (1 - vs[j])

            # Fetch the data, restrict to the specified max_t
            ts, x, y, theta = np.array(data.trajectory)[:, 0:4].T
            mask = ts < max_t
            if not np.any(mask):
                continue
            ts, x, y, theta = ts[mask], x[mask], y[mask], theta[mask]

            # Draw the base trajectory
            ax.plot(x, y, color=colour, linewidth=0.75, zorder=1)

            # Draw direction arrows every 1s, except for the last 0.5 seconds
            ts_mark = np.arange(0, max(ts), 1.0)
            al = 0.005  # Arrow length
            for t in ts_mark:
                t_idx = np.argmin(np.abs(ts - t))
                ax.arrow(
                    x[t_idx] - np.cos(theta[t_idx]) * al,
                    y[t_idx] - np.cos(theta[t_idx]) * al,
                    np.cos(theta[t_idx]) * al,
                    np.sin(theta[t_idx]) * al,
                    head_width=0.125,
                    facecolor=colour * np.array([1, 1, 1,
                                                 min(1, max_t - t)]),
                    edgecolor=colour * np.array([1, 1, 1, 0]),
                    zorder=4)

            if plot_end_location:
                ax.add_artist(
                    plt.Circle(
                        (x[-1], y[-1]),
                        data.radius,
                        color=colour,
                        fill=False,
                        zorder=5))
                ax.arrow(
                    x[-1] - np.cos(theta[-1]) * al,
                    y[-1] - np.sin(theta[-1]) * al,
                    np.cos(theta[-1]) * al,
                    np.sin(theta[-1]) * al,
                    head_width=0.125,
                    facecolor=colour,
                    edgecolor=colour * np.array([1, 1, 1, 0]),
                    zorder=5)

    # Add legend keys
    legend_handles = [
        mlines.Line2D(
            [], [],
            linewidth=0,
            marker="x",
            markersize=7,
            color="k",
            label="Start"),
        mlines.Line2D(
            [], [],
            linewidth=0,
            marker="+",
            markersize=7,
            color="k",
            label="Goal")
    ]
    for i, key in enumerate(agent_runs):
        legend_handles.append(
            mpatches.Patch(
                color=agent_runs[key][0].color,
                label=agent_runs[key][0].agent_name))
    ax.legend(
        ncol=4,
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.025))

    # Set the plot boundaries
    if (max_x - min_x) < 2:
        min_x -= 1
        max_x += 1
    if (max_y - min_y) < 2:
        min_y -= 1
        max_y += 1
    ax.set_xlim(
        np.floor(min_x - (max_x - min_x) * 0.1),
        np.ceil(max_x + (max_x - min_x) * 0.1))
    ax.set_ylim(
        np.floor(min_y - (max_y - min_y) * 0.1),
        np.ceil(max_y + (max_y - min_y) * 0.1))
    ax.set_aspect(1)
    ax.set_xlabel(r"$x$-location $[\mathrm{m}]$")
    ax.set_ylabel(r"$y$-location $[\mathrm{m}]$")

    return fig


def plot_distance(name,
                  map_data,
                  agent_runs,
                  plot_shortest_path=True,
                  over_time=False):
    fig, ax = plt.subplots(figsize=(5.25, 2))

    goal, obstacles = map_data.goal, map_data.obstacles

    # Iterate over all agents and experiments
    for i, key in enumerate(agent_runs.keys()):
        n = len(agent_runs[key])
        vs = np.linspace(1.0, 0.5, n)
        for j, data in enumerate(agent_runs[key]):
            colour = rgba(data.color) * vs[j] + np.array([1.0, 1.0, 1.0, 1.0
                                                          ]) * (1 - vs[j])

            trajectory = np.asarray(data.trajectory, dtype=np.float64)
            pnts = trajectory[:, 1:3]
            if over_time:
                # Time
                xs = trajectory[:, 0]
            else:
                # Distance travelled
                xs = np.cumsum(geometry.polyline_segment_length(pnts))
            dist_to_goal = geometry.polyline_distance_to_point(pnts, data.goal)
            if data.reached_goal:
                dist_to_goal[-1] = 0

            ax.plot(xs, dist_to_goal, color=colour, linewidth=0.75)

    # Plot the reference data
    # Plot shortest paths
    if plot_shortest_path and not over_time:
        for path_per_start in map_data.shortest_paths.values():
            for goal_idx, path in path_per_start.items():
                dist_traveled = np.cumsum(
                    geometry.polyline_segment_length(path))
                dist_to_goal = geometry.polyline_distance_to_point(
                    path, map_data.goal[goal_idx])
                ax.plot(
                    dist_traveled,
                    dist_to_goal,
                    linewidth=0.5,
                    linestyle=(0, (0.25, 0.25)),
                    color=[0.5, 0.5, 0.5],
                    zorder=-1)

    # Add legend keys
    legend_handles = []
    for i, key in enumerate(agent_runs):
        legend_handles.append(
            mpatches.Patch(
                color=agent_runs[key][0].color,
                label=agent_runs[key][0].agent_name))
    ax.legend(
        ncol=4,
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.025))
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    if over_time:
        ax.set_xlabel("Time $t$ [s]")
    else:
        ax.set_xlabel("Distance travelled [m]")
    ax.set_ylabel("Distance to goal [m]")

    return fig


def plot_trace(name, map_data, agent_runs, trace_ids=None):

    trace_description = agent_runs[0].trace_description
    if not trace_ids is None:
        trace_description = list(
            filter(lambda x: x.id in trace_ids, trace_description))

    N = len(trace_description)

    fig, axarr = plt.subplots(N, figsize=(5.25, 4), sharex=True)
    if N == 1:
        axarr = [axarr]
    min_x, max_x = np.inf, -np.inf
    for i, trace in enumerate(trace_description):
        ax = axarr[i]
        ax.set_ylabel(trace.name)
        if i == N - 1:
            ax.set_xlabel("Time $t$ [s]")
        n = len(agent_runs)
        vs = np.linspace(1.0, 0.5, n)
        if isinstance(trace.dtype, list):
            ax.set_yticks(range(len(trace.dtype)))
            ax.set_yticklabels(trace.dtype)
        for j, data in enumerate(agent_runs):
            colour = rgba(data.color) * vs[j] + np.array([1.0, 1.0, 1.0, 1.0
                                                          ]) * (1 - vs[j])

            ax.plot(data.trace[:, 0], data.trace[:, i + 1], color=colour)

            min_x = min(min_x, np.min(data.trace[:, 0]))
            max_x = max(max_x, np.max(data.trace[:, 0]))

    # Add legend keys
    legend_handles = [
        mpatches.Patch(
            color=agent_runs[0].color, label=agent_runs[0].agent_name)
    ]
    axarr[0].legend(
        ncol=4,
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.025))

    for i, _ in enumerate(trace_description):
        axarr[i].set_xlim(min_x, max_x)

    return fig


if __name__ == "__main__":
    import sys, os
    import multiprocessing
    import argparse
    import analysis

    def str2bool(v):
        # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(
        description="Neural Bug Algorithm Implementation Visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'files',
        help="List of *.pcl files generated by the simulator",
        nargs='+')
    parser.add_argument(
        "--tar", type=str, help="Output filename prefix", default="./out/vis_")
    parser.add_argument(
        "-B",
        action='store_true',
        help="Ignore times, force building",
        default=False)
    parser.add_argument(
        "--animate",
        action='store_true',
        help="Store a MP4 video showing the animation (requires ffmpeg)",
        default=False)
    parser.add_argument(
        "--rate",
        type=float,
        help="Frame rate of the animation.",
        default=30.0)
    parser.add_argument(
        "--plot-trajectory",
        type=str2bool,
        help="Plots the robot trajectory.",
        default=True)
    parser.add_argument(
        "--plot-distance",
        type=str2bool,
        help="Plots the distance to the goal over the distance traveled",
        default=True)
    parser.add_argument(
        "--plot-trace",
        type=str2bool,
        help=
        "If true, plots the agent trace data. Separate plots for each agent class",
        default=True)
    parser.add_argument(
        "--plot-shortest-path",
        type=str2bool,
        help="If true, plots reference shortest path data",
        default=True)
    parser.add_argument(
        "--bgcolor",
        type=str,
        help=
        "Background colour to be used when rendering a video. Leave empty for white.",
        default="")
    args = parser.parse_args()

    # Load the data
    maps, map_data = analysis.process_simulation_files(args.files,
                                                       args.plot_shortest_path)

    def skip_file(fn, ref_mtime):
        mtime = os.stat(
            fn).st_mtime if not args.B and os.path.isfile(fn) else 0
        if mtime > ref_mtime:
            print("Skipping " + fn + ", already up-to-date.")
            return True
        return False

    def thread_trajectory(map_name):
        # Generate the output filename, check whether the file is up-to-date
        ext = ".mp4" if args.animate else ".pdf"
        fn = (args.tar + "traj_" + map_name.lower() + "_" +
              "_".join(map(lambda s: s.lower(), maps[map_name].keys())) + ext)
        if skip_file(fn, map_data[map_name].mtime):
            return

        if not args.animate:
            print("Plotting " + fn)
            fig = plot_experiments(
                map_name,
                map_data[map_name],
                maps[map_name],
                plot_shortest_path=args.plot_shortest_path)

            print("Saving " + fn)
            fig.savefig(
                fn, format='pdf', bbox_inches='tight', transparent=True)
            plt.close(fig)
        else:
            from subprocess import Popen, PIPE

            # Calculate the number of frames and the individual times
            rate = args.rate
            duration = map_data[map_name].duration
            frames = int(duration * rate)
            ts = np.arange(0, duration, 1.0 / rate)

            print("Animating " + fn + ", " + str(frames) + " frames")

            # Open ffmpeg with suitable parameters. The funny "scale" expression
            # makes sure the matplotlib output is scaled down to Full HD
            # resolution and centred, with white border. "-tune animation" tunes
            # the libx264 for comic rendering. "-crf 18" sets the desired
            # quality, where "18" is, according to the documentation, "visually"
            # lossless.
            p = Popen(
                [
                    'ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'mjpeg',
                    '-r',
                    str(rate), '-i', '-', '-vf',
                    'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:'
                    + ('white'
                       if args.bgcolor == "" else hexcolor(args.bgcolor)),
                    '-c:v', 'libx264', '-tune', 'animation', '-crf', '18',
                    '-pix_fmt', 'yuv420p', '-r',
                    str(rate), fn
                ],
                stdin=PIPE)

            # Render each frame individual, encode the images as lossless JPEGs
            # and pipe into the ffmpeg subprocess
            for i, max_t in enumerate(ts):
                fig = plot_experiments(
                    map_name,
                    map_data[map_name],
                    maps[map_name],
                    max_t=max_t,
                    plot_end_location=True,
                    plot_shortest_path=args.plot_shortest_path)
                if args.bgcolor != "":
                    fig.set_facecolor(args.bgcolor)
                    fig.gca().set_facecolor(args.bgcolor)
                fig.savefig(
                    p.stdin,
                    format='jpg',
                    dpi=400,
                    quality=100,  # No quantisation
                    subsampling='4:4:4',  # Disable color subsampling
                    facecolor=fig.get_facecolor(),
                    transparent=True,
                    bbox_inches='tight')
                plt.close(fig)

            # Indicate that the stream is at an end and wait for ffmpeg to be
            # done
            p.stdin.close()
            p.wait()

    def thread_distance(map_name, over_time=False):
        # Generate the output filename, check whether the file is up-to-date
        fn = (
            args.tar + ("dist_t_"
                        if over_time else "dist_") + map_name.lower() + "_" +
            "_".join(map(lambda s: s.lower(), maps[map_name].keys())) + ".pdf")
        if skip_file(fn, map_data[map_name].mtime):
            return

        if len(map_data[map_name].goal) == 0:
            return

        print("Plotting " + fn)
        fig = plot_distance(
            map_name,
            map_data[map_name],
            maps[map_name],
            plot_shortest_path=args.plot_shortest_path,
            over_time=over_time)

        print("Saving " + fn)
        fig.savefig(fn, format='pdf', bbox_inches='tight', transparent=True)
        plt.close(fig)

    def thread_trace(map_name, over_time=False):
        for i, key in enumerate(maps[map_name].keys()):
            # Skip agents with empty trace description
            if len(maps[map_name][key][0].trace_description) == 0:
                continue

            # Generate the output filename, check whether the file is up-to-date
            fn = (args.tar + "trace_" + map_name.lower() + "_" + key.lower() +
                  ".pdf")
            if skip_file(fn, map_data[map_name].mtime):
                continue

            print("Plotting " + fn)
            fig = plot_trace(map_name, map_data[map_name], maps[map_name][key])

            print("Saving " + fn)
            fig.savefig(fn, format='pdf', bbox_inches='tight', transparent=True)
            plt.close(fig)


    def thread_distance_over_time(map_name):
        thread_distance(map_name, True)

    # Plot all files in parallel
    pool = multiprocessing.Pool()
    tasks = []
    if args.plot_trajectory:
        tasks.append(pool.map_async(thread_trajectory, map_data.keys()))
    if args.plot_distance:
        tasks.append(pool.map_async(thread_distance, map_data.keys()))
        tasks.append(pool.map_async(thread_distance_over_time, map_data.keys()))
    if args.plot_trace:
        tasks.append(pool.map_async(thread_trace, map_data.keys()))

    # Wait for all tasks to complete
    for task in tasks:
        if hasattr(task, "wait"):
            task.wait()
