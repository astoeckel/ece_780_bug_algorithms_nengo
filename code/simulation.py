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
# simulation.py
#
# This is the main program which conducts a simulation and writes all state
# variables to a Pickle file. Visualisation and evaluation of the trajectories are
# performed by external programs.
#

import sys, time, math


def run_simulation(start,
                   goal,
                   obstacles,
                   agent_class,
                   T=100.0,
                   dt_coarse=1e-2,
                   dt_fine=1e-3,
                   theta_offs=0,
                   seed=None,
                   show_progress=True):
    from environment import Environment

    # Draw a random seed if none was given
    if seed is None:
        seed = np.random.randint(np.iinfo(np.int32))

    # If we do not have a goal, set a dummy goal
    has_goal = not goal is None
    if not has_goal:
        goal = (0, 0)

    # Instantiate the agent, let the agent point at the target location
    x0, y0 = start
    theta0 = ((math.atan2(goal[1] - y0, goal[0] - x0)
               if has_goal else 0) + theta_offs)
    agent = agent_class(
        goal, x0, y0, theta0, dt_coarse=dt_coarse, dt_fine=dt_fine, seed=seed)

    # Instantiate the environment instance
    environment = Environment(start, goal, obstacles)

    # Simulation state
    t, i = 0, 0
    trajectory = []
    t_last_progress = 0
    while t < T:
        # Update the agent
        agent.update(environment, dt_coarse, dt_fine)

        # Abort if the agent reached the goal
        if has_goal and agent.reached_goal:
            break

        # Advance the time
        t += dt_coarse
        i += 1

        # Print the progress bar
        if show_progress:
            t_progress = time.clock()
            if t_progress - t_last_progress > 200e-3 or i > 100:
                t_last_progress = t_progress
                i = 0
                sys.stderr.write("Simulating: {:0.2f}%        \r".format(
                    t / (T - dt_coarse) * 100))
                sys.stderr.flush()
    if show_progress:
        print("Done.                  ")

    # Return the agent instance
    return agent


def classloader(name):
    """
    Loads an agent class with the given name from the "agents" subdirectory.
    """
    import os, sys, importlib, importlib.util

    # Make sure both the current script directory and the agent directory are in
    # the searchpath
    this_dir = os.path.dirname(os.path.realpath(__file__))
    agents_dir = os.path.join(this_dir, "agents")
    if not this_dir in sys.path:
        sys.path = [this_dir] + sys.path
    if not agents_dir in sys.path:
        sys.path = [agents_dir] + sys.path

    # Load the class from the agents subdirectory
    class_ = None
    for filename in os.listdir(agents_dir):
        if filename.endswith(".py"):
            module_name = filename[:-3]
            if module_name in sys.modules.keys():
                # Reload the module if it has already been loaded
                module = sys.modules[module_name]
                importlib.reload(module)
            else:
                # Build the module specification for the py file
                spec = importlib.util.spec_from_file_location(
                    module_name, os.path.join(agents_dir, filename))

                # Load and execute the module
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Make sure the module is registered
                sys.modules[module_name] = module

            # Search for a class with the given class name
            if hasattr(module, name):
                if not class_ is None:
                    raise Exception("Class name \"" + name +
                                    "\" is ambiguous!")
                class_ = getattr(module, name)
    if class_ is None:
        raise Exception("Class not found, \"" + name + "\".")

    return class_


#
# Main program
#

if __name__ == "__main__":
    import os, itertools
    import multiprocessing
    import argparse
    import ascii_map
    import pickle

    parser = argparse.ArgumentParser(
        description="Neural Bug Algorithm Implementation Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--agent",
        type=str,
        help=
        "Name of the agent class that should be instantiated. A class with a "
        + "corresponding name must be located in the 'agents' subfolder.",
        required=True)
    parser.add_argument(
        "--map", type=str, help="Name of the ASCII map file", required=True)
    parser.add_argument(
        "--tar", type=str, help="Output filename prefix", default="./out/sim_")
    parser.add_argument(
        "--repeat",
        type=int,
        help="Number of times the experiment should be repeated",
        default=1)
    parser.add_argument(
        "--seed",
        type=int,
        help="The base seed allowing to reproduce the experiment. Base seed is "
        + "incremented by one for each experiment repetition.",
        default=48264)
    parser.add_argument(
        "--T",
        type=float,
        help="Maximum simulation time in seconds. Simulation will be aborted "
        + "earlier once the goal is reached.",
        default=100.0)
    parser.add_argument(
        "--dt-coarse",
        type=float,
        help="Robot simulator timestep.",
        default=1e-2)
    parser.add_argument(
        "--dt-fine",
        type=float,
        help="Neuronal network simulator timestep.",
        default=1e-3)
    parser.add_argument(
        "--theta-offs",
        type=float,
        help="Initial robot orientation offset in degrees.",
        default=0)

    args = parser.parse_args()

    # Create the output directory if it does not exist
    outdir = os.path.dirname(args.tar)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # Load the agent class
    agent_class = classloader(args.agent)

    # Load the map file
    if not os.path.isfile(args.map):
        raise Exception("Map file \"" + args.map + "\" does not exist.")
    with open(args.map, 'r') as f:
        start, goal, obstacles = ascii_map.parse_ascii_map(f.read())

    # Insert a dummy goal if none was given
    if goal.size == 0:
        goal = [None]

    # Generate a list with all start/goal pairs
    experiments = list(
        itertools.product(
            range(args.repeat), range(len(start)), range(len(goal))))

    def thread(experiment):
        """
        This function is executed for each start/goal point and each repetition.
        The functions are executed from a multiprocessing process pool.
        """
        # Extract the experiment description
        seed = args.seed + experiment[0]
        start_pnt = start[experiment[1]]
        goal_pnt = goal[experiment[2]]

        # Run the simulation
        agent = run_simulation(start_pnt, goal_pnt, obstacles, agent_class,
                               args.T, args.dt_coarse, args.dt_fine,
                               args.theta_offs / 180 * math.pi,
                               args.seed + seed)

        # Write the recorded data to the target file. Add provenance data to be
        # able to reconstruct the exact experiment that caused the result.
        map_name = os.path.basename(args.map).split(".", 2)[0]
        tar_filename = (
            args.tar + map_name.lower() + "_" + args.agent.lower() + "_" +
            "_".join(map(lambda i: "{:02d}".format(i), experiment)) + ".pcl")
        with open(tar_filename, "wb") as f:
            pickle.dump({
                "seed":
                seed,
                "start":
                start_pnt,
                "goal":
                goal_pnt,
                "repeat_idx":
                experiment[0],
                "start_idx":
                experiment[1],
                "goal_idx":
                experiment[2],
                "reached_goal":
                agent.reached_goal,
                "obstacles":
                obstacles,
                "trajectory":
                agent.trajectory(),
                "trace":
                agent.trace(),
                "radius":
                agent.radius,
                "map_file":
                args.map,
                "map_name":
                map_name,
                "agent_class":
                args.agent,
                "agent_name":
                agent.name(),
                "color":
                agent.color(),
                "trace_description":
                agent.trace_description(),
                "T":
                args.T,
                "duration":
                max(agent.trajectory()[:, 0]),
                "dt_coarse":
                args.dt_coarse,
                "dt_fine":
                args.dt_fine
            }, f, -1)

    # Create a process pool and run the thread function for each experiment
    # descriptor in the experiments list
    if len(experiments) == 1:
        thread(experiments[0])  # Run in main thread if only one experiment.
    else:
        pool = multiprocessing.Pool()
        pool.map(thread, experiments)

