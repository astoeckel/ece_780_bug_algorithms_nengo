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

from neural_agent import NeuralAgent
import common_neural

import nengo
import numpy as np

STATE_INIT_DIR_MEMORY = 0
STATE_GO_TOWARDS_GOAL = 1
STATE_INIT_DIST_MEMORY = 2
STATE_FOLLOW_OBSTACLE = 3


class Bug_2_neural(NeuralAgent):
    """
    Neural implementation of the Bug 2 algorithm. Closely follows the reference
    implementation in so far as a four-state state machine is implemented,
    where the first state is used to store the initial location of the goal, the
    second state to drive towards the goal, the third state to store the
    distance to the goal in memory and the fourth state to follow the obstacle
    outline.
    """

    def name(self):
        return "Bug 2 (Neural)"

    def color(self):
        return "#5c3566"

    def init(self):
        with self.network:
            # Integrate the hit_obstacle signal with slight decay
            ens_hit_obstacle = nengo.Ensemble(
                n_neurons=100,
                dimensions=1,
                seed=self.seed_gen(),
                label="ens_hit_obstacle")
            nengo.Connection(
                ens_hit_obstacle,
                ens_hit_obstacle,
                transform = 0.9,
                synapse=100e-3)
            nengo.Connection(
                self.node_hit_obstacle, ens_hit_obstacle, transform=0.25)

            # Represent the compass direction in its own ensemble
            ens_compass = nengo.Ensemble(
                n_neurons=100,
                dimensions=2,
                label="ens_compass",
                seed=self.seed_gen())
            nengo.Connection(self.node_compass, ens_compass)

            # Calculate the follow obstacle gradient using the follow obstacle
            # subnetwork
            net_follow_obstacle = common_neural.make_follow_obstacle_network(
                self.seed_gen, self.sensors.radar_range,
                self.sensors.radar_vectors, self.radius)

            # Feed both the movement direction from ens_compass and the movement
            # direction from net_follow_obstacle into the motor output
            nengo.Connection(ens_compass, self.node_motor)
            nengo.Connection(self.node_radar_depth,
                             net_follow_obstacle.radar_depth)
            nengo.Connection(net_follow_obstacle.motor, self.node_motor)

            # Memories storing the x and the y direction towards the goal at the
            # initial robot location
            tau_int = 100e-3
            ens_memory_dir = nengo.networks.EnsembleArray(
                n_neurons=200,
                n_ensembles=2,
                label="ens_memory_dir",
                seed=self.seed_gen())
            ens_memory_dir_diff = nengo.networks.EnsembleArray(
                n_neurons=100,
                n_ensembles=2,
                label="ens_memory_dir_diff",
                seed=self.seed_gen())
            ens_memory_dir_init = nengo.Ensemble(
                n_neurons=100,
                dimensions=1,
                label="ens_memory_dir_init",
                seed=self.seed_gen())
            for i in range(2):
                nengo.Connection(
                    ens_memory_dir.ensembles[i],
                    ens_memory_dir.ensembles[i],
                    synapse=tau_int)
                nengo.Connection(self.node_absolute_compass[i],
                                 ens_memory_dir_diff.ensembles[i])
                nengo.Connection(
                    ens_memory_dir.ensembles[i],
                    ens_memory_dir_diff.ensembles[i],
                    transform=-1)
                nengo.Connection(
                    ens_memory_dir_diff.ensembles[i],
                    ens_memory_dir.ensembles[i],
                    transform=4 * tau_int)
                nengo.Connection(
                    ens_memory_dir_diff.ensembles[i],
                    ens_memory_dir_init,
                    synapse=tau_int,
                    function=lambda x: np.abs(x))

            # Memory to store the distance to the goal once an obstacle is hit
            ens_memory_dist = nengo.Ensemble(
                n_neurons=300,
                dimensions=1,
                label="ens_memory_dist",
                seed=self.seed_gen())
            ens_memory_dist_diff = nengo.Ensemble(
                n_neurons=100,
                dimensions=1,
                label="ens_memory_dist_diff",
                seed=self.seed_gen())
            nengo.Connection(ens_memory_dist, ens_memory_dist, synapse=tau_int)
            nengo.Connection(self.node_distance_to_goal, ens_memory_dist_diff)
            nengo.Connection(
                ens_memory_dist, ens_memory_dist_diff, transform=-1)
            nengo.Connection(
                ens_memory_dist_diff, ens_memory_dist, transform=4 * tau_int)

            # Calculate the angle between the current absolute compass direction
            # and the angle in memory.
            ens_compass_diff_pre = nengo.Ensemble(
                n_neurons=400,
                dimensions=4,
                radius=1.5,
                label="ens_compass_diff_pre",
                seed=self.seed_gen())
            ens_compass_diff = nengo.Ensemble(
                n_neurons=50,
                dimensions=1,
                label="ens_compass_diff",
                seed=self.seed_gen())
            nengo.Connection(self.node_absolute_compass,
                             ens_compass_diff_pre[0:2])
            nengo.Connection(ens_memory_dir.ensembles[0],
                             ens_compass_diff_pre[2])
            nengo.Connection(ens_memory_dir.ensembles[1],
                             ens_compass_diff_pre[3])
            nengo.Connection(ens_compass_diff_pre, ens_compass_diff,
                function=lambda x: (x[0] * x[2] + x[1] * x[3]) / np.maximum(0.1, np.hypot(x[2], x[3])))

            # Calculate the difference between the current distance to the goal
            # and the remembered distance to the goal
            ens_dist_diff = nengo.Ensemble(
                n_neurons=50,
                dimensions=1,
                label="ens_dist_diff",
                seed=self.seed_gen())
            nengo.Connection(
                self.node_distance_to_goal, ens_dist_diff, transform=-1)
            nengo.Connection(ens_memory_dist, ens_dist_diff, transform=1)

            # Combine the angle difference and the distance difference in a
            # single ensemble
            ens_stop_follow_obstacle = nengo.Ensemble(
                n_neurons=100,
                dimensions=2,
                radius=1.5,
                label="ens_stop_follow_obstacle",
                seed=self.seed_gen())
            nengo.Connection(
                ens_compass_diff,
                ens_stop_follow_obstacle[0],
                function=lambda x: 1.5 * (np.abs(x) - 0.8),
                synapse=10e-3)
            nengo.Connection(
                ens_dist_diff,
                ens_stop_follow_obstacle[1],
                function=lambda x: 1.5 * (x - 0.05),
                synapse=10e-3)

            xs = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]]) * np.sqrt(2)
            def attractor_field(x):
                return 0.1 * (xs[np.argmax(xs @ x.T)] - x) + x
            nengo.Connection(
                ens_stop_follow_obstacle,
                ens_stop_follow_obstacle,
                function=attractor_field,
                synapse=10e-3)

            # Instantiate the state network, output determines which states are
            # inactive.
            net_state = common_neural.make_state_network(
                self.seed_gen, 4, transitions=[1, 2, 3, 1])

            # Leave the first state once the memory initialisation is done
            nengo.Connection(
                ens_memory_dir_init,
                net_state.input,
                function=lambda x: 1 - 1.5 * x)

            # Switch to the next state once we hit an obstacle
            nengo.Connection(ens_hit_obstacle, net_state.input)

            # Switch the next state once we've remembered the distance to the
            # goal (the value in the difference ensemble is close to zero)
            nengo.Connection(
                ens_memory_dist_diff,
                net_state.input,
                function=lambda x: 1 - 5 * np.abs(x),
                synapse=50e-3)

            # Switch to the next state once we've encountered a point on the
            # obstacle outline with approximately the same angle as the
            # angle-to-goal line at a point that is closer to the obstacle than
            # the obstacle hit point.
            nengo.Connection(
                ens_stop_follow_obstacle,
                net_state.input,
                function=lambda x: 2.0 if x[0] > 0.1 and x[1] > 0.1 else -2.0)

            # Shut down the memory initialisation subnetwork if we are not in
            # state zero
            nengo.Connection(
                net_state.output[STATE_INIT_DIR_MEMORY],
                ens_memory_dir_init.neurons,
                transform=-5 * np.ones((ens_memory_dir_init.n_neurons, 1)))
            for i in range(2):
                nengo.Connection(
                    net_state.output[STATE_INIT_DIR_MEMORY],
                    ens_memory_dir_diff.ensembles[i].neurons,
                    transform=-5 * np.ones(
                        (ens_memory_dir_diff.ensembles[i].n_neurons, 1)))

            # Shut down the hit obstacle ensemble if we are not in the move
            # towards goal state, as well as the compass ensemble
            nengo.Connection(
                net_state.output[STATE_GO_TOWARDS_GOAL],
                ens_hit_obstacle.neurons,
                transform=-5 * np.ones((ens_hit_obstacle.n_neurons, 1)))
            nengo.Connection(
                net_state.output[STATE_GO_TOWARDS_GOAL],
                ens_compass.neurons,
                transform=-5 * np.ones((ens_compass.n_neurons, 1)))

            # Shut down the distance memory update ensembles if we are not in the
            # remember obstacle state
            nengo.Connection(
                net_state.output[STATE_INIT_DIST_MEMORY],
                ens_memory_dist_diff.neurons,
                transform=-5 * np.ones((ens_memory_dist_diff.n_neurons, 1)))

            # Shut down the follow_obstacle ensemble, as well as the
            # ens_stop_follow_obstacle, if we are not in the follow obstacle
            # state
            nengo.Connection(
                net_state.output[STATE_FOLLOW_OBSTACLE],
                net_follow_obstacle.stop,
                transform=-1)
            nengo.Connection(
                net_state.output[STATE_FOLLOW_OBSTACLE],
                ens_stop_follow_obstacle.neurons,
                transform=-5 * np.ones((ens_stop_follow_obstacle.n_neurons,
                                        1)))

            # Add some traces
            self.add_trace(net_state.output, "state", "State",
                           lambda x: np.argmin(x) + 1 if np.any(x < 0.2) else 0,
                           ["IDLE", "DIR\\_MEM", "MOVE", "DIST\\_MEM", "FOLLOW"])
            self.add_trace(net_state.input, "net_state.input", "Clk")
            self.add_trace(ens_compass_diff, "ens_compass_diff", "$\\Delta \\vec\\alpha$")
            self.add_trace(ens_dist_diff, "ens_dist_diff", "$\\Delta d$")
            self.add_trace(ens_memory_dist_diff, "ens_memory_dist_diff", "Mem dist diff")
            self.add_trace(ens_memory_dist, "ens_memory_dist", "Dist diff")
            self.add_trace(ens_stop_follow_obstacle[0], "ens_stop_follow_obstacle 0", "Stop follow 0")
            self.add_trace(ens_stop_follow_obstacle[1], "ens_stop_follow_obstacle 1", "Stop follow 1")
            self.add_trace(ens_hit_obstacle, "ens_hit_obstacle", "Hit")

