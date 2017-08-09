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
# common_neural.py
#
# Contains commonly used subnetworks shared by all neural network based agent
# implementations.
#

import nengo
import numpy as np
import common


def make_follow_obstacle_network(seed_gen,
                                 radar_range,
                                 radar_vectors,
                                 min_dist,
                                 max_dist=1.0):
    """
    The make_follow_obstacle_network function generates a Nengo subnetwork which
    implements the follow_obstacle function from the "common" module. The
    network receives the radar depths as an input and generates motor commands
    which cause the agent to follow the corresponding obstacle outline.

    The network simply implements the function from the "common" module in a
    feed-forward fashion. The only non-linearity that has to be taken into
    account is the depth-to-weight conversion. However, this function is well
    realisable in a neural substrate.
    """

    N = len(radar_vectors)  # Number of radar vectors

    # Rotate the static direction vectors by M_ROT
    radar_vectors_rotated = (radar_vectors @ common.M_ROT)

    # Build a network implementing the follow_obstacle function
    network = nengo.Network(label="follow_obstacle")
    with network:
        # Input and output nodes
        network.radar_depth = nengo.Node(size_in=N, label="radar_depth")
        network.motor = nengo.Node(size_in=2, label="motor")
        network.stop = nengo.Node(size_in=1, label="stop")

        # Represent the depth values and weights in independent neuron ensembles
        ea_radar_depth = nengo.networks.EnsembleArray(
            n_neurons=50,
            n_ensembles=N,
            label="ens_radar_depth",
            seed=seed_gen())
        ea_weight = nengo.networks.EnsembleArray(
            n_neurons=50,
            n_ensembles=N,
            label="ens_radar_weight",
            seed=seed_gen())

        # Compute the weight for each vector
        nengo.Connection(
            network.radar_depth, ea_radar_depth.input, synapse=None)
        for i in range(N):
            nengo.Connection(
                ea_radar_depth.ensembles[i],
                ea_weight.ensembles[i],
                function=lambda x: \
                    common.follow_obstacle_vector_weight(
                        x,
                        min_dist / radar_range,
                        max_dist / radar_range))

        # From the weights compute the individual scaled vector components and
        # sum them in the motor output ensemble. The neurons will automatically
        # normalise the vector length to (a little bit more than) the given
        # radius.
        ens_motor = nengo.Ensemble(
            n_neurons=50,
            dimensions=2,
            label="ens_motor",
            radius=0.9,
            seed=seed_gen())
        nengo.Connection(
            ea_weight.output, ens_motor, transform=radar_vectors_rotated.T)
        nengo.Connection(
            network.stop,
            ens_motor.neurons,
            transform=5 * np.ones((ens_motor.n_neurons, 1)),
            synapse=10e-3)
        nengo.Connection(ens_motor, network.motor, synapse=None)

    return network


def make_track_minimum_network(seed_gen):
    """
    Creates a network which tracks the minimum of the input. Provides a reset
    input which allows to reset the tracker to the maximum value.
    """
    network = nengo.Network(label="track_minimum")
    with network:
        # Input and output nodes
        network.input = nengo.Node(size_in=1)
        network.output = nengo.Node(size_in=1)
        network.reset = nengo.Node(size_in=1)

        # Memory/integrator ensemble
        tau_int = 100e-3
        ens_minimum = nengo.Ensemble(
            n_neurons=200,
            dimensions=1,
            radius=1.1,
            label="ens_minimum",
            seed=seed_gen())
        nengo.Connection(ens_minimum, ens_minimum, synapse=tau_int)

        # Ensemble holding the difference between the current input and the
        # content of the memory. Tuning curves of this ensemble are chosen in
        # such a way that only positive values can be represented -- this
        # non-linearity allows to track the minimum
        ens_diff = nengo.Ensemble(
            n_neurons=50,
            dimensions=1,
            intercepts=nengo.dists.Uniform(0.05, 0.95),
            encoders=nengo.dists.Choice([[1]]),
            label="ens_diff")
        nengo.Connection(network.input, ens_diff, transform=-1)
        nengo.Connection(ens_minimum, ens_diff, transform=1)
        nengo.Connection(ens_diff, ens_minimum, transform=-tau_int)
        nengo.Connection(ens_minimum, network.output)

        # Allow the reset input to directly influence the integrator state
        nengo.Connection(network.reset, ens_minimum)
    return network


def make_state_network(seed_gen, N, invert_output=True, transitions=None):
    """
    Clocked state transition network. This network can be used to implement a
    state machine with fixed state transitions. Transitions are clocked by the
    rising edge of the input signal.

    seed_gen: a function returning a pseudo-random number when being called.
    N: number of states.
    invert_output: if True, the output is a state vector containing ones for all
    inactive states and zeros for all active states. Otherwise, active states
    are represented by ones in the output vector.
    transitions: state transitions. If None is given, states are aranged in a
    ring.
    """
    if transitions is None:
        transitions = np.roll(np.arange(0, N), -1)
    vs = np.linspace(0, 2 * np.pi, N + 1)[0:-1]
    xs = np.array((np.cos(vs), np.sin(vs))).T
    network = nengo.Network(label="state")
    with network:
        # I/O nodes: a one dimensional input ("the clock") and an N-dimensional
        # output
        network.input = nengo.Node(size_in=1)
        network.output = nengo.Node(size_in=N)

        #### NEGATIVE/POSITIVE INPUT ####

        # Buffer the input in an ensemble, make sure the input is clearly zero
        # or one.
        ens_in = nengo.Ensemble(
            n_neurons=50, dimensions=1, label="ens_in", seed=seed_gen())
        nengo.Connection(network.input, ens_in, synapse=None)
        nengo.Connection(ens_in, ens_in, function=lambda x: x - 0.5)

        ens_in_pos = nengo.Ensemble(
            n_neurons=10,
            dimensions=1,
            label="ens_in_pos",
            intercepts=nengo.dists.Uniform(0.05, 0.95),
            encoders=nengo.dists.Choice([[1]]),
            seed=seed_gen())
        nengo.Connection(ens_in, ens_in_pos)

        ens_in_neg = nengo.Ensemble(
            n_neurons=10,
            dimensions=1,
            label="ens_in_neg",
            intercepts=nengo.dists.Uniform(0.05, 0.95),
            encoders=nengo.dists.Choice([[1]]),
            seed=seed_gen())
        nengo.Connection(ens_in, ens_in_neg, function=lambda x: 1 - x)

        #### STATE ENSEMBLES ####

        # Two integrators, one holding the old state and one converging to
        # the next state. The attractor_field function allows the integrator
        # to act like an associative memory. The additive term ensures
        # the memory converges to the initial state when starting at zero.
        def attractor_field(x):
            return 0.1 * (xs[np.argmax(xs @ x.T)] - x) * np.linalg.norm(x) + x

        tau_int = 100e-3
        ens_state_old = nengo.Ensemble(
            n_neurons=200,
            dimensions=2,
            label="ens_state_old",
            seed=seed_gen())
        ens_state = nengo.Ensemble(
            n_neurons=200,  #
            dimensions=2,
            label="ens_state",
            seed=seed_gen())
        nengo.Connection(
            ens_state_old,
            ens_state_old,
            synapse=tau_int,
            function=attractor_field,
            transform=1.1)
        nengo.Connection(
            ens_state,
            ens_state,
            synapse=tau_int,
            function=attractor_field,
            transform=1.1)

        #### GATING ENSEMBLES ####

        # Two sets of interneurons sending data from the old to the new state
        # ensemble and vice versa
        ens_inter_1 = nengo.Ensemble(
            n_neurons=50, dimensions=2, label="ens_inter_1", seed=seed_gen())
        ens_inter_2 = nengo.Ensemble(
            n_neurons=50, dimensions=2, label="ens_inter_2", seed=seed_gen())
        nengo.Connection(
            ens_state_old,
            ens_inter_1,
            function=lambda x: xs[transitions[np.argmax(xs @ x.T)]])
        nengo.Connection(ens_inter_1, ens_state, transform=1)
        nengo.Connection(ens_state, ens_inter_1, transform=-1)

        nengo.Connection(ens_state, ens_inter_2)
        nengo.Connection(ens_inter_2, ens_state_old, transform=1)
        nengo.Connection(ens_state_old, ens_inter_2, transform=-1)

        # Let the input ensemble gate the interneurons
        nengo.Connection(
            ens_in_pos,
            ens_inter_2.neurons,
            transform=-5 * np.ones((ens_inter_1.n_neurons, 1)))
        nengo.Connection(
            ens_in_neg,
            ens_inter_1.neurons,
            transform=-5 * np.ones((ens_inter_2.n_neurons, 1)))

        #### OUTPUT ENSEMBLES ####

        # Decode the state vector from the state ensemble, use an output
        # ensemble array with mutual inhibition
        def mk_output_fun(j):
            return lambda x: 1.0 * np.argmax(xs @ x.T) == j

        tau_lp = 1.0
        ea_output_raw = nengo.networks.EnsembleArray(
            n_neurons=20,
            n_ensembles=N,
            label="ea_output_raw",
            seed=seed_gen())
        ea_output_raw_lp = nengo.networks.EnsembleArray(
            n_neurons=20,
            n_ensembles=N,
            label="ea_output_raw_lp",
            seed=seed_gen())
        ea_output_hyst = nengo.networks.EnsembleArray(
            n_neurons=100,
            n_ensembles=N,
            ens_dimensions=2,
            label="ea_output_hyst",
            seed=seed_gen())
        ea_output = nengo.networks.EnsembleArray(
            n_neurons=20,
            n_ensembles=N,
            label="ea_output",
            intercepts=nengo.dists.Uniform(0.1, 0.95),
            encoders=nengo.dists.Choice([[1]]),
            seed=seed_gen())
        for i in range(N):
            # Low pass filter the raw decoded output
            nengo.Connection(
                ea_output_raw.ensembles[i],
                ea_output_raw_lp.ensembles[i],
                transform=tau_int / tau_lp,
                function=lambda x: 1 if x > 0.25 else -1)
            nengo.Connection(
                ea_output_raw_lp.ensembles[i],
                ea_output_raw_lp.ensembles[i],
                transform=(1 - tau_int / tau_lp),
                synapse=tau_int)

            # Apply some hysteresis to the low pass filtered signal
            nengo.Connection(ea_output_raw_lp.ensembles[i],
                             ea_output_hyst.ensembles[i][0])
            nengo.Connection(
                ea_output_hyst.ensembles[i],
                ea_output_hyst.ensembles[i][1],
                function=
                lambda x: 2.0 * (((x[1] > 0) and (x[0] > 0.0)) or ((x[1] < 0) and (x[0] > 0.5))) - 1
            )
            nengo.Connection(
                ea_output_hyst.ensembles[i][1],
                ea_output.ensembles[i],
                transform=-1 if invert_output else 1)

        nengo.Connection(ea_output.output, network.output, synapse=None)

        for i in range(N):
            # Decode the state from the state ensemble
            nengo.Connection(
                ens_state,
                ea_output_raw.ensembles[i],
                function=mk_output_fun(i))

            # Slow mutual inhibition in the output_raw ensemble
            for j in range(N):
                if i != j:
                    nengo.Connection(
                        ea_output_raw.ensembles[i],
                        ea_output_raw.ensembles[j],
                        function=lambda x: -np.maximum(0, x),
                        synapse=tau_int)

            # All-or-nothing dynamics in the final output ensemble
            nengo.Connection(
                ea_output.ensembles[i],
                ea_output.ensembles[i],
                function=lambda x: x - 0 - 0.5)

        #### INITIALIZATION ####

        # Slightly push the inital ens_state towards [1, 0]
        ens_init = nengo.Ensemble(
            n_neurons=50, dimensions=1, label="ens_init", seed=seed_gen())
        nengo.Connection(
            ens_init,
            ens_init,
            function=lambda x: x + 1.5 * tau_int,
            synapse=tau_int)
        nengo.Connection(
            ens_init,
            ens_state[0],
            function=lambda x: 1 - x,
            transform=2 * tau_int)

        # If the output is inverted, we additionally need to pull all outputs
        # to "high" in the beginning. Otherwise all states are activated at the
        # same time.
        if invert_output:
            for i in range(N):
                nengo.Connection(
                    ens_init, ea_output.ensembles[i], function=lambda x: 1 - x)

    return network


model = make_state_network(lambda: 0, 4, True, [1, 2, 3, 1])
with model:
    stim = nengo.Node(lambda t: 1.0 * (np.sin(1.5 * np.pi * t - np.pi) > 0.5))
    nengo.Connection(stim, model.input)

