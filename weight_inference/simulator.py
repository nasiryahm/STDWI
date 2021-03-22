import numpy as np

def wiener_process(nb_trials, nb_timesteps, timestep, drift, diffusion, seed=42):
    """Produces multiple discrete wiener processes all with specified drift/diffusion

    Args:
        nb_trials (int): the number of ``neurons'' or trials to simulate
        nb_timesteps (int): the number of steps to simulate
        timestep (float): the width of each time step (affects drift/diffusion)
        drift (float): the wiener drift
        diffusion (float): the diffusion constant which scales the wiener process noise
        seed (int, optional): a seed for the random number generator

    Returns:
         output (array, nb_trials x nb_timesteps)
    """
    r = np.random.RandomState(seed)
    stepwise = r.normal(0, np.sqrt(diffusion * timestep), size=(nb_trials, nb_timesteps)) + timestep * drift
    return np.cumsum(stepwise, axis=1)


def correlated_poisson_spike_train(num_neurons, firing_rate, correlation, simulation_time, timestep, seed=42):
    """Produces a set of Poisson process sampled spikes with a thinning process to achieve correlation
    This function creates the desired firing rate as a threshold and draws random numbers (tested against this threshold) to determine spikes.
    This is augmented with a shared random process to determine when neurons should share their activity with some global correlated spike-set.
    Based on Single Interaction Process Model described:
    Kuhn et al (2003). Higher-order statistics of input ensembles and the response of simple model neurons. NeCo
    Args:
        num_neurons (int): the number of neurons to simulate
        firing_rate (float): the firing rate to simulate for these neurons (spikes/ms)
        correlation (int): the within-group correlation
        simulation_time (float): the number of ms of simulation time (ms)
        timestep (float): the simulation timestepping (ms)
    Returns:
        spike_trains (list, np arrays): spike times over the simulation time per neuron (ms)
    """
    nb_timesteps = int(simulation_time / timestep)
    firing_rate_adjusted = firing_rate * timestep # Converting to spikes/timestep

    # Creating a global spike train which all other spikes will correlate 
    r = np.random.RandomState(seed)
    global_correlating_spiketrain = r.rand(nb_timesteps) < firing_rate_adjusted

    spike_trains = list()
    for n_indx in range(num_neurons):
        r = np.random.RandomState(seed + 2 + n_indx)

        neuron_spiketrain = r.rand(nb_timesteps) < (1 - correlation)*firing_rate_adjusted
        correlate_steps = r.rand(nb_timesteps) < correlation
        neuron_spiketrain[correlate_steps & global_correlating_spiketrain] = 1

        spike_trains.append(timestep*np.where(neuron_spiketrain)[0])
    return spike_trains


def clip_spike_trains(spike_trains, max_time, min_time=0.0):
    """Clips spike trains to a particular time

    Args:
        spike_trains (list, np arrays): spike times by neuron in a list
        max_time (float): time after which the spike trains should be clipped (ms)
        min_time (float, optional): time from which spike trains will be clipped (ms)
    
    Returns:
        clipped_spike_trains (list, np arrays): spike trains within the min/max time only
    """
    clipped_spike_trains = list()
    for train in spike_trains:
        mask = train > min_time
        mask = mask & (train < max_time)
        clipped_spike_trains.append(train[mask])

    return clipped_spike_trains


def random_sample_spike_train(spike_trains, simulation_time, timestep, resample_period, ratio_active):
    """Randomly samples units from a spike train to be active/inactive. This shifts all spike trains when inactive.

    Note, this function expects spike trains to have positive only values.

    Args:
        spike_trains (list, np arrays): spike times by neuron in a list
        simulation_time (int): maximum time of spikes (ms)
        timestep (float): time per step (ms)
        resample_period (int): time after which active/inactive neurons should be resampled
        ratio_active (float): the ratio of neurons to keep active during each period

    Returns:
        clipped_spike_trains (list, np arrays): spike trains within the min/max time only
    """
    nb_resamples = int(simulation_time // resample_period)
    nb_neurons = len(spike_trains)
    masks = [np.zeros((len(spikes))) for spikes in spike_trains]

    for s_indx in range(nb_resamples):
        r = np.random.RandomState(s_indx + 1)
        if ratio_active < 0.5:
            chosen_units = r.choice(
                nb_neurons,
                int(ratio_active * nb_neurons),
                replace=False)
        else:
            chosen_units = r.choice(
                nb_neurons,
                int((1.0 - ratio_active) * nb_neurons),
                replace=False)

        for n_indx in chosen_units:
            masks[n_indx] += (spike_trains[n_indx] > (resample_period * s_indx)) & (spike_trains[n_indx] <= (resample_period * (s_indx + 1)))

    for n_indx in range(nb_neurons):
        if ratio_active < 0.5:
            spike_trains[n_indx] = spike_trains[n_indx][masks[n_indx] > 0.0]
        else:
            spike_trains[n_indx] = spike_trains[n_indx][masks[n_indx] == 0.0]

    return spike_trains


def xpsp_filterer(train, nb_timesteps, timestep, tau_slow, tau_fast):
    """Convolves a spike train with a double exponential causal XPSP filter

    Args:
        train (np array): list of spike times forming a single train
        nb_timesteps (int): the total number of timesteps of simulation
        timestep (float): timestep in ms
        tau_slow (float): slow decay time constant
        tau_fast (float): fast decay time constant
    
    Returns:
        xpsps (1D numpy array): the post synaptic potential for each timesteps
    """
    xpsp = np.zeros((nb_timesteps))
    spiketrain_indices = np.round(train / timestep).astype(int)
    xpsp[spiketrain_indices] += 1.0 / (tau_slow - tau_fast)


    window_size = int(7*(tau_slow / timestep))
    window_vals = timestep*np.arange(window_size)

    fast_filter = np.exp(-window_vals / tau_fast)
    fast_xpsp = np.convolve(xpsp, fast_filter)[:-(window_size - 1)]

    slow_filter = np.exp(-window_vals / tau_slow)
    slow_xpsp = np.convolve(xpsp, slow_filter)[:-(window_size - 1)]

    return slow_xpsp - fast_xpsp


def spike_trains_to_xpsps(spike_trains, sim_time, timestep, tau_slow=10.0, tau_fast=3.0):
    """Converts a list of spike trains into a 2D numpy array of post-synaptic potentials

    Assumes that all spikes cause an equivalent shaped fast E/I PSP -- all psps are positive
    
    Args:
        spike_trains (list, np arrays): spike times by neuron in a list
        sim_time (float): total simulation time in ms
        timestep (float): timestep in ms
        tau_slow (float, optional): slow decay time constant
        tau_fast (float, optional): fast decay time constant
    
    Returns:
        xpsps (2D numpy array): the post synaptic potentials for all timesteps per neuron
    """
    nb_timesteps = int(sim_time / timestep)
    xpsps = np.zeros((len(spike_trains), nb_timesteps))
    for n_indx, train in enumerate(spike_trains):
        xpsps[n_indx] = xpsp_filterer(train,
                                      nb_timesteps,
                                      timestep,
                                      tau_slow, tau_fast)
    return xpsps


def lif_dynamics(xpsps, weight_matrix, timestep, tau=20.0, thresh=1.0, rest=0.0, reset=-1.0, drift=0.0, coupling_ratio=1.0):
    """Computing leaky integrator spiking neuron dynamics given an incident XPSP and weight matrix

    Args:
        xpsps (2D numpy array, float): post synaptic potentials by pre-synaptic neuron population
        weight_matrix (2D np array, float): a weight matrix of size (postsynaptic population, presynaptic population)
        timestep (float): the timestep of simulated dynamics (Forward-Euler method)
        tau (float, optional): the time constant of the leaky integrator
        thresh (float, optional): the threshold which determines when a neuron spikes
        rest (float, optional): the resting membrane voltage, baseline
        reset (float, optional): the voltage to which neuron membranes are reset after a spike
        drift (float, optional): the constant background input to a cells
        coupling_ratio (float, optional): the dendritic vs somatic coupling ratio (g_D / g_L)

    Returns:
        acc_voltage (2D np array, float): the membrane voltage of output population without resetting at spike times
        mem_voltage (2D np array, float): the membrane voltages of output population for the duration
        spike_times (list of numpy arrays, float): spike times of the output population
    """
    nb_post_neurons = weight_matrix.shape[0]
    nb_timesteps = xpsps.shape[1]
    mem_voltage = np.zeros((nb_post_neurons, nb_timesteps))
    acc_voltage = np.zeros(mem_voltage.shape)
    spike_times = [[] for p in range(nb_post_neurons)]
    timestep_inputs = np.einsum('ij, jn->in', weight_matrix, xpsps)

    for t_indx in np.arange(1, nb_timesteps):
        total_input = timestep_inputs[:, t_indx - 1]
        # Membrane voltage update
        dmem = ((rest + drift) - mem_voltage[:, t_indx - 1]) + coupling_ratio*(total_input - mem_voltage[:, t_indx - 1])
        mem_voltage[:, t_indx] = mem_voltage[:, t_indx - 1] + (timestep / tau) * dmem
        dacc = ((rest + drift) - acc_voltage[:, t_indx - 1]) + coupling_ratio*(total_input - acc_voltage[:, t_indx - 1])
        acc_voltage[:, t_indx] = acc_voltage[:, t_indx - 1] + (timestep / tau) * dacc
        # Spike reset
        mask = mem_voltage[:, t_indx] >= thresh
        mem_voltage[:, t_indx][mask] = reset
        # Storing spikes
        spiked_neurons = np.where(mask)[0]
        for s in spiked_neurons:
            spike_times[s].append(t_indx * timestep)

    for n in range(nb_post_neurons):
        spike_times[n] = np.asarray(spike_times[n])
    return acc_voltage, mem_voltage, spike_times


def pure_if_dynamics(input_spike_times, weight_matrix, timestep, thresh=1.0, reset=0.0):
    """Computing integrator (NO LEAK) spiking neuron dynamics given an incident spike train (causing jumps) and weight matrix

    Args:
        input_spike_times (2D numpy array, int): pre-synaptic population binary spiketime matrix
        weight_matrix (2D np array, float): a weight matrix of size (postsynaptic population, presynaptic population)
        timestep (float): the timestep of simulated dynamics (Forward-Euler method)
        thresh (float, optional): the threshold which determines when a neuron spikes
        reset (float, optional): the voltage to which neuron membranes are reset after a spike
    
    Returns:
        mem_voltage (2D np array, float): the membrane voltages of output population for the duration
        spike_times (list of numpy arrays, float): spike times of the output population
    """
    nb_post_neurons = weight_matrix.shape[0]
    nb_timesteps = input_spike_times.shape[1]
    mem_voltage = np.zeros((nb_post_neurons, nb_timesteps))
    spike_times = [[] for p in range(nb_post_neurons)]
    for t_indx in np.arange(1, nb_timesteps):
        total_input = np.matmul(weight_matrix, input_spike_times[:, t_indx - 1])
        # Membrane voltage update
        mem_voltage[:, t_indx] = mem_voltage[:, t_indx - 1] + total_input
        # Spike reset
        mask = mem_voltage[:, t_indx] >= thresh
        mem_voltage[:, t_indx][mask] = reset
        # Storing spikes
        spiked_neurons = np.where(mask)[0]
        for s in spiked_neurons:
            spike_times[s].append(t_indx * timestep)

    for n in range(nb_post_neurons):
        spike_times[n] = np.asarray(spike_times[n])
    return mem_voltage, spike_times


def firing_rates(spike_trains, sim_time, unit_conversion=0.001):
    """Provides a measure of the firing rate of neurons given a spike train

    Args:
        spike_trains (list, np arrays): spike times by neuron in a list
        sim_time (float): simulation time assumed to be in ms
        unit_conversion (float, optional): used to convert from sim_time to seconds
    Returns:
        firing_rates (1D array): firing rate per neuron across all simulation time
    """
    rates = np.zeros(len(spike_trains))
    for indx, spikes in enumerate(spike_trains):
        rates[indx] = len(spikes[~np.isnan(spikes)]) / (sim_time * unit_conversion)
    return rates


def binary_spike_matrix(spike_trains, sim_time, timestep):
    """Converts a list of spike trains into a large NxT binary spike matrix

    Args:
        spike_trains (list, np arrays): spike times by neuron in a list
        sim_time (float): simulation time total
        timestep (float): timestep with which to divide the simulation time
    
    Returns:
        binary_spike_matrix (2D array)
    """
    nb_timesteps = int(sim_time / timestep)
    nb_neurons = len(spike_trains)

    binary_matrix = np.zeros((nb_neurons, nb_timesteps))
    for n_indx in range(nb_neurons):
        spiketime_indices = (spike_trains[n_indx] / timestep).astype(int)
        binary_matrix[n_indx, spiketime_indices] = 1.0
    return binary_matrix
