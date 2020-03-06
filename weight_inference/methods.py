import numpy as np


def sign_alignment(guess_matrix, true_matrix):
    """Measurement of the alignment of signs (+/-) between two matrices

    Args:
        guess_matrix (MxN float array): estimated matrix
        true_matrix (MxN float array): true matrix
    
    Returns:
        alignment (float): ratio similarity of signs between guess and true matrices
    """
    pos_alignment = np.sum((guess_matrix >= 0) & (true_matrix >= 0))
    neg_alignment = np.sum((guess_matrix < 0) & (true_matrix < 0))
    alignment = (pos_alignment + neg_alignment) / true_matrix.size
    return alignment


def akrout(guess_matrix, input_neuron_spiketimes, input_baseline, output_neuron_spiketimes, output_baseline,
           stimulus_length, decay_factor, time_offset=0.0):
    """Akrout method for estimating feedback weights from firing rates

    Args:
        guess_matrix (MxN float array): a prior estimate of the weight matrix
        input_neuron_spiketimes (list, np arrays): spike times over the simulation time per pre-neuron (ms)
        input_baseline (np array): baseline firing rate to be removed per neuron (calculated elsewhere) - Hz
        output_neuron_spiketimes (list, np arrays): spike times over the simulation time per post-neuron (ms)
        output_baseline (np array): baseline firing rate to be removed per neuron (calculated elsewhere) - Hz
        stimulus_length (float): duration of an individual stimulus
        batch_size (int): number of stimuli in the data to be consumed in a single pass
        decay_factor (float): the relative degree of multiplicative weight decay
        time_offset (float, optional): any offset to the spike-times necessary for proper data interpretation
        unit_conversion (float, optional): factor by which to convert measure of firing to Hz (i.e. spikes/second)

    Returns:
        update_matrix (MxN float array): the measured updated to the guess matrix given the data
    """
    num_input_neurons = len(input_neuron_spiketimes)
    num_output_neurons = len(output_neuron_spiketimes)
    input_neuron_firingrates = np.zeros((num_input_neurons))
    output_neuron_firingrates = np.zeros((num_output_neurons))

    for i in range(num_input_neurons):
        mask = input_neuron_spiketimes[i] >= time_offset
        mask = mask & (input_neuron_spiketimes[i] < (time_offset + stimulus_length))
        input_neuron_firingrates[i] = np.sum(mask)
    for o in range(num_output_neurons):
        mask = output_neuron_spiketimes[o] >= time_offset
        mask = mask & (output_neuron_spiketimes[o] < (time_offset + stimulus_length))
        output_neuron_firingrates[o] = np.sum(mask)

    meaned_input_firingrates = input_neuron_firingrates - input_baseline
    meaned_output_firingrates = output_neuron_firingrates - output_baseline

    update_matrix = np.matmul(meaned_output_firingrates[:, np.newaxis], meaned_input_firingrates[np.newaxis, :])
    LTD = decay_factor * guess_matrix
    update_matrix -= LTD
    return update_matrix


def stdwi(guess_matrix, input_binary_spikes, output_binary_spikes, slow_in_trace, fast_in_trace, learning_rate,
         decay_weighting):
    """Spike Timing-Dependent based inference of weights

    Args:
        guess_matrix (IxJ float array): A prior estimate of the weight matrix
        input_binary_spikes (JxN binary array): A binary matrix giving spike location of M input neuron in N timesteps
        output_binary_spikes (IxN binary array): A binary matrix giving spike location of M output neuron in N timesteps
        slow_in_trace (JxN float array): A slow exponential moving trace of the input neuron spikes for the N timesteps 
        fast_in_trace (JxN float array): A fast exponential moving trace of the input neuron spikes for the N timesteps
        learning_rate (float): A weighting of the synaptic inference update
        decay_weighting (float): The relative strength of depression

    Returns:
        update_matrix (IxJ float array): the measured updated to the guess matrix given the data
    """
    update_matrix = np.copy(guess_matrix)
    nb_sub_timesteps = input_binary_spikes.shape[1]
    relative_input_trace = fast_in_trace - slow_in_trace

    for t_indx in range(nb_sub_timesteps):
        # Potentiation is based upon output spike times also but also by mult. with the relevant input traces
        LTP_update = np.matmul(
            output_binary_spikes[:, t_indx][:, np.newaxis],
            relative_input_trace[:, t_indx][np.newaxis, :])

        # Calculating depression based upon output spikes
        LTD_update = np.copy(update_matrix)
        LTD_mask = np.repeat(
            output_binary_spikes[:, t_indx][:, np.newaxis], update_matrix.shape[1], axis=1)
        LTD_update *= LTD_mask

        # Complete update based upon LTP and LTD incl. learning rate
        update_matrix += learning_rate * (LTP_update - decay_weighting * LTD_update)

    return update_matrix


def create_stdwi_trace(prev_trace, binary_spike_matrix, alpha, tau, timestep, alltoall):
    """Produces an exponential moving average estimation of firing rate from spike times
    
    Args:
        prev_trace (MxN float matrix): The trace which is being continued (normally initialised with zeros)
        binary_spike_matrix (MxN int matrix): A binary matrix indicating spike times of the M neurons in N timesteps
        alpha (float): The height of exponential moving average filter
        tau (float): Time constant of the exponential moving average filter
        timestep (float): Timestep of spiking data to compute traces with tau
        alltoall (bool): All to all vs. nearest only trace
    Returns:
        next_trace (MxN float matrix): The trace given the prev_trace and provided spikes/config
    """
    next_trace = np.zeros(prev_trace.shape)
    decay_factor = np.exp(-timestep / tau)
    nb_sub_timesteps = prev_trace.shape[1]

    next_trace[:, 0] = decay_factor * prev_trace[:, -1]
    if alltoall:
        next_trace += alpha * binary_spike_matrix

    for t_indx in np.arange(1, nb_sub_timesteps):
        next_trace[:, t_indx] += decay_factor * next_trace[:, t_indx - 1]
        if not alltoall:
            next_trace[binary_spike_matrix[:, t_indx] > 0.0, t_indx] = alpha

    return next_trace


def rdd(initial_guess_matrix, presynaptic_membrane_voltages, presynaptic_accum_voltages, postsynaptic_XPSPs, alpha,
        window_size, threshold, timestep, learning_rate, check_interval, maximum_u):
    """Regression discontinuity design method for feedback weight estimation

    Args:
        initial_guess_matrix (MxN float array): a prior estimate of the weight matrix
        presynaptic_membrane_voltages (NxT float array): presynaptic membrane voltages for all timesteps (T)
        presynaptic_accum_voltages (NxT float array): presynaptic accummulated voltages for all timesteps (T)
        postsynaptic_XPSPs (MxT float array): the post-synaptic output convolved by our synaptic kernel
        alpha (float): The margin around the threshold which is used to find ``events''
        window_size (int): The number of timesteps after we hit the margin that we use for analysis
        threshold (float): The spiking threshold
        learning_rate (float): Factor by which estimate is updated with every sample
        check_interval (int): The number of timesteps which form the interval on which we store copies of our update
        timestep (float): The step of time per datapoint shift
        maximum_u (float): The upper limit of u -- if above this value, don't do update
    Returns:
        update_matrix (MxN float array): the measured updated to the guess matrix given the data
    """

    nb_pre_neurons = presynaptic_membrane_voltages.shape[0]
    nb_timesteps = presynaptic_membrane_voltages.shape[1]

    rdd_c1 = -np.copy(initial_guess_matrix)
    rdd_c2 = -np.copy(initial_guess_matrix)
    rdd_c3 = np.copy(initial_guess_matrix)
    rdd_c4 = np.copy(initial_guess_matrix)

    mask = initial_guess_matrix < 0.0
    rdd_c1[~mask] = 0.0
    rdd_c2[~mask] = 0.0
    rdd_c3[mask] = 0.0
    rdd_c4[mask] = 0.0

    # Running across all time
    # Performing linear regression by gradient descent
    rdd_guesses = []
    prev_window = np.zeros((nb_pre_neurons)).astype(int) - 1
    for t_indx in range(nb_timesteps):
        # If we are at a check-interval, dump a copy of the current estimate
        if (t_indx % int((1e2 / timestep) * check_interval)) == 0:
            rdd_guesses.append((rdd_c3 + rdd_c4) - (rdd_c1 + rdd_c2))

        # Get the binary list of pre-synaptic neurons whose voltages exceeded the boundary at this timestep
        mask = (presynaptic_membrane_voltages[np.arange(nb_pre_neurons), t_indx] >= (threshold - alpha)) & (
            t_indx > prev_window)

        # Iterate through the ids for the the neurons above boundary
        for h in np.arange(nb_pre_neurons)[mask]:
            # Get the maximum voltage that these neurons get to within a window
            u_max = np.max(presynaptic_accum_voltages[h, t_indx:(t_indx + window_size)])

            # Only carry out an update if we did not exceed a high value with u_max
            if np.abs(u_max - threshold) <= maximum_u :
                # Get the amount above baseline that the post-synaptic output was on average in this window
                delta_XPSPs = np.mean(postsynaptic_XPSPs[:, t_indx:(t_indx + window_size)], axis=1) -\
                              postsynaptic_XPSPs[:, t_indx]
                # If our input drive never reached threshold, then update the below-threshold linear fit
                if u_max < threshold:
                    rdd_c1[:, h] -= learning_rate * u_max * (u_max * rdd_c1[:, h] + rdd_c2[:, h] - delta_XPSPs)
                    rdd_c2[:, h] -= learning_rate * (u_max * rdd_c1[:, h] + rdd_c2[:, h] - delta_XPSPs)
                # If the input drive exceeded the threshold, use its value to update the above-threshold fit
                else:
                    rdd_c3[:, h] -= learning_rate * u_max * (u_max * rdd_c3[:, h] + rdd_c4[:, h] - delta_XPSPs)
                    rdd_c4[:, h] -= learning_rate * (u_max * rdd_c3[:, h] + rdd_c4[:, h] - delta_XPSPs)

                prev_window[h] = t_indx + window_size

    return rdd_guesses


def bayesian_hitting(guess_matrix, guess_variance, last_input_spike_times, last_output_spike_times,
                     current_output_spikes, delta, time, drift, diffusion_const, variance_bound, incremental,
                     learning_rate):
    """A hitting-time derived method for bayesian estimation of the weight matrix

    Args:
        guess_matrix (MxN float array): a prior estimate of the mean weight
        guess_variance (MxN float array): a prior estimate of the weight variance
        last_input_spike_times (Nx1 float array): a list of the last times that input neurons spiked
        last_output_spike_times (Mx1 float array): a list of the last times that output neurons spiked
        current_output_spikes (Mx1 int array): a binary vector indicating output neuron spikes in this timestep
        delta (Mx1 float array): The distance to threshold of each post-synaptic neuron
        time (float): the current time (using which time since last spikes are calculated)
        drift (float): Wiener process drift approximation
        diffusion_const (float): instantaneous variance of the Wiener process
        variance_bound (float): The minimum value which the variance can take (in order to ensure continued learning)
        incremental (bool): False means use the full bayesian update. True means don't use variance
        learning_rate (float): learning rate for incremental version
    Returns:
        weight_matrix (MxN float array): the updated guess matrix
        var_matrix (MxN float array): the updated variance matrix 
    """
    weight_matrix = np.copy(guess_matrix)
    var_matrix = np.copy(guess_variance)
    # Eligible Updates
    output_spikes = np.where(current_output_spikes > 0.0)[0]

    # Time since previous spikes
    time_since_input_spike = time - last_input_spike_times
    assert (np.sum(time_since_input_spike < 0.0) == 0.0)
    time_since_output_spike = time - last_output_spike_times
    assert (np.sum(time_since_output_spike < 0.0) == 0.0)

    for n in output_spikes:
        # This is an estimate of the synaptic weights to output neuron (n) based upon time since spike and delta
        weight = delta[n] - (drift * time_since_input_spike +
                             np.sqrt((drift * time_since_input_spike) ** 2 + 4 * diffusion_const * time_since_input_spike)) / 2
        if incremental:
            mask = time_since_input_spike < time_since_output_spike[n]
            weight_matrix[n, :][mask] = learning_rate * (weight[mask] - weight_matrix[n, :][mask])
            continue
        # This is the variance of our above estimate
        secder = - (delta[n] - weight) ** -2 - (diffusion_const * time_since_input_spike) ** -1
        var = -1 / secder
        var[var < variance_bound] = variance_bound

        # Using our estimates and prior to get a posterior
        posterior_var = (1 / guess_variance[n, :] + 1 / var) ** -1
        # Bounding the variance
        # posterior_var[posterior_var < variance_bound] = variance_bound
        posterior_weight = posterior_var * (guess_matrix[n, :] / guess_variance[n, :] + weight / var)

        mask = time_since_input_spike < time_since_output_spike[n]
        var_matrix[n, :][mask] = posterior_var[mask]
        weight_matrix[n, :][mask] = posterior_weight[mask]

    return weight_matrix, var_matrix
