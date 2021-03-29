import numpy as np
from weight_inference import simulator, methods


def akrout(initial_guess_matrix, input_neuron_spiketimes, output_neuron_spiketimes, simulation_time, stimulus_length, batch_size, learning_rate, check_interval, decay_factor=0.5, input_baseline=None, output_baseline=None):
    """A function processing spiking data to rate information per stimulus and batch-wise doing the Akrout algorithm
    """
    nb_stimuli = int(simulation_time / stimulus_length)
    nb_batches = int(nb_stimuli / batch_size)
    # Calculating firing rate baselines
    if input_baseline is None:
        input_baseline = np.zeros((len(input_neuron_spiketimes), nb_batches))
        for i_indx in range(len(input_neuron_spiketimes)):
            for b_indx in range(nb_batches):
                mask = input_neuron_spiketimes[i_indx] >= (stimulus_length*batch_size*b_indx)
                mask = mask & (input_neuron_spiketimes[i_indx] < (stimulus_length*batch_size*(b_indx+1)))
                input_baseline[i_indx, b_indx] = np.sum(mask) / batch_size
    if output_baseline is None:
        output_baseline = np.zeros((len(output_neuron_spiketimes), nb_batches))
        for o_indx in range(len(output_neuron_spiketimes)):
            for b_indx in range(nb_batches):
                mask = output_neuron_spiketimes[o_indx] >= (stimulus_length*batch_size*b_indx)
                mask = mask & (output_neuron_spiketimes[o_indx] < (stimulus_length*batch_size*(b_indx+1)))
                output_baseline[o_indx, b_indx] = np.sum(mask) / batch_size

    akrout_guess = np.copy(initial_guess_matrix)
    akrout_weight_dumps = []    
    for s_indx in range(nb_stimuli):
        if (s_indx % check_interval) == 0:
            akrout_weight_dumps.append(np.copy(akrout_guess))
        b_indx = int(s_indx / batch_size)
        akrout_guess += learning_rate * methods.akrout(
            akrout_guess,
            input_neuron_spiketimes,
            input_baseline[:,b_indx],
            output_neuron_spiketimes,
            output_baseline[:,b_indx],
            stimulus_length,
            decay_factor,
            time_offset=(s_indx*stimulus_length))
    return akrout_weight_dumps, input_baseline, output_baseline


def stdwi(initial_guess_matrix, input_neuron_spiketimes, output_neuron_spiketimes, simulation_time, stimulus_length, timestep, a_slow, t_slow, a_fast, t_fast, learning_rate, check_interval, decay_factor=0.1, alltoall=True,  offsetanalysis=0, fast_input_trace=None):
    """A function which preprocesses spiking data and carries out stdwi "inference" to estimate weights
    """
    nb_input_neurons = len(input_neuron_spiketimes)
    nb_timesteps_per_stimulus = int(stimulus_length / timestep)
    nb_stimuli = int(simulation_time / stimulus_length)
    
    stdwi_guess = np.copy(initial_guess_matrix)
    stdwi_weight_dumps = []
    
    # For the STDWI, it is algorithmically more efficient to have a binary spike matrix
    input_binary_spike_matrix = simulator.binary_spike_matrix(input_neuron_spiketimes, simulation_time, timestep)
    output_binary_spike_matrix = simulator.binary_spike_matrix(output_neuron_spiketimes, simulation_time, timestep)

    slow_input_trace = np.zeros((nb_input_neurons, nb_timesteps_per_stimulus*nb_stimuli))
    slow_input_trace = methods.create_stdwi_trace(slow_input_trace, input_binary_spike_matrix, a_slow, t_slow, timestep, alltoall)
    
    if fast_input_trace is None:
        fast_input_trace = np.zeros((nb_input_neurons, nb_timesteps_per_stimulus*nb_stimuli))
        fast_input_trace = methods.create_stdwi_trace(fast_input_trace, input_binary_spike_matrix, a_fast, t_fast, timestep, alltoall)

    if not alltoall:
        slow_input_trace *= (np.sum(fast_input_trace) / np.sum(slow_input_trace))

    for s_indx in range(nb_stimuli):
        if s_indx % check_interval == 0:
            stdwi_weight_dumps.append(np.copy(stdwi_guess))
        
        input_binary_submatrix = input_binary_spike_matrix[:, (s_indx*nb_timesteps_per_stimulus):((s_indx+1)*nb_timesteps_per_stimulus)] 
        output_binary_submatrix = output_binary_spike_matrix[:, (s_indx*nb_timesteps_per_stimulus):((s_indx+1)*nb_timesteps_per_stimulus)]
        slow_input_subtrace = slow_input_trace[:, (s_indx*nb_timesteps_per_stimulus):((s_indx+1)*nb_timesteps_per_stimulus)] 
        fast_input_subtrace = fast_input_trace[:, (s_indx*nb_timesteps_per_stimulus):((s_indx+1)*nb_timesteps_per_stimulus)] 

        if s_indx > offsetanalysis:
            stdwi_guess = methods.stdwi(
                stdwi_guess,
                input_binary_submatrix,
                output_binary_submatrix,
                slow_input_subtrace, fast_input_subtrace,
                learning_rate, decay_factor)

    return stdwi_weight_dumps, fast_input_trace


def rdd(initial_guess_matrix, presynaptic_membrane_voltages, presynaptic_accum_voltages, postsynaptic_XPSPs, alpha,
        window_size, threshold, timestep, learning_rate, check_interval, maximum_u=10.0):
    # Method for RDD is feature complete
    return methods.rdd(initial_guess_matrix, presynaptic_membrane_voltages, presynaptic_accum_voltages, postsynaptic_XPSPs, alpha,
        window_size, threshold, timestep, learning_rate, check_interval, maximum_u)


def bayes(
    initial_guess_matrix,
    input_neuron_spiketimes,
    output_neuron_spiketimes,
    output_neuron_mem,
    simulation_time,
    stimulus_length,
    timestep,
    dump_interval,
    drift, diffusion_term, variance_bound=0.0, threshold=1.0, offsetanalysis=0, incremental=False, learning_rate=0.0):
    nb_input_neurons = len(input_neuron_spiketimes)
    nb_output_neurons = len(output_neuron_spiketimes)
    nb_timesteps_per_stimulus = int(stimulus_length / timestep)
    nb_stimuli = int(simulation_time / stimulus_length)
    
    bayes_guess = np.copy(initial_guess_matrix)
    bayes_var = np.ones(initial_guess_matrix.shape)
    bayes_weight_dumps = []
    bayes_var_dumps = []

    # Have binary spike matrices make the algorithm writing more transparent
    input_binary_spike_matrix = simulator.binary_spike_matrix(input_neuron_spiketimes, simulation_time, timestep)
    output_binary_spike_matrix = simulator.binary_spike_matrix(output_neuron_spiketimes, simulation_time, timestep)

    delta_tracker = np.ones((nb_output_neurons, nb_input_neurons))
    last_input_spike_times = np.zeros((nb_input_neurons))
    last_output_spike_times = np.zeros((nb_output_neurons))

    for s_indx in range(nb_stimuli):
        if s_indx % dump_interval == 0:
            bayes_weight_dumps.append(np.copy(bayes_guess))
            bayes_var_dumps.append(bayes_var)
        
        for t_indx in range(nb_timesteps_per_stimulus*s_indx, nb_timesteps_per_stimulus*(s_indx+1)):
            if t_indx > (offsetanalysis * nb_timesteps_per_stimulus):
                bayes_guess, bayes_var = methods.bayesian_hitting(
                    bayes_guess,
                    bayes_var,
                    last_input_spike_times,
                    last_output_spike_times,
                    output_binary_spike_matrix[:,t_indx],
                    delta_tracker,
                    t_indx*timestep,
                    drift,
                    diffusion_term,
                    variance_bound,
                    incremental,
                    learning_rate)

            curr_delta = threshold - output_neuron_mem[:, t_indx]
            assert(np.sum(curr_delta < 0.0) == 0)
            mask = input_binary_spike_matrix[:, t_indx] > 0.0
            for i in np.where(mask)[0]:
                delta_tracker[:, i] = curr_delta
            last_input_spike_times[mask] = t_indx*timestep

            mask = output_binary_spike_matrix[:, t_indx] > 0.0
            last_output_spike_times[mask] = t_indx*timestep
            assert(np.sum(delta_tracker < 0.0) == 0)
    return bayes_weight_dumps, bayes_var_dumps
