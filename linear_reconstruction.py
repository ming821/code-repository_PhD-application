import numpy as np


def _make_response_matrix(spikes, trigger, filter_duration):
    """Make response matrix

    Args:
        spikes (List[np.ndarray]): List of list with the spike times of the individual cells in a population (in ms).
        trigger (np.ndarray): Time point of each stimulus frame (in ms).
        filter_duration (int): Duration of the filter (in ms).

    Returns:
        np.ndarray: delay-embedded response matrix, approx. [len(trigger), nb_cells * filter_duration + 1] (+1 because of constant bias term)
    """
    nb_frames = trigger.shape[0]
    N = filter_duration
    M = nb_frames - N

    index_mat = np.tile(np.arange(N), (M,1))  # delays
    tt = np.tile(np.arange(M), (N, 1)).T  # times
    index_mat=index_mat + tt

    nb_frames = trigger.shape[0]
    N = filter_duration
    M = nb_frames - N
    nb_cells = len(spikes)

    R = np.ones((M, nb_cells * N + 2))  # response matrix (~PSTH)
    for cell in range(nb_cells):
        t, _ = np.histogram(spikes[cell], bins=trigger)
        R[:, cell*N+2:(cell+1)*N+2] = t[index_mat]
    return R


def train(spikes, trigger, stimulus, filter_duration):
    """Make reconstruction filter

    Args:
        spikes (List[np.ndarray]): List of np arrays with the spike times of the individual cells in a population (in ms).
        trigger (np.ndarray): Time point of each stimulus frame (in ms).
        stimulus (np.ndarray): Time point of each stimulus frame (in ms).
        filter_duration (int): Duration of the filter (in ms).

    Returns:
        np.ndarray: reconstruction filter [nb_cells * filter_duration + 1] (+1 because of constant bias term)
    """
    R = _make_response_matrix(spikes, trigger, filter_duration)
    s = stimulus[:R.shape[0]]
    # calculate filter
    A = (R.T@R)
    B = np.dot(R.T, s)
    reconstruction_filter, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return reconstruction_filter


def test(reconstruction_filter, spikes, trigger, filter_duration):
    """Reconstruct stimulus

    Args:
        reconstruction_filter (np.ndarray): [nb_cells * filter_duration + 1]
        spikes (List[np.ndarray]): List of np arrays with the spike times of the individual cells in a population (in ms).
        trigger (np.ndarray): Time point of each stimulus frame (in ms).
        stimulus (np.ndarray): Time point of each stimulus frame (in ms).
        filter_duration (int): Duration of the filter (in ms).

    Returns:
        np.ndarray: reconstructed stimulus approx [len(stimulus)]
    """
    R = _make_response_matrix(spikes, trigger, filter_duration)
    reconstructed_stimulus = np.dot(R, reconstruction_filter)
    return reconstructed_stimulus
