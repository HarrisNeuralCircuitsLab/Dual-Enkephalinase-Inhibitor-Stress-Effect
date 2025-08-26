import numpy as np
import pynapple as pna
from sklearn.linear_model import LinearRegression
import scipy.stats

def correct_photobleaching(raw_tsd, poly_degree=2):
    """
    Corrects photobleaching by fitting and subtracting a polynomial from a raw signal.

    Args:
        raw_tsd (pna.Tsd): The raw time series data (GCaMP or Isosbestic).
        poly_degree (int): The degree of the polynomial to fit for detrending.

    Returns:
        tuple: A tuple containing:
            - pna.Tsd: The detrended (photobleach-corrected) signal.
            - pna.Tsd: The fitted polynomial baseline (F0 estimate).
    """
    poly_coeffs = np.polyfit(raw_tsd.t, raw_tsd.d, deg=poly_degree)
    bleach_fit = np.polyval(poly_coeffs, raw_tsd.t)
    detrended_data = raw_tsd.d - bleach_fit
    
    detrended_tsd = pna.Tsd(t=raw_tsd.t, d=detrended_data, time_units='s')
    bleach_fit_tsd = pna.Tsd(t=raw_tsd.t, d=bleach_fit, time_units='s')
    
    return detrended_tsd, bleach_fit_tsd

def correct_motion(signal_tsd, control_tsd):
    """
    Performs motion correction using a control signal (isosbestic).

    It regresses the control signal onto the main signal and subtracts the
    scaled motion artifact.

    Args:
        signal_tsd (pna.Tsd): The GCaMP signal Tsd (ideally after photobleaching correction).
        control_tsd (pna.Tsd): The Isosbestic signal Tsd (ideally after photobleaching correction).

    Returns:
        pna.Tsd: The motion-corrected GCaMP signal.
    """
    # Ensure signals are aligned to the same time points via interpolation if necessary
    if not np.array_equal(signal_tsd.t, control_tsd.t):
        control_aligned_d = np.interp(signal_tsd.t, control_tsd.t, control_tsd.d)
        control_tsd_aligned = pna.Tsd(t=signal_tsd.t, d=control_aligned_d, time_units='s')
    else:
        control_tsd_aligned = control_tsd

    # Reshape data for sklearn
    X_isos = control_tsd_aligned.d.reshape(-1, 1)
    y_gcamp = signal_tsd.d.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X_isos, y_gcamp)
    
    motion_artifact = model.predict(X_isos)
    
    corrected_data = signal_tsd.d - motion_artifact.flatten()
    motion_corrected_tsd = pna.Tsd(t=signal_tsd.t, d=corrected_data, time_units='s')
    
    return motion_corrected_tsd

def calculate_dff(signal_tsd, f0_tsd, epsilon=1e-9):
    """
    Calculates dF/F.

    Args:
        signal_tsd (pna.Tsd): The motion-corrected GCaMP signal.
        f0_tsd (pna.Tsd): The baseline fluorescence (F0) signal, typically the
                           photobleaching fit from the original GCaMP signal.
        epsilon (float): A small value to add to the denominator to prevent division by zero.

    Returns:
        pna.Tsd: The calculated dF/F signal.
    """
    f0_for_dff_data = f0_tsd.d
    if not np.array_equal(signal_tsd.t, f0_tsd.t):
        f0_for_dff_data = np.interp(signal_tsd.t, f0_tsd.t, f0_tsd.d)
        
    dff_values = signal_tsd.d / (f0_for_dff_data + epsilon)
    dff_values[np.isinf(dff_values)] = np.nan
    
    dff_tsd = pna.Tsd(t=signal_tsd.t, d=dff_values, time_units='s')
    return dff_tsd

def extract_epochs(signal_tsd, event_times_ts, window_before_sec, window_after_sec):
    """
    Extracts epochs (snippets) of a signal around given event times.

    Args:
        signal_tsd (pna.Tsd): The dF/F time series data.
        event_times_ts (pna.Ts): The timestamps of the events to align to.
        window_before_sec (float): The time in seconds to include before each event.
        window_after_sec (float): The time in seconds to include after each event.

    Returns:
        np.ndarray: A 2D numpy array where each row is an epoch. Returns an
                    empty array if no valid epochs are found.
    """
    fs = signal_tsd.rate
    samples_before = int(window_before_sec * fs)
    samples_after = int(window_after_sec * fs)
    total_epoch_samples = samples_before + samples_after
    
    all_epochs = []
    
    for event_time in event_times_ts.t:
        event_idx_in_signal = np.argmin(np.abs(signal_tsd.t - event_time))
        start_idx = event_idx_in_signal - samples_before
        end_idx = event_idx_in_signal + samples_after
        
        if start_idx >= 0 and end_idx <= len(signal_tsd.d):
            epoch = signal_tsd.d[start_idx:end_idx]
            if len(epoch) == total_epoch_samples:
                all_epochs.append(epoch)
                
    return np.array(all_epochs)

def get_rewarded_trial_flags(cue_epochs_array, dff_tsd, cue_onsets_ts, reward_onsets_ts, 
                             window_before_cue_sec, window_after_cue_sec, reward_search_window_sec):
    """
    Identifies which cue epochs were followed by a reward.

    Args:
        cue_epochs_array (np.ndarray): The array of extracted cue epochs.
        dff_tsd (pna.Tsd): The main dF/F signal Tsd.
        cue_onsets_ts (pna.Ts): Timestamps of all cue onsets.
        reward_onsets_ts (pna.Ts): Timestamps of all reward onsets.
        window_before_cue_sec (float): The pre-event window used to create epochs.
        window_after_cue_sec (float): The post-event window used to create epochs.
        reward_search_window_sec (float): The window after a cue to search for a reward.

    Returns:
        tuple: A tuple containing:
            - list[bool]: A list of booleans, same length as the number of valid epochs,
                          True if the cue was rewarded.
            - np.ndarray: The absolute timestamps of the cues that correspond to the epochs.
    """
    fs = dff_tsd.rate
    samples_before_cue = int(window_before_cue_sec * fs)
    samples_after_cue = int(window_after_cue_sec * fs)
    total_epoch_samples_cue = samples_before_cue + samples_after_cue

    actual_cue_times_for_epochs = []
    if cue_onsets_ts is not None and len(cue_onsets_ts.t) > 0:
        for cue_time in cue_onsets_ts.t:
            event_idx_in_signal = np.argmin(np.abs(dff_tsd.t - cue_time))
            start_idx = event_idx_in_signal - samples_before_cue
            end_idx = event_idx_in_signal + samples_after_cue
            if start_idx >= 0 and end_idx <= len(dff_tsd.d) and (end_idx - start_idx) == total_epoch_samples_cue:
                actual_cue_times_for_epochs.append(cue_time)

    if not actual_cue_times_for_epochs:
        return [], np.array([])
        
    actual_cue_times_for_epochs = np.array(actual_cue_times_for_epochs)

    is_rewarded_cue = []
    if reward_onsets_ts is None or not hasattr(reward_onsets_ts, 't') or len(reward_onsets_ts.t) == 0:
        return [False] * len(actual_cue_times_for_epochs), actual_cue_times_for_epochs
    
    for cue_time in actual_cue_times_for_epochs:
        reward_found = np.any((reward_onsets_ts.t > cue_time) & (reward_onsets_ts.t <= cue_time + reward_search_window_sec))
        is_rewarded_cue.append(reward_found)
        
    return is_rewarded_cue, actual_cue_times_for_epochs

def calculate_event_triggered_average(epochs_array):
    """
    Calculates the event-triggered average and SEM from an array of epochs.

    Args:
        epochs_array (np.ndarray): A 2D array where each row is an epoch.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The mean trace (average dF/F).
            - np.ndarray: The standard error of the mean (SEM).
            Returns (None, None) if input is empty.
    """
    if epochs_array.size == 0:
        return None, None
    
    avg_dff = np.mean(epochs_array, axis=0)
    sem_dff = scipy.stats.sem(epochs_array, axis=0)
    
    return avg_dff, sem_dff 