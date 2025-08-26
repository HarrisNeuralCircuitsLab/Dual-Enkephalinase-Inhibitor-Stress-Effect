# fp_signal_processing.py
# This module will contain functions for processing raw behavioral and fiber photometry signals.

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter # For smoothing
from sklearn.linear_model import LinearRegression # For motion correction
# from scipy.signal import butter, filtfilt # Example for potential filtering later
import traceback
import pynapple as pna

def demultiplex_fp_data(fp_raw_df: pd.DataFrame, 
                        led_state_A: int, 
                        led_state_B: int, 
                        ignore_led_state: int = None):
    """
    Demultiplexes interleaved fiber photometry data from a single signal column 
    based on LedState values. Normalizes the 'time_sec' column to start from 0
    based on the first timestamp in the filtered data for each signal.
    Also estimates and returns the median sampling rate.

    Args:
        fp_raw_df (pd.DataFrame): DataFrame containing raw FP data with columns 
                                  'ComputerTimestamp_sec', 'LedState', and 'G0'.
        led_state_A (int): The LedState value for the first signal channel.
        led_state_B (int): The LedState value for the second signal channel.
        ignore_led_state (int, optional): A LedState value to ignore. Defaults to None.

    Returns:
        dict: A dictionary containing:
              'signal_A_data': pd.DataFrame({'time_sec': normalized_time_A, 'signal': ...}),
              'signal_B_data': pd.DataFrame({'time_sec': normalized_time_B, 'signal': ...}),
              'estimated_sampling_rate_hz': float (median sampling rate of signal A),
              'first_timestamp_sec_A': float (original first timestamp of signal A),
              'first_timestamp_sec_B': float (original first timestamp of signal B)
    """
    print(f"Demultiplexing signals for LedState {led_state_A} and LedState {led_state_B}...")
    if ignore_led_state is not None:
        fp_filtered_df = fp_raw_df[fp_raw_df['LedState'] != ignore_led_state].copy()
        print(f"  Ignoring LedState {ignore_led_state}. Filtered data points: {len(fp_filtered_df)}")
    else:
        fp_filtered_df = fp_raw_df.copy()

    data_A = fp_filtered_df[fp_filtered_df['LedState'] == led_state_A][['ComputerTimestamp_sec', 'G0']].rename(
        columns={'ComputerTimestamp_sec': 'time_sec', 'G0': 'signal'}
    )
    data_B = fp_filtered_df[fp_filtered_df['LedState'] == led_state_B][['ComputerTimestamp_sec', 'G0']].rename(
        columns={'ComputerTimestamp_sec': 'time_sec', 'G0': 'signal'}
    )
    
    print(f"  Found {len(data_A)} points for LedState {led_state_A} (Signal A)")
    print(f"  Found {len(data_B)} points for LedState {led_state_B} (Signal B)")

    # Reset index for cleaner dataframes
    data_A.reset_index(drop=True, inplace=True)
    data_B.reset_index(drop=True, inplace=True)
    
    if data_A.empty:
        print(f"Warning: No data found for LedState {led_state_A} (Signal A).")
        signal_A_df = pd.DataFrame(columns=['time_sec', 'signal'])
        first_ts_A = np.nan
        sampling_rate_A = np.nan
    else:
        first_ts_A = data_A['time_sec'].iloc[0]
        time_A_normalized = data_A['time_sec'] - first_ts_A
        signal_A_df = pd.DataFrame({'time_sec': time_A_normalized, 'signal': data_A['signal'].values})
        if len(time_A_normalized) > 1:
            sampling_rate_A = 1.0 / np.median(np.diff(time_A_normalized))
        else:
            sampling_rate_A = np.nan
        print(f"Found {len(signal_A_df)} data points for LedState {led_state_A} (Signal A). Original first timestamp: {first_ts_A:.4f}s. Normalized time starts at 0.")
        if not np.isnan(sampling_rate_A):
            print(f"  Estimated median sampling rate for Signal A: {sampling_rate_A:.2f} Hz")

    if data_B.empty:
        print(f"Warning: No data found for LedState {led_state_B} (Signal B).")
        signal_B_df = pd.DataFrame(columns=['time_sec', 'signal'])
        first_ts_B = np.nan
    else:
        first_ts_B = data_B['time_sec'].iloc[0]
        time_B_normalized = data_B['time_sec'] - first_ts_B
        signal_B_df = pd.DataFrame({'time_sec': time_B_normalized, 'signal': data_B['signal'].values})
        print(f"Found {len(signal_B_df)} data points for LedState {led_state_B} (Signal B). Original first timestamp: {first_ts_B:.4f}s. Normalized time starts at 0.")

    if ignore_led_state is not None:
        num_ignored = len(fp_filtered_df[fp_filtered_df['LedState'] == ignore_led_state])
        if num_ignored > 0:
            print(f"Ignored {num_ignored} data points for LedState {ignore_led_state}.")

    # Use Signal A's sampling rate as the primary estimate, assuming they are interleaved from the same system.
    # If Signal A is empty, this will be NaN.
    final_estimated_sampling_rate = sampling_rate_A

    return {
        'signal_A_data': signal_A_df, 
        'signal_B_data': signal_B_df,
        'estimated_sampling_rate_hz': final_estimated_sampling_rate,
        'first_timestamp_sec_A': first_ts_A, # Store original first timestamp for reference
        'first_timestamp_sec_B': first_ts_B
    }

def preprocess_fp_signal(main_signal_df: pd.DataFrame, 
                         isosbestic_signal_df: pd.DataFrame, 
                         smoothing_window_size: int = 11,
                         smoothing_polyorder: int = 2,
                         photobleaching_poly_degree: int = 2):
    """
    Preprocesses a main fiber photometry signal using an isosbestic control signal.
    Assumes 'time_sec' in input DataFrames is already normalized (starts near 0).
    Steps:
    1. Align signals: Interpolate isosbestic signal to main signal's timestamps.
    2. Smoothing: Apply Savitzky-Golay filter to both signals.
    3. Motion Correction: Fit isosbestic to main signal and subtract scaled isosbestic.
    4. Photobleaching Correction: Fit a polynomial to the motion-corrected main signal 
       to model photobleaching and subtract it. This fitted polynomial becomes F0.
    5. dF/F Calculation: Calculate (F - F0) / F0.

    Args:
        main_signal_df (pd.DataFrame): DataFrame with 'time_sec' (normalized) and 'signal'.
        isosbestic_signal_df (pd.DataFrame): DataFrame with 'time_sec' (normalized) and 'signal'.
        smoothing_window_size (int): Window size for Savitzky-Golay filter. Must be odd.
        smoothing_polyorder (int): Polynomial order for Savitzky-Golay filter.
        photobleaching_poly_degree (int): Degree of polynomial for photobleaching fit.

    Returns:
        pd.DataFrame: DataFrame with 'time_sec' and 'dff' (processed signal).
                      Returns None if preprocessing fails.
    """
    if main_signal_df.empty or 'time_sec' not in main_signal_df or 'signal' not in main_signal_df:
        print("Warning: Main signal DataFrame is empty or missing columns. Skipping preprocessing.")
        return None
    if isosbestic_signal_df.empty or 'time_sec' not in isosbestic_signal_df or 'signal' not in isosbestic_signal_df:
        print("Warning: Isosbestic signal DataFrame is empty or missing columns. Skipping preprocessing.")
        # Optionally, could proceed with main_signal only if isosbestic is missing and skip motion correction
        return None 
    
    # Check if time is normalized (starts near zero)
    if not (main_signal_df['time_sec'].iloc[0] < 1.0 and isosbestic_signal_df['time_sec'].iloc[0] < 1.0) :
         print(f"Warning: Input 'time_sec' for main ({main_signal_df['time_sec'].iloc[0]:.2f}s) or isosbestic ({isosbestic_signal_df['time_sec'].iloc[0]:.2f}s) does not appear to be normalized (i.e., start near 0). Results may be incorrect.")

    try:
        # Ensure window size is odd for Savitzky-Golay
        if smoothing_window_size % 2 == 0:
            smoothing_window_size += 1
            # print(f"Adjusted smoothing_window_size to be odd: {smoothing_window_size}") # Less verbose

        main_s = main_signal_df.copy()
        isos_s = isosbestic_signal_df.copy()

        # 1. Align signals (interpolation)
        # Ensure isos_s['time_sec'] is monotonically increasing for np.interp
        isos_s = isos_s.sort_values(by='time_sec').drop_duplicates(subset=['time_sec'], keep='first')
        main_s_time_for_interp = main_s['time_sec'].values # Use main signal's time as reference
        
        # Check if time vectors are identical, if so, skip interpolation
        if len(main_s_time_for_interp) == len(isos_s['time_sec'].values) and np.allclose(main_s_time_for_interp, isos_s['time_sec'].values):
            isos_s_aligned_signal = isos_s['signal'].values
        else:
            if len(isos_s['time_sec']) < 2 or len(main_s_time_for_interp) < 1 : # not enough points to interpolate
                 print("Warning: Not enough data points in isosbestic or main signal for interpolation. Using raw isosbestic signal if lengths match, else error.")
                 if len(main_s['signal']) == len(isos_s['signal']):
                      isos_s_aligned_signal = isos_s['signal'].values
                 else:
                      raise ValueError("Cannot align signals due to insufficient data points and unequal lengths post-filtering.")
            else:
                 isos_s_aligned_signal = np.interp(main_s_time_for_interp, isos_s['time_sec'].values, isos_s['signal'].values)
        
        # 2. Smoothing
        if len(main_s['signal']) < smoothing_window_size:
            # print(f"Warning: Main signal length ({len(main_s['signal'])}) < smoothing window. Skipping smoothing.")
            main_s_smoothed = main_s['signal'].values
        else:
            main_s_smoothed = savgol_filter(main_s['signal'].values, smoothing_window_size, smoothing_polyorder)
        
        if len(isos_s_aligned_signal) < smoothing_window_size:
            # print(f"Warning: Isosbestic length ({len(isos_s_aligned_signal)}) < smoothing window. Skipping smoothing.")
            isos_s_smoothed = isos_s_aligned_signal
        else:
            isos_s_smoothed = savgol_filter(isos_s_aligned_signal, smoothing_window_size, smoothing_polyorder)

        # 3. Motion Correction
        X_isos = isos_s_smoothed.reshape(-1, 1)
        y_main = main_s_smoothed.reshape(-1, 1)
        
        main_signal_motion_corrected = main_s_smoothed # Default if fit fails
        try:
            if X_isos.shape[0] > 0 : # Ensure there's data to fit
                regressor = LinearRegression()
                regressor.fit(X_isos, y_main)
                slope = regressor.coef_[0][0]
                # intercept = regressor.intercept_[0] # Not using intercept for correction here
                main_signal_motion_corrected = main_s_smoothed - (slope * isos_s_smoothed)
            else:
                print("Warning: Isosbestic signal is empty after alignment/smoothing. Skipping motion correction regression.")

        except np.linalg.LinAlgError as lae:
            print(f"Linear algebra error during motion correction fitting: {lae}. Using uncorrected main signal.")
        except ValueError as ve: # Can happen if X_isos or y_main are empty or have NaNs not handled
            print(f"ValueError during motion correction fitting (possibly empty arrays or NaNs): {ve}. Using uncorrected main signal.")


        # 4. Photobleaching Correction (on motion-corrected signal)
        time_points_for_polyfit = main_s['time_sec'].values # Already normalized
        
        if len(time_points_for_polyfit) <= photobleaching_poly_degree:
             # print(f"Warning: Not enough data points for photobleaching polyfit degree {photobleaching_poly_degree}.")
             F0_photobleaching = np.mean(main_signal_motion_corrected) 
             signal_photobleaching_corrected = main_signal_motion_corrected - F0_photobleaching
        else:
            poly_coeffs = np.polyfit(time_points_for_polyfit, main_signal_motion_corrected, photobleaching_poly_degree)
            F0_photobleaching = np.polyval(poly_coeffs, time_points_for_polyfit)
            signal_photobleaching_corrected = main_signal_motion_corrected - F0_photobleaching

        # 5. dF/F Calculation
        epsilon = 1e-9 
        f0_for_dff = F0_photobleaching.copy()
        # To prevent issues with F0 being zero or negative, ensure it's positive by using the mean of the motion_corrected signal
        # as a baseline for F0 if the polynomial fit results in non-positive values, or add a small positive constant.
        # A common practice is to ensure F0 is robust.
        # If F0_photobleaching represents the baseline, then dF/F = (F - F0) / F0.
        # A robust F0 should be positive. Let's use the mean of the *original main smoothed signal* if polyfit is too low.
        # Or, shift the polyfit F0 to be positive.

        # If poly_fit F0 is problematic (e.g. all zeros or negative), fall back to a global median of motion corrected signal.
        if np.allclose(F0_photobleaching, 0) or np.median(F0_photobleaching) <= epsilon:
            print("Warning: Fitted F0 from polynomial is problematic (zero or negative median). Using median of motion-corrected signal as F0.")
            robust_F0_val = np.median(main_signal_motion_corrected)
            if robust_F0_val <= epsilon: # If median is also too small
                robust_F0_val = np.mean(main_signal_motion_corrected) # Try mean
                if robust_F0_val <= epsilon: robust_F0_val = epsilon # Absolute fallback
            
            f0_for_dff = np.full_like(main_signal_motion_corrected, robust_F0_val)
            signal_photobleaching_corrected = main_signal_motion_corrected - f0_for_dff # F - F0_robust

        min_f0_val = np.min(f0_for_dff)
        if min_f0_val <= 0: # If still zero/negative after potential fallback
            f0_for_dff_shifted = f0_for_dff + (-min_f0_val + epsilon) # Shift to be positive
        else:
            f0_for_dff_shifted = f0_for_dff
        
        dff = signal_photobleaching_corrected / (f0_for_dff_shifted + epsilon) # (F-F0)/F0_shifted

        processed_df = pd.DataFrame({
            'time_sec': main_s['time_sec'].values, 
            'dff': dff, 
            'debug_F0': F0_photobleaching, # The actual polynomial fit for F0
            'debug_motion_corrected': main_signal_motion_corrected
        })
        # print("Preprocessing complete. dF/F calculated.") # Less verbose
        return processed_df

    except Exception as e:
        print(f"Error during preprocessing in preprocess_fp_signal: {e}")
        traceback.print_exc()
        return None

# Helper function for edge detection (can be called by get_event_timestamps)
def _detect_fp_edges(signal_series: pd.Series, threshold: float, edge_type: str = 'rising'):
    """
    Detects rising or falling edges in a pandas Series.
    Assumes signal_series.values is a 1D numpy array.
    Returns indices of the edges.
    """
    if not isinstance(signal_series, pd.Series):
        raise TypeError("signal_series must be a pandas Series.")
    
    signal_values = signal_series.values
    if threshold is None: # Try to infer if signal is already binary (0s and 1s)
        if set(np.unique(signal_values)).issubset({0, 1}):
            binary_signal = signal_values.astype(bool)
        else: # If not binary and no threshold, this won't work well.
            raise ValueError("Threshold must be provided if signal is not already binary (0s and 1s).")
    else:
        binary_signal = signal_values > threshold
        
    diff_signal = np.diff(binary_signal.astype(int))
    
    if edge_type == 'rising':
        edges_indices = np.where(diff_signal > 0)[0] + 1
    elif edge_type == 'falling':
        edges_indices = np.where(diff_signal < 0)[0] + 1
    else:
        raise ValueError("edge_type must be 'rising' or 'falling'.")
    return edges_indices

def get_event_timestamps(behavioral_df: pd.DataFrame, 
                         signal_column_name: str, 
                         threshold: float, 
                         edge_type: str = 'rising') -> np.ndarray:
    """
    Extracts event timestamps based on edge detection in a specified signal column.

    Args:
        behavioral_df (pd.DataFrame): DataFrame containing behavioral data,
                                      must include 'time_sec' and signal_column_name.
        signal_column_name (str): Name of the column containing the event signal.
        threshold (float): Threshold for detecting edges in the signal_column_name.
                           If None, assumes signal is already binary (0s and 1s)
                           and looks for changes from 0 to 1 (rising) or 1 to 0 (falling).
        edge_type (str): 'rising' or 'falling'. Defaults to 'rising'.

    Returns:
        np.ndarray: Array of event timestamps in seconds. Returns empty array if no events.
    """
    if signal_column_name not in behavioral_df.columns:
        print(f"Warning: Column '{signal_column_name}' not found in behavioral_df. Cannot get event timestamps.")
        return np.array([])
    if 'time_sec' not in behavioral_df.columns:
        print("Warning: Column 'time_sec' not found in behavioral_df. Cannot get event timestamps.")
        return np.array([])

    event_signal = behavioral_df[signal_column_name]
    edge_indices = _detect_fp_edges(event_signal, threshold, edge_type)
    
    if edge_indices.size > 0:
        event_times = behavioral_df['time_sec'].iloc[edge_indices].values
        print(f"Detected {len(event_times)} '{edge_type}' edges for '{signal_column_name}' at threshold {threshold}.")
        return event_times
    else:
        print(f"No '{edge_type}' edges detected for '{signal_column_name}' at threshold {threshold}.")
        return np.array([])

def calculate_event_triggered_average(dff_df: pd.DataFrame, 
                                      event_times_on_fp_clock: np.ndarray, 
                                      window_before_event_sec: float, 
                                      window_after_event_sec: float,
                                      verbose: bool = True) -> pd.DataFrame:
    """
    Calculates the event-triggered average of a dF/F signal.

    Args:
        dff_df (pd.DataFrame): DataFrame with 'time_sec' and 'dff'.
        event_times_on_fp_clock (np.ndarray): Array of event timestamps, ALIGNED to the
                                             dff_df['time_sec'] clock.
        window_before_event_sec (float): Time window before event (e.g., 2.0 for 2s).
        window_after_event_sec (float): Time window after event (e.g., 5.0 for 5s).
        verbose (bool): If True, prints detailed messages.

    Returns:
        pd.DataFrame: DataFrame with 'time_relative_to_event_sec' and 'mean_dff', 
                      'sem_dff'. Returns empty DataFrame if no valid epochs.
    """
    if dff_df.empty or 'time_sec' not in dff_df or 'dff' not in dff_df:
        if verbose: print("dff_df is empty or missing required columns. Cannot calculate ETA.")
        return pd.DataFrame(columns=['time_relative_to_event_sec', 'mean_dff', 'sem_dff'])
    
    if event_times_on_fp_clock.size == 0:
        if verbose: print("No event times provided. Cannot calculate ETA.")
        return pd.DataFrame(columns=['time_relative_to_event_sec', 'mean_dff', 'sem_dff'])

    # Estimate sampling rate from dff_df
    if len(dff_df['time_sec']) < 2:
        if verbose: print("Not enough data points in dff_df to estimate sampling rate.")
        return pd.DataFrame(columns=['time_relative_to_event_sec', 'mean_dff', 'sem_dff'])
    
    estimated_dt = np.mean(np.diff(dff_df['time_sec']))
    if estimated_dt <= 0:
        if verbose: print(f"Estimated dt is non-positive ({estimated_dt:.4f}s). Cannot proceed.")
        return pd.DataFrame(columns=['time_relative_to_event_sec', 'mean_dff', 'sem_dff'])
        
    sampling_rate_hz = 1.0 / estimated_dt
    if verbose: print(f"Estimated sampling rate: {sampling_rate_hz:.2f} Hz (dt={estimated_dt*1000:.2f} ms)")

    samples_before = int(window_before_event_sec * sampling_rate_hz)
    samples_after = int(window_after_event_sec * sampling_rate_hz)
    total_samples_per_epoch = samples_before + samples_after + 1 # +1 for the event sample itself

    all_epochs = []
    fp_time_vector = dff_df['time_sec'].values
    fp_dff_vector = dff_df['dff'].values

    for event_time in event_times_on_fp_clock:
        # Find the index in fp_time_vector closest to the event_time
        event_idx_in_fp = np.searchsorted(fp_time_vector, event_time) # More robust than argmin(abs(diff)) for sorted time
        
        # Ensure the event_idx is within bounds for fp_time_vector
        if event_idx_in_fp >= len(fp_time_vector): # event is after last fp sample
            event_idx_in_fp = len(fp_time_vector) -1 
        
        # Check actual time difference if searchsorted picked a slightly off index
        if event_idx_in_fp > 0 and \
           np.abs(fp_time_vector[event_idx_in_fp-1] - event_time) < np.abs(fp_time_vector[event_idx_in_fp] - event_time):
            event_idx_in_fp = event_idx_in_fp -1

        start_idx = event_idx_in_fp - samples_before
        end_idx = event_idx_in_fp + samples_after + 1 # Slice goes up to, but not including, end_idx

        if start_idx >= 0 and end_idx <= len(fp_dff_vector):
            epoch = fp_dff_vector[start_idx:end_idx]
            if len(epoch) == total_samples_per_epoch: # Ensure consistent length
                all_epochs.append(epoch)
            elif verbose:
                print(f"Skipping epoch around event_time {event_time:.2f}s: extracted length {len(epoch)}, expected {total_samples_per_epoch}.")
        elif verbose:
            print(f"Skipping epoch around event_time {event_time:.2f}s: window [{start_idx}:{end_idx}] out of bounds [0:{len(fp_dff_vector)}].")


    if not all_epochs:
        if verbose: print("No valid epochs found to average.")
        return pd.DataFrame(columns=['time_relative_to_event_sec', 'mean_dff', 'sem_dff'])

    all_epochs_array = np.array(all_epochs)
    mean_dff_trace = np.mean(all_epochs_array, axis=0)
    sem_dff_trace = np.std(all_epochs_array, axis=0) / np.sqrt(all_epochs_array.shape[0]) # SEM

    # Create time vector for the averaged trace
    # Time point 0 is the event itself.
    time_relative = np.linspace(-window_before_event_sec, window_after_event_sec, total_samples_per_epoch)
    
    if verbose: print(f"Averaged {len(all_epochs)} epochs.")

    return pd.DataFrame({
        'time_relative_to_event_sec': time_relative,
        'mean_dff': mean_dff_trace,
        'sem_dff': sem_dff_trace
    })

def get_individual_event_epochs(dff_df: pd.DataFrame, 
                                event_times_on_fp_clock: np.ndarray, 
                                window_before_event_sec: float, 
                                window_after_event_sec: float,
                                verbose: bool = True) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Extracts individual dF/F epochs around specified event times.

    Args:
        dff_df (pd.DataFrame): DataFrame with 'time_sec' (normalized) and 'dff'.
        event_times_on_fp_clock (np.ndarray): Array of event timestamps, ALIGNED to the
                                             dff_df['time_sec'] clock.
        window_before_event_sec (float): Time window before event.
        window_after_event_sec (float): Time window after event.
        verbose (bool): If True, prints detailed messages.

    Returns:
        tuple[np.ndarray, np.ndarray] | tuple[None, None]: 
            - all_epochs_array (np.ndarray): 2D array where each row is an event epoch (dF/F values).
                                             Shape: (num_valid_epochs, num_samples_per_epoch)
            - time_relative_to_event_sec (np.ndarray): 1D array of time points for the epochs.
            Returns (None, None) if no valid epochs are found or on error.
    """
    if dff_df.empty or 'time_sec' not in dff_df or 'dff' not in dff_df:
        if verbose: print("dff_df is empty or missing required columns. Cannot extract epochs.")
        return None, None
    
    if event_times_on_fp_clock.size == 0:
        if verbose: print("No event times provided. Cannot extract epochs.")
        return None, None

    if len(dff_df['time_sec']) < 2:
        if verbose: print("Not enough data points in dff_df to estimate sampling rate.")
        return None, None
    
    estimated_dt = np.mean(np.diff(dff_df['time_sec']))
    if estimated_dt <= 0:
        if verbose: print(f"Estimated dt is non-positive ({estimated_dt:.4f}s). Cannot proceed.")
        return None, None
        
    sampling_rate_hz = 1.0 / estimated_dt
    # if verbose: print(f"(get_individual_event_epochs) Estimated sampling rate: {sampling_rate_hz:.2f} Hz") # Less verbose here

    samples_before = int(window_before_event_sec * sampling_rate_hz)
    samples_after = int(window_after_event_sec * sampling_rate_hz)
    total_samples_per_epoch = samples_before + samples_after + 1

    all_epochs = []
    fp_time_vector = dff_df['time_sec'].values
    fp_dff_vector = dff_df['dff'].values

    for event_time in event_times_on_fp_clock:
        event_idx_in_fp = np.searchsorted(fp_time_vector, event_time)
        if event_idx_in_fp >= len(fp_time_vector): event_idx_in_fp = len(fp_time_vector) - 1
        if event_idx_in_fp > 0 and \
           np.abs(fp_time_vector[event_idx_in_fp-1] - event_time) < np.abs(fp_time_vector[event_idx_in_fp] - event_time):
            event_idx_in_fp -= 1

        start_idx = event_idx_in_fp - samples_before
        end_idx = event_idx_in_fp + samples_after + 1

        if start_idx >= 0 and end_idx <= len(fp_dff_vector):
            epoch = fp_dff_vector[start_idx:end_idx]
            if len(epoch) == total_samples_per_epoch:
                all_epochs.append(epoch)
            # elif verbose: # Be less verbose for individual epoch extraction unless debugging
            #     print(f"Skipping epoch for event {event_time:.2f}s: length {len(epoch)}, expected {total_samples_per_epoch}.")
        # elif verbose:
        #     print(f"Skipping epoch for event {event_time:.2f}s: window out of bounds.")

    if not all_epochs:
        if verbose: print("No valid individual epochs found.")
        return None, None

    all_epochs_array = np.array(all_epochs)
    time_relative = np.linspace(-window_before_event_sec, window_after_event_sec, total_samples_per_epoch)
    
    if verbose: print(f"Extracted {len(all_epochs)} individual epochs.")
    return all_epochs_array, time_relative

def align_events_with_drift_correction(behavioral_data, session_meta, dff_tsd, 
                                       cue_column='sound_ttl', cue_threshold=500,
                                       reward_column='reward_signal', reward_threshold=500):
    """
    Detects behavioral events and aligns them to the FP timeline using the
    full drift correction logic from the notebook.

    Args:
        behavioral_data (pd.DataFrame): DataFrame with behavioral data.
        session_meta (dict): Dictionary with session metadata.
        dff_tsd (pna.Tsd): The final dF/F Tsd object, used to filter events.
        cue_column (str): Name of the cue signal column.
        cue_threshold (float): Threshold for cue detection.
        reward_column (str): Name of the reward signal column.
        reward_threshold (float): Threshold for reward detection.

    Returns:
        tuple: A tuple of pna.Ts objects (cue_onsets_ts, reward_onsets_ts).
    """
    # --- Step 1: Apply Gradual Drift Correction to Behavioral Timestamps ---
    corrected_behavioral_data = behavioral_data.copy()
    T_fp_start_on_drift_corrected_behav_timeline = 0.0

    original_timestamps = behavioral_data['time_sec'].values
    new_corrected_timestamps = np.zeros_like(original_timestamps)
    if len(original_timestamps) > 0:
        new_corrected_timestamps[0] = original_timestamps[0]

    T_behav_fp_start_orig_clock = session_meta.get('fp_start_time_on_behavioral_clock_sec')
    Dur_behav_ttl_orig_clock = session_meta.get('fp_duration_from_ttl_sec')
    Dur_fp_csv = session_meta.get('fp_duration_from_csv_sec')

    apply_scaling = (T_behav_fp_start_orig_clock is not None and
                     Dur_behav_ttl_orig_clock is not None and Dur_behav_ttl_orig_clock > 0 and
                     Dur_fp_csv is not None and
                     not np.isnan(T_behav_fp_start_orig_clock) and
                     not np.isnan(Dur_behav_ttl_orig_clock) and
                     not np.isnan(Dur_fp_csv))

    if apply_scaling:
        interval_scale_factor = Dur_fp_csv / Dur_behav_ttl_orig_clock
        T_behav_fp_end_orig_clock = T_behav_fp_start_orig_clock + Dur_behav_ttl_orig_clock
    else:
        interval_scale_factor = 1.0
        T_behav_fp_end_orig_clock = None

    for i in range(1, len(original_timestamps)):
        dt_original = original_timestamps[i] - original_timestamps[i-1]
        current_interval_effective_scale = 1.0
        if apply_scaling:
            interval_start_orig_clock = original_timestamps[i-1]
            if (interval_start_orig_clock >= T_behav_fp_start_orig_clock and
                interval_start_orig_clock < T_behav_fp_end_orig_clock):
                current_interval_effective_scale = interval_scale_factor
        
        dt_corrected = dt_original * current_interval_effective_scale
        new_corrected_timestamps[i] = new_corrected_timestamps[i-1] + dt_corrected

    corrected_behavioral_data['time_sec_drift_corrected'] = new_corrected_timestamps

    if apply_scaling:
        T_fp_start_on_drift_corrected_behav_timeline = np.interp(T_behav_fp_start_orig_clock, original_timestamps, new_corrected_timestamps)
    else:
        T_fp_start_on_drift_corrected_behav_timeline = T_behav_fp_start_orig_clock if T_behav_fp_start_orig_clock is not None else 0.0

    # --- Step 2: Detect Events and Map to Corrected Timeline ---
    raw_cue_ts = get_event_timestamps(behavioral_data, cue_column, cue_threshold, 'rising')
    raw_reward_ts = get_event_timestamps(behavioral_data, reward_column, reward_threshold, 'rising')
    
    cue_onsets_drift_corrected = np.interp(raw_cue_ts, original_timestamps, new_corrected_timestamps)
    reward_onsets_drift_corrected = np.interp(raw_reward_ts, original_timestamps, new_corrected_timestamps)

    # --- Step 3: Align to FP timeline (t=0) and filter ---
    cue_onsets_fp_time = cue_onsets_drift_corrected - T_fp_start_on_drift_corrected_behav_timeline
    reward_onsets_fp_time = reward_onsets_drift_corrected - T_fp_start_on_drift_corrected_behav_timeline

    if dff_tsd is not None and not dff_tsd.empty:
        min_time, max_time = dff_tsd.t[0], dff_tsd.t[-1]
        cue_onsets_fp_time = cue_onsets_fp_time[(cue_onsets_fp_time >= min_time) & (cue_onsets_fp_time <= max_time)]
        reward_onsets_fp_time = reward_onsets_fp_time[(reward_onsets_fp_time >= min_time) & (reward_onsets_fp_time <= max_time)]

    return pna.Ts(t=cue_onsets_fp_time), pna.Ts(t=reward_onsets_fp_time)

print("fp_signal_processing.py loaded with demultiplex_fp_data, preprocess_fp_signal, get_event_timestamps, calculate_event_triggered_average, and get_individual_event_epochs functions.") 