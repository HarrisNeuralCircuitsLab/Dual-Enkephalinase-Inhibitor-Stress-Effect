"""
Signal processing utilities for behavioral analysis.
"""
import os
import numpy as np
import pandas as pd
import traceback
from scipy.io import loadmat

def detect_rising_edges(signal, threshold=500):
    """
    Detect rising edges in a signal (transitions from below to above threshold)
    """
    # Threshold the signal to create a binary signal
    binary_signal = signal > threshold
    
    # Detect rising edges (transitions from 0 to 1)
    edges = np.where(np.diff(binary_signal.astype(int)) > 0)[0] + 1
    
    return edges

def detect_falling_edges(signal, threshold=500):
    """
    Detect falling edges in a signal (transitions from above to below threshold)
    """
    # Threshold the signal to create a binary signal
    binary_signal = signal > threshold
    
    # Detect falling edges (transitions from 1 to 0)
    edges = np.where(np.diff(binary_signal.astype(int)) < 0)[0] + 1
    
    return edges

def analyze_lick_sensor_cut_time(session_data, df):
    """
    Analyze the percentage of time the lick sensor is cut (down) during cue vs ITI
    and during symmetrized pre-cue vs post-cue windows
    
    Parameters:
    -----------
    session_data : dict
        Dictionary containing session data
    df : pandas.DataFrame
        DataFrame containing the raw monitoring data
        
    Returns:
    --------
    dict
        Dictionary with lick sensor cut time metrics
    """
    trial_results = session_data['trial_results']
    lick_sensor_metrics = []
    
    # Threshold for considering lick sensor as "cut" (down)
    threshold = 500
    
    for trial in trial_results:
        trial_num = trial['trial_num']
        sound_start = trial['sound_start']  # This is the trial start
        sound_end = trial['sound_end']
        
        # Calculate cue duration
        cue_duration = sound_end - sound_start
        
        # --- Define fixed 1.5 second windows ---
        window_duration = 1.5
        post_cue_end = sound_start + window_duration
        pre_cue_window_start = sound_start - window_duration
        
        # Get next trial start time or end of recording
        if trial_num < len(trial_results):
            # Get the start time of the next trial from the trial_results list
            next_trial_start = trial_results[trial_num -1 + 1]['sound_start'] # trial_num is 1-based, list is 0-based
        else:
            next_trial_start = df['time_sec'].iloc[-1]

        # Define the ITI window start point (using the new post_cue_end)
        # Note: This might be different from previous definition which used sound_end + 5.0
        iti_start = post_cue_end
        # --- End of window definitions ---

        # Get indices for different phases
        cue_start_idx = np.argmin(np.abs(df['time_sec'] - sound_start))
        cue_end_idx = np.argmin(np.abs(df['time_sec'] - sound_end))
        post_cue_end_idx = np.argmin(np.abs(df['time_sec'] - post_cue_end)) # Index for end of 1.5s post-cue window
        next_trial_idx = np.argmin(np.abs(df['time_sec'] - next_trial_start))
        iti_start_idx = np.argmin(np.abs(df['time_sec'] - iti_start)) # Index for start of ITI analysis window
        
        # Get index for pre-cue window (if it exists in the recording)
        if pre_cue_window_start > df['time_sec'].iloc[0]:
            pre_cue_window_start_idx = np.argmin(np.abs(df['time_sec'] - pre_cue_window_start))
        else:
            # If pre-cue window goes before recording start, use the beginning of the recording
            pre_cue_window_start_idx = 0
            print(f"Warning: Trial {trial_num}: Pre-cue window starts before recording begins.")
        
        # Get lick signal during cue presentation
        lick_during_cue = df['lick_signal'].iloc[cue_start_idx:cue_end_idx].values
        
        # Get lick signal during ITI (from NEW post_cue_end to next trial start)
        lick_during_iti = df['lick_signal'].iloc[iti_start_idx:next_trial_idx].values # Use iti_start_idx
        
        # Get lick signal during pre-cue window (1.5s before cue start)
        lick_during_pre_cue = df['lick_signal'].iloc[pre_cue_window_start_idx:cue_start_idx].values
        
        # Get lick signal during post-cue window (1.5s after cue start)
        lick_during_post_cue = df['lick_signal'].iloc[cue_start_idx:post_cue_end_idx].values # Use post_cue_end_idx
        
        # Calculate absolute cut duration (in seconds) using the function
        post_cue_cut_duration = calculate_cut_duration_in_window(
            lick_downs_relative=trial['lick_downs_relative'],
            lick_ups_relative=trial['lick_ups_relative'],
            window_start_relative=0.0,  # Start at cue onset (which is relative 0.0)
            window_duration=window_duration  # Use the same 1.5s window
        )
        
        # Calculate percentage of time lick sensor is cut
        if len(lick_during_cue) > 0:
            pct_cut_during_cue = np.mean(lick_during_cue < threshold) * 100
        else:
            pct_cut_during_cue = 0
            
        if len(lick_during_iti) > 0:
            pct_cut_during_iti = np.mean(lick_during_iti < threshold) * 100
        else:
            pct_cut_during_iti = 0
        
        if len(lick_during_pre_cue) > 0:
            pct_cut_during_pre_cue = np.mean(lick_during_pre_cue < threshold) * 100
        else:
            pct_cut_during_pre_cue = 0
            
        if len(lick_during_post_cue) > 0:
            pct_cut_during_post_cue = np.mean(lick_during_post_cue < threshold) * 100
        else:
            pct_cut_during_post_cue = 0
        
        # Store metrics for this trial
        lick_sensor_metrics.append({
            'trial_num': trial_num,
            'pct_cut_during_cue': pct_cut_during_cue,
            'pct_cut_during_iti': pct_cut_during_iti,
            'pct_cut_during_pre_cue': pct_cut_during_pre_cue,
            'pct_cut_during_post_cue': pct_cut_during_post_cue,
            'post_cue_cut_duration': post_cue_cut_duration,  # Store the absolute duration in seconds
            'rewarded': trial['rewarded']
        })
    
    # Calculate session-wide metrics
    if lick_sensor_metrics:
        avg_pct_cut_cue = np.mean([t['pct_cut_during_cue'] for t in lick_sensor_metrics])
        avg_pct_cut_iti = np.mean([t['pct_cut_during_iti'] for t in lick_sensor_metrics])
        avg_pct_cut_pre_cue = np.mean([t['pct_cut_during_pre_cue'] for t in lick_sensor_metrics])
        avg_pct_cut_post_cue = np.mean([t['pct_cut_during_post_cue'] for t in lick_sensor_metrics])
        avg_post_cue_cut_duration = np.mean([t['post_cue_cut_duration'] for t in lick_sensor_metrics])
        
        # Calculate for rewarded trials only
        rewarded_metrics = [t for t in lick_sensor_metrics if t['rewarded']]
        if rewarded_metrics:
            avg_pct_cut_cue_rewarded = np.mean([t['pct_cut_during_cue'] for t in rewarded_metrics])
            avg_pct_cut_iti_rewarded = np.mean([t['pct_cut_during_iti'] for t in rewarded_metrics])
            avg_pct_cut_pre_cue_rewarded = np.mean([t['pct_cut_during_pre_cue'] for t in rewarded_metrics])
            avg_pct_cut_post_cue_rewarded = np.mean([t['pct_cut_during_post_cue'] for t in rewarded_metrics])
            avg_post_cue_cut_duration_rewarded = np.mean([t['post_cue_cut_duration'] for t in rewarded_metrics])
        else:
            avg_pct_cut_cue_rewarded = 0
            avg_pct_cut_iti_rewarded = 0
            avg_pct_cut_pre_cue_rewarded = 0
            avg_pct_cut_post_cue_rewarded = 0
            avg_post_cue_cut_duration_rewarded = 0
    else:
        avg_pct_cut_cue = 0
        avg_pct_cut_iti = 0
        avg_pct_cut_pre_cue = 0
        avg_pct_cut_post_cue = 0
        avg_post_cue_cut_duration = 0
        avg_pct_cut_cue_rewarded = 0
        avg_pct_cut_iti_rewarded = 0
        avg_pct_cut_pre_cue_rewarded = 0
        avg_pct_cut_post_cue_rewarded = 0
        avg_post_cue_cut_duration_rewarded = 0
    
    return {
        'trial_metrics': lick_sensor_metrics,
        'avg_pct_cut_cue': avg_pct_cut_cue,
        'avg_pct_cut_iti': avg_pct_cut_iti,
        'avg_pct_cut_pre_cue': avg_pct_cut_pre_cue,
        'avg_pct_cut_post_cue': avg_pct_cut_post_cue,
        'avg_post_cue_cut_duration': avg_post_cue_cut_duration,
        'avg_pct_cut_cue_rewarded': avg_pct_cut_cue_rewarded,
        'avg_pct_cut_iti_rewarded': avg_pct_cut_iti_rewarded,
        'avg_pct_cut_pre_cue_rewarded': avg_pct_cut_pre_cue_rewarded,
        'avg_pct_cut_post_cue_rewarded': avg_pct_cut_post_cue_rewarded,
        'avg_post_cue_cut_duration_rewarded': avg_post_cue_cut_duration_rewarded,
        'session_day': session_data['session_day']
    }

def process_monitoring_data(folder_path):
    """
    Process monitoring data from a session folder.
    It now checks for 'monitoring_data_FP.mat' as a fallback.
    """
    from modules.data_loader import extract_animal_id_from_folder, extract_session_day
    
    # Check for the new file name first, then fall back to the old one.
    monitoring_file_fp = os.path.join(folder_path, "monitoring_data_FP.mat")
    monitoring_file_orig = os.path.join(folder_path, "monitoring_data.mat")
    
    monitoring_file_to_load = None
    if os.path.exists(monitoring_file_fp):
        monitoring_file_to_load = monitoring_file_fp
    elif os.path.exists(monitoring_file_orig):
        monitoring_file_to_load = monitoring_file_orig
    
    if monitoring_file_to_load is None:
        print(f"Error: No monitoring data file ('monitoring_data.mat' or 'monitoring_data_FP.mat') found in {folder_path}")
        return None
    
    try:
        # Extract metadata from folder path
        animal_id = extract_animal_id_from_folder(folder_path)
        session_day = extract_session_day(folder_path)
        
        # Load the .mat file
        print(f"Loading {monitoring_file_to_load}...")
        monitoring_data = loadmat(monitoring_file_to_load)
        
        # Extract the data buffer and buffer index
        if 'data_buffer' not in monitoring_data:
            print(f"Error: data_buffer not found in {monitoring_file_to_load}")
            return None
        
        data_buffer = monitoring_data['data_buffer']
        buffer_idx = monitoring_data['buffer_idx'][0][0] if 'buffer_idx' in monitoring_data else data_buffer.shape[0]
        
        # Extract the valid data (up to buffer_idx)
        raw_data = data_buffer[:buffer_idx]
        
        # Define column names
        columns = ['timestamp', 'sound_envelope', 'sound_ttl', 'reward_signal', 'lick_signal', 'fp_ttl']
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(raw_data, columns=columns)
        
        # Convert timestamp to seconds (from microseconds)
        df['time_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1e6
        
        # Detect sound onsets and offsets
        sound_onset_indices = detect_rising_edges(df['sound_ttl'], threshold=500)
        sound_offset_indices = detect_falling_edges(df['sound_ttl'], threshold=500)
        
        # Convert indices to timestamps (in seconds)
        sound_onset_times = df['time_sec'].iloc[sound_onset_indices].values
        sound_offset_times = df['time_sec'].iloc[sound_offset_indices].values
        
        # Detect reward deliveries
        reward_indices = detect_rising_edges(df['reward_signal'], threshold=500)
        reward_times = df['time_sec'].iloc[reward_indices].values
        
        # Detect lick events (using falling edges)
        lick_indices = detect_falling_edges(df['lick_signal'], threshold=500)
        lick_times = df['time_sec'].iloc[lick_indices].values
        
        # Detect lick sensor return to high
        lick_return_indices = detect_rising_edges(df['lick_signal'], threshold=500)
        lick_return_times = df['time_sec'].iloc[lick_return_indices].values
        
        # Assign rewards to the closest sound onset
        reward_assignments = {}  # Maps trial index to list of reward times
        for reward_time in reward_times:
            # Find the closest sound onset
            distances = [abs(reward_time - onset) for onset in sound_onset_times]
            closest_trial_idx = np.argmin(distances)
            
            # Add this reward to the trial's list
            if closest_trial_idx not in reward_assignments:
                reward_assignments[closest_trial_idx] = []
            reward_assignments[closest_trial_idx].append(reward_time)
        
        # Process each trial
        trial_results = []
        
        for i in range(min(len(sound_onset_times), len(sound_offset_times))):
            sound_start = sound_onset_times[i]
            sound_end = sound_offset_times[i]
            
            # Calculate trial duration (time to next trial start or end of recording)
            if i < len(sound_onset_times) - 1:
                next_trial_start = sound_onset_times[i + 1]
                trial_duration = next_trial_start - sound_start
            else:
                # For the last trial, use the end of recording
                trial_duration = df['time_sec'].iloc[-1] - sound_start
            
            # Get rewards assigned to this trial
            trial_rewards = reward_assignments.get(i, [])
            reward_time = min(trial_rewards) if trial_rewards else None
            reward_latency = reward_time - sound_start if reward_time else None
            
            # Find all licks during this trial
            if i < len(sound_onset_times) - 1:
                trial_end_time = sound_onset_times[i + 1]
            else:
                trial_end_time = df['time_sec'].iloc[-1]
            
            # Get lick down times (sensor breaks)
            trial_lick_downs = [t for t in lick_times if sound_start <= t < trial_end_time]
            
            # Get lick up times (sensor returns)
            trial_lick_ups = [t for t in lick_return_times if sound_start <= t < trial_end_time]
            
            # Find first lick after sound onset
            first_lick_latency = trial_lick_downs[0] - sound_start if trial_lick_downs else None
            
            # CORRECTION: Check if this is a rewarded trial with suspicious first lick time
            if reward_time is not None and first_lick_latency is not None:
                # If first lick is significantly after reward (200ms threshold), it's suspicious
                if first_lick_latency > (reward_latency + 0.2):  # 200ms threshold
                    print(f"  Trial {i+1}: Suspicious first lick time detected ({first_lick_latency:.3f}s after cue, {first_lick_latency - reward_latency:.3f}s after reward)")
                    
                    # Check if lick sensor was down during cue presentation
                    # Get the lick signal during cue presentation
                    cue_start_idx = np.argmin(np.abs(df['time_sec'] - sound_start))
                    cue_end_idx = np.argmin(np.abs(df['time_sec'] - sound_end))
                    lick_during_cue = df['lick_signal'].iloc[cue_start_idx:cue_end_idx].values
                    
                    # If lick sensor was down (below threshold) during the entire cue presentation
                    if np.all(lick_during_cue < 500):
                        print(f"  Trial {i+1}: Lick sensor was cut during entire cue presentation, setting first lick latency to 0")
                        first_lick_latency = 0.0
                    else:
                        # Find the first time lick sensor went down during cue presentation
                        lick_binary = lick_during_cue < 500
                        if np.any(lick_binary):
                            first_down_idx = np.argmax(lick_binary)
                            corrected_latency = (first_down_idx / len(lick_during_cue)) * (sound_end - sound_start)
                            print(f"  Trial {i+1}: Correcting first lick latency from {first_lick_latency:.3f}s to {corrected_latency:.3f}s")
                            first_lick_latency = corrected_latency
            
            # Calculate lick times relative to sound onset
            lick_downs_relative = [t - sound_start for t in trial_lick_downs]
            lick_ups_relative = [t - sound_start for t in trial_lick_ups]
            
            # Store trial information
            trial_results.append({
                'trial_num': i + 1,
                'sound_start': sound_start,
                'sound_end': sound_end,
                'trial_duration': trial_duration,
                'reward_latency': reward_latency,
                'first_lick_latency': first_lick_latency,
                'lick_downs_relative': lick_downs_relative,
                'lick_ups_relative': lick_ups_relative,
                'rewarded': reward_time is not None
            })
        
        # Calculate session statistics
        total_trials = len(trial_results)
        rewarded_trials = sum(1 for t in trial_results if t['rewarded'])
        performance = (rewarded_trials / total_trials * 100) if total_trials > 0 else 0
        
        # Extract first lick times for rewarded trials
        rewarded_trial_results = [t for t in trial_results if t['rewarded']]
        first_lick_times = [t['first_lick_latency'] for t in rewarded_trial_results if t['first_lick_latency'] is not None]
        
        # Create session data dictionary
        session_data = {
            'animal_id': animal_id,
            'session_day': session_day,
            'folder_name': os.path.basename(folder_path),
            'folder_path': folder_path,
            'trial_results': trial_results,
            'total_trials': total_trials,
            'rewarded_trials': rewarded_trials,
            'performance': performance,
            'first_lick_times': first_lick_times
        }
        
        # Calculate averages if data exists
        if first_lick_times:
            session_data['avg_first_lick_time'] = np.mean(first_lick_times)
            session_data['std_first_lick_time'] = np.std(first_lick_times)
            # Calculate standard error of the mean
            session_data['sem_first_lick_time'] = np.std(first_lick_times) / np.sqrt(len(first_lick_times))
        else:
            session_data['avg_first_lick_time'] = None
            session_data['std_first_lick_time'] = None
            session_data['sem_first_lick_time'] = None
        
        # Analyze lick sensor cut time
        session_data['lick_sensor_cut_analysis'] = analyze_lick_sensor_cut_time(session_data, df)
        
        return session_data
    
    except Exception as e:
        print(f"Error processing {monitoring_file_to_load}: {str(e)}")
        traceback.print_exc()
        return None 

def calculate_cut_duration_in_window(lick_downs_relative, lick_ups_relative, window_start_relative, window_duration):
    """
    Calculates the total duration (in seconds) the lick sensor was 'down'
    within a specified relative time window.

    Args:
        lick_downs_relative (list): Sorted list of lick sensor down times relative to trial start.
        lick_ups_relative (list): Sorted list of lick sensor up times relative to trial start.
        window_start_relative (float): Start time of the analysis window relative to trial start.
        window_duration (float): Duration of the analysis window.

    Returns:
        float: Total duration (seconds) the sensor was down within the window,
               or 0.0 if calculation fails or window invalid.
    """
    # Return 0 for invalid windows, as duration is additive
    if window_duration <= 1e-9:
        return 0.0

    window_end_relative = window_start_relative + window_duration

    # Create timeline of relevant events (relative to trial start)
    events = []
    events.append({'time': window_start_relative, 'type': 'window_start'})
    events.append({'time': window_end_relative, 'type': 'window_end'})
    event_buffer = 0.1 # Include events slightly outside window bounds for state determination
    for t in lick_downs_relative:
        if window_start_relative - event_buffer < t < window_end_relative + event_buffer:
             events.append({'time': t, 'type': 'lick_down'})
    for t in lick_ups_relative:
        if window_start_relative - event_buffer < t < window_end_relative + event_buffer:
             events.append({'time': t, 'type': 'lick_up'})

    # Ensure events are unique in time and sorted (handle potential coincident events)
    if not events: return 0.0
    try:
        # Use pandas only if available and needed for complex duplicates
        event_df = pd.DataFrame(events).drop_duplicates(subset=['time']).sort_values('time')
        events = event_df.to_dict('records')
    except ImportError: # Fallback if pandas isn't available in this specific context (less likely)
        events.sort(key=lambda x: x['time'])
        unique_events = []
        last_time = -np.inf
        for e in events:
            if e['time'] > last_time:
                unique_events.append(e)
                last_time = e['time']
        events = unique_events


    # Calculate cut duration within the window
    total_cut_duration_in_window = 0
    is_down = False # Current state: is the sensor down?

    # Determine initial state at window_start_relative
    last_down_before = -np.inf
    last_up_before = -np.inf
    if lick_downs_relative and lick_downs_relative[0] < window_start_relative:
       last_down_before = max([t for t in lick_downs_relative if t < window_start_relative], default=-np.inf)
    if lick_ups_relative and lick_ups_relative[0] < window_start_relative:
       last_up_before = max([t for t in lick_ups_relative if t < window_start_relative], default=-np.inf)

    if last_down_before > last_up_before:
        is_down = True # Started the window in a 'down' state

    # Iterate through sorted events
    for i in range(len(events) - 1):
        t1_event = events[i]
        t2_event = events[i+1]

        # Clip event times to the actual window boundaries
        t1 = max(t1_event['time'], window_start_relative)
        t2 = min(t2_event['time'], window_end_relative)

        duration = t2 - t1
        if duration <= 1e-9: continue # Ignore zero or negative duration intervals

        if is_down:
            total_cut_duration_in_window += duration

        # Update state based on the type of the *next* event
        if t2_event['type'] == 'lick_up': is_down = False
        elif t2_event['type'] == 'lick_down': is_down = True

    # Clamp duration just in case of float issues
    total_cut_duration_in_window = max(0.0, min(total_cut_duration_in_window, window_duration))

    return total_cut_duration_in_window 