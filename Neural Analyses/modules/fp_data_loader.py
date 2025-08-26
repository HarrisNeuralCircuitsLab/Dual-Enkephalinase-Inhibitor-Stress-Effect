# fp_data_loader.py
# This module will contain functions for loading fiber photometry and behavioral data.

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
import warnings
import cv2 # For video processing
import scipy.io as spio

# We will populate this with adapted functions from the old data_loader.py
# and new functions as needed.

# --- Helper Function for Edge Detection (from rewardAssociationVideo.ipynb) ---
def _detect_edges(signal, threshold=500, edge_type='rising'):
    """Detects rising or falling edges in a signal."""
    binary_signal = signal > threshold
    diff_signal = np.diff(binary_signal.astype(int))
    if edge_type == 'rising':
        edges = np.where(diff_signal > 0)[0] + 1
    elif edge_type == 'falling':
        edges = np.where(diff_signal < 0)[0] + 1
    else:
        raise ValueError("edge_type must be 'rising' or 'falling'")
    return edges

def load_session_data_for_fp(session_folder_path: str):
    """
    Loads behavioral data, raw fiber photometry data, video timestamp data, and reward sizes.
    Performs initial synchronization checks.
    """
    session_folder = Path(session_folder_path)
    session_metadata = {
        'session_folder': str(session_folder),
        'errors': [],
        'warnings': []
    }
    behavioral_df = None # Initialize to prevent UnboundLocalError
    reward_sizes = [] # NEW: Initialize reward sizes
    print(f"Processing session: {session_folder.name}")

    # --- File Discovery ---
    mat_file_pattern = os.path.join(session_folder, 'monitoring_data*.mat')
    mat_files = glob.glob(mat_file_pattern)
    fp_csv_files = list(session_folder.glob("FPdata*.csv"))
    video_ts_csv_files = list(session_folder.glob("video_timestamps*.csv"))
    video_files = list(session_folder.glob("video*.avi")) # Or .mp4, etc.

    if not mat_files: raise FileNotFoundError(f"No monitoring_data*.mat file found in {session_folder}")
    if not fp_csv_files: raise FileNotFoundError(f"FPdata*.csv file not found in {session_folder}")
    if not video_ts_csv_files: warnings.warn(f"Video Timestamp CSV (video_timestamps*.csv) not found in {session_folder}"); session_metadata['warnings'].append("Video TS CSV missing")
    if not video_files: warnings.warn(f"Video file (video*.avi) not found in {session_folder}"); session_metadata['warnings'].append("Video AVI missing")

    mat_file_path = mat_files[0] # Use the first one found
    print(f"Loading MAT file: {os.path.basename(mat_file_path)}...")
    
    try:
        mat_data = spio.loadmat(mat_file_path, squeeze_me=True)
        
        # Check for all possible data structures to ensure backward compatibility
        if 'task_data' in mat_data:
            behavioral_data_raw = mat_data['task_data']
            print("Found 'task_data' in MAT file.")
        elif 'test_data_FP' in mat_data:
            behavioral_data_raw = mat_data['test_data_FP']
            print("Found 'test_data_FP' in MAT file.")
        elif 'data_buffer' in mat_data and 'buffer_idx' in mat_data:
            print("Found 'data_buffer' structure in MAT file.")
            data_buffer = mat_data['data_buffer']
            # Safely extract the scalar index value, handling both arrays and scalars
            buffer_idx_val = mat_data['buffer_idx']
            if isinstance(buffer_idx_val, np.ndarray):
                buffer_idx = int(buffer_idx_val.item())
            else:
                buffer_idx = int(buffer_idx_val)
            behavioral_data_raw = data_buffer[:buffer_idx]
        else:
            raise KeyError("Could not find a valid behavioral data key ('task_data', 'test_data_FP', or 'data_buffer') in the loaded MAT file.")

        # Define column names based on observed structures
        # This might need adjustment if column orders differ
        columns = ['timestamp', 'sound_envelope', 'sound_ttl', 'reward_signal', 'lick_signal', 'fp_ttl']
        if behavioral_data_raw.shape[1] < len(columns):
            warnings.warn(f"Behavioral data has {behavioral_data_raw.shape[1]} columns, expected at least {len(columns)}. Truncating expected column names.")
            columns = columns[:behavioral_data_raw.shape[1]]
        elif behavioral_data_raw.shape[1] > len(columns):
            warnings.warn(f"Behavioral data has {behavioral_data_raw.shape[1]} columns, expected {len(columns)}. Using first {len(columns)} names and ignoring extras.")
            behavioral_data_raw = behavioral_data_raw[:, :len(columns)]

        behavioral_df = pd.DataFrame(behavioral_data_raw, columns=columns)
        if 'timestamp' in behavioral_df.columns and not behavioral_df['timestamp'].empty:
            behavioral_df['time_sec'] = (behavioral_df['timestamp'] - behavioral_df['timestamp'].iloc[0]) / 1e6
            print("Behavioral MAT file loaded and 'time_sec' column created.")
        else:
            raise ValueError("Behavioral data missing 'timestamp' column or is empty.")

    except Exception as e:
        print(f"Error loading or processing MAT file: {e}")
        session_metadata['errors'].append(f"MAT loading error: {e}")
        # When an error occurs here, behavioral_df remains None

    # --- Load Test-Specific MAT file for Reward Sizes ---
    # This file often has the same name as the session folder.
    print(f"\nSearching for test-specific MAT file for reward sizes...")
    # Check for both possible file names (matching original logic)
    test_data_file_fp = session_folder / "test_data_FP.mat"
    test_data_file_orig = session_folder / "test_data.mat"
    
    test_data_file_to_load = None
    if test_data_file_fp.exists():
        test_data_file_to_load = test_data_file_fp
    elif test_data_file_orig.exists():
        test_data_file_to_load = test_data_file_orig
    
    if test_data_file_to_load:
        print(f"Found test data MAT file: {test_data_file_to_load.name}")
        try:
            test_mat_data = spio.loadmat(test_data_file_to_load, squeeze_me=True)
            
            # --- DETAILED DIAGNOSTIC ---
            print("  MAT file keys:", list(test_mat_data.keys()))
            # --------------------------

            # Check for the correct structure: trialData and sessionInfo
            if 'trialData' in test_mat_data and 'sessionInfo' in test_mat_data:
                trial_data = test_mat_data['trialData'][0,0]  # Access the nested structure
                
                # --- DETAILED DIAGNOSTIC ---
                if hasattr(trial_data, 'dtype'):
                    print("  'trialData' structure fields:", trial_data.dtype.names)
                # --------------------------

                # Helper function to safely extract values
                def safe_extract(data_field, field_name, default=None):
                    try:
                        if field_name in data_field.dtype.names:
                            field_value = data_field[field_name]
                            if field_value.size == 1:
                                val = field_value.item()
                                if isinstance(val, np.ndarray) and val.size == 1:
                                   val = val.item()
                                return val
                            elif field_value.size > 0:
                                if field_value.dtype.kind in ('U', 'S'):
                                    return field_value[0] if field_value.size > 0 else default
                                return default
                            else:
                                return default
                        else:
                            return default
                    except Exception:
                        return default

                # Extract reward sizes from trials using the CORRECT method
                if 'trials' in trial_data.dtype.names and trial_data['trials'].size > 0:
                    num_trials = len(trial_data['trials'][0])
                    for i in range(num_trials):
                        trial = trial_data['trials'][0][i]
                        if not trial.dtype:
                            continue
                        
                        reward_size = safe_extract(trial, 'rewardSize')
                        if reward_size is not None:
                            reward_sizes.append(reward_size)
                    
                    print(f"  SUCCESS: Extracted {len(reward_sizes)} reward sizes. First 5: {reward_sizes[:5]}")
                    session_metadata['reward_sizes_loaded'] = True
                else:
                    warnings.warn("'trials' field missing or empty in trialData.")
                    session_metadata['warnings'].append("trials field missing")
            else:
                warnings.warn("'trialData' or 'sessionInfo' not found in the test MAT file.")
                session_metadata['warnings'].append("trialData/sessionInfo missing")
        except Exception as e:
            print(f"Error loading or parsing test data MAT file: {e}")
            session_metadata['errors'].append(f"Test MAT loading error: {e}")
    else:
        print("No test-specific MAT file found. Continuing without reward size data.")
        session_metadata['warnings'].append("Test MAT file not found")

    # --- Load FP Timestamps and Signal CSV ---
    print(f"\nLoading FP CSV file: {fp_csv_files[0].name}...")
    fp_raw_df = None
    try:
        fp_csv_data = pd.read_csv(fp_csv_files[0])
        if 'ComputerTimestamp' not in fp_csv_data.columns or 'LedState' not in fp_csv_data.columns or 'G0' not in fp_csv_data.columns:
            raise ValueError("FP CSV missing one or more required columns: 'ComputerTimestamp', 'LedState', 'G0'.")
        
        fp_raw_df = fp_csv_data[['ComputerTimestamp', 'LedState', 'G0']].copy()
        fp_raw_df['ComputerTimestamp_sec'] = fp_raw_df['ComputerTimestamp'] / 1000.0 # Assuming ms to s
        print("FP CSV file loaded successfully.")
        
        if len(fp_raw_df) > 1:
            fp_duration_from_csv = fp_raw_df['ComputerTimestamp_sec'].iloc[-1] - fp_raw_df['ComputerTimestamp_sec'].iloc[0]
            session_metadata['fp_duration_from_csv_sec'] = fp_duration_from_csv
            print(f"Duration from FP CSV timestamps: {fp_duration_from_csv:.4f} s")
        else:
            session_metadata['fp_duration_from_csv_sec'] = np.nan
            print("Not enough FP timestamps in CSV to calculate duration.")

    except Exception as e:
        print(f"Error loading or processing FP CSV file: {e}")
        session_metadata['errors'].append(f"FP CSV loading error: {e}")
        # raise

    # --- Compare FP Recording Durations (Synchronization Check 1) ---
    if behavioral_df is not None and fp_raw_df is not None:
        print("\nComparing FP Recording Durations:")
        fp_ttl_duration_sec = np.nan
        if 'fp_ttl' in behavioral_df.columns and 'time_sec' in behavioral_df.columns:
            fp_ttl_signal = behavioral_df['fp_ttl'].values
            fp_start_edges = _detect_edges(fp_ttl_signal, edge_type='rising')
            fp_end_edges = _detect_edges(fp_ttl_signal, edge_type='falling')

            if fp_start_edges.size > 0 and fp_end_edges.size > 0:
                fp_start_time_behavioral_system = behavioral_df['time_sec'].iloc[fp_start_edges[0]]
                fp_end_time_behavioral_system = behavioral_df['time_sec'].iloc[fp_end_edges[-1]]
                fp_ttl_duration_sec = fp_end_time_behavioral_system - fp_start_time_behavioral_system
                
                session_metadata['fp_start_time_on_behavioral_clock_sec'] = fp_start_time_behavioral_system
                session_metadata['fp_end_time_on_behavioral_clock_sec'] = fp_end_time_behavioral_system
                session_metadata['fp_duration_from_ttl_sec'] = fp_ttl_duration_sec
                
                print(f"  Based on fp_ttl signal (Behavioral System Clock): {fp_ttl_duration_sec:.4f} s "
                      f"(from t={fp_start_time_behavioral_system:.4f} to t={fp_end_time_behavioral_system:.4f} on its clock)")
                
                # Print initial offset
                fp_system_first_timestamp_sec = fp_raw_df['ComputerTimestamp_sec'].iloc[0]
                session_metadata['fp_system_first_timestamp_sec'] = fp_system_first_timestamp_sec
                print(f"  FP system's first ComputerTimestamp: {fp_system_first_timestamp_sec:.4f} s (Computer Clock)")
                # Note: A direct subtraction here is only meaningful if behavioral_df['timestamp'].iloc[0] is also on the same computer clock
                # and fp_start_time_behavioral_system is converted to that absolute clock.
                # For now, we mainly use fp_start_time_behavioral_system as the t=0 for behavioral events.

            else:
                print("  Could not robustly detect start/end edges from fp_ttl signal.")
                session_metadata['fp_duration_from_ttl_sec'] = np.nan
        else:
            print("  'fp_ttl' or 'time_sec' not found in behavioral_df. Cannot calculate duration from fp_ttl.")
            session_metadata['fp_duration_from_ttl_sec'] = np.nan

        if 'fp_duration_from_csv_sec' in session_metadata and not np.isnan(session_metadata['fp_duration_from_csv_sec']):
            print(f"  Based on FP CSV timestamps (FP System's Computer Clock): {session_metadata['fp_duration_from_csv_sec']:.4f} s")
            if not np.isnan(fp_ttl_duration_sec):
                duration_diff = abs(fp_ttl_duration_sec - session_metadata['fp_duration_from_csv_sec'])
                session_metadata['duration_diff_sec'] = duration_diff
                print(f"  Difference between durations: {duration_diff:.4f} s")
                if duration_diff < 0.5: # Example threshold: 500ms
                    print("  Durations are acceptably close.")
                else:
                    warnings.warn(f"Duration difference ({duration_diff:.4f}s) is > 0.5s. Check synchronization.")
                    session_metadata['warnings'].append("FP Duration Diff > 0.5s")
            else:
                 print("  Cannot compare durations as fp_ttl duration was not calculated.")
    else:
        print("\nSkipping FP duration comparison due to missing behavioral_df or fp_raw_df.")


    # --- Load Video Timestamps CSV & Video File Checks ---
    video_ts_df = None
    if video_ts_csv_files and video_files:
        print(f"\nLoading Video Timestamps CSV: {video_ts_csv_files[0].name}...")
        try:
            # video_ts_df = pd.read_csv(video_ts_csv_filepath, header=None, names=['ComputerTimestamp_ms', 'FrameNumber', 'Other']) # Adjust names based on actual CSV
            # Assuming first column is ComputerTimestamp in ms and second is frame number
            # The 'rewardAssociationVideo.ipynb' output suggested 'Item2' was used for timestamps, implying no header
            # And it printed "Number of timestamps in CSV: 4767", so just one column of timestamps
            # Let's refine based on common format: often it's just one column of timestamps.
            # Re-reading simply, assuming one column of timestamps in ms.
            # OLD: video_ts_df = pd.read_csv(video_ts_csv_filepath, header=None, names=['ComputerTimestamp_ms'])
            # NEW: Skip the first row, which seems to contain non-numeric data like "Item2"
            video_ts_df = pd.read_csv(video_ts_csv_files[0], header=None, names=['ComputerTimestamp_ms'], skiprows=1)
            
            # Convert to numeric, coercing errors. This will turn non-numeric values (if any still exist after skiprows) into NaN
            video_ts_df['ComputerTimestamp_ms'] = pd.to_numeric(video_ts_df['ComputerTimestamp_ms'], errors='coerce')
            
            # Drop rows where timestamp conversion failed (became NaN)
            video_ts_df.dropna(subset=['ComputerTimestamp_ms'], inplace=True)
            
            video_ts_df['ComputerTimestamp_sec'] = video_ts_df['ComputerTimestamp_ms'] / 1000.0 # Assuming ms
            num_csv_timestamps = len(video_ts_df)
            session_metadata['video_num_csv_timestamps'] = num_csv_timestamps
            print(f"Video Timestamps CSV loaded. Number of timestamps: {num_csv_timestamps}")
            print(f"First 5 Video timestamps (Computer Clock, seconds):\n{video_ts_df['ComputerTimestamp_sec'].head()}")

            print(f"\nChecking video file: {video_files[0].name}...")
            cap = cv2.VideoCapture(str(video_files[0]))
            if not cap.isOpened():
                warnings.warn(f"Could not open video file: {video_files[0]}")
                session_metadata['warnings'].append("Video file open failed")
            else:
                num_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                session_metadata['video_num_file_frames'] = num_video_frames
                session_metadata['video_file_fps'] = video_fps
                print(f"Number of frames reported by video file: {num_video_frames}")
                print(f"FPS reported by video file: {video_fps:.2f}")
                cap.release()

                if num_csv_timestamps == num_video_frames:
                    print("SUCCESS: Number of video timestamps matches number of video frames.")
                else:
                    warnings.warn(f"MISMATCH: Video CSV timestamps ({num_csv_timestamps}) vs video file frames ({num_video_frames}).")
                    session_metadata['warnings'].append("Video timestamp/frame mismatch")
                
                # Video Inter-Frame Interval (IFI) Analysis
                if num_csv_timestamps > 1:
                    video_ifis_sec = np.diff(video_ts_df['ComputerTimestamp_sec'].values)
                    session_metadata['video_ifi_mean_ms'] = np.mean(video_ifis_sec) * 1000
                    session_metadata['video_ifi_median_ms'] = np.median(video_ifis_sec) * 1000
                    session_metadata['video_ifi_std_ms'] = np.std(video_ifis_sec) * 1000
                    session_metadata['video_ifi_min_ms'] = np.min(video_ifis_sec) * 1000
                    session_metadata['video_ifi_max_ms'] = np.max(video_ifis_sec) * 1000
                    avg_fps_from_ifi = 1.0 / np.mean(video_ifis_sec) if np.mean(video_ifis_sec) > 0 else np.nan
                    session_metadata['video_avg_fps_from_ifi'] = avg_fps_from_ifi

                    print("Video Inter-Frame Interval (IFI) Analysis (from CSV timestamps):")
                    print(f"  Mean IFI: {session_metadata['video_ifi_mean_ms']:.2f} ms | Median: {session_metadata['video_ifi_median_ms']:.2f} ms | StdDev: {session_metadata['video_ifi_std_ms']:.2f} ms")
                    print(f"  Min: {session_metadata['video_ifi_min_ms']:.2f} ms | Max: {session_metadata['video_ifi_max_ms']:.2f} ms")
                    if not np.isnan(avg_fps_from_ifi): print(f"  Average FPS (from mean IFI): {avg_fps_from_ifi:.2f} FPS")
        except Exception as e:
            print(f"Error loading or processing video timestamp CSV: {e}")
            session_metadata['errors'].append(f"Video TS CSV loading error: {e}")
    else:
        print("\nVideo timestamp CSV or video file not found. Skipping video checks.")

    print("\n--- load_session_data_for_fp complete ---")
    return behavioral_df, fp_raw_df, video_ts_df, session_metadata, reward_sizes # MODIFIED RETURN

# Example of how you might add other utility functions from old data_loader:
# def extract_animal_id_from_folder(folder_path): ...
# def find_animal_sessions(base_dir, animal_id): ...

print("fp_data_loader.py loaded with load_session_data_for_fp function.") 