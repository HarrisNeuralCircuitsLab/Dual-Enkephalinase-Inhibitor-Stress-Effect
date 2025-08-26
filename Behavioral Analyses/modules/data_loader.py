"""
Data loading and manipulation module for animal behavior analysis.

This module contains functions for extracting data from session folders,
including finding session folders, extracting session days, and more.
"""
import os
import re
import glob
import numpy as np
from scipy.io import loadmat

def extract_animal_id_from_folder(folder_path):
    """
    Extract animal ID from folder name
    Example: If folder is "vgat_e1_f7_Day1", extract "vgat_e1_f7"
    """
    folder_name = os.path.basename(folder_path)
    
    # Try to extract ID using the Day pattern
    match = re.search(r'(.+?)_Day\d+', folder_name)
    if match:
        return match.group(1)
    
    # If no Day pattern, try other common patterns
    match = re.search(r'(.*?)_\d{8}', folder_name)  # Pattern with date
    if match:
        return match.group(1)
    
    # Default: return first part of folder name
    parts = folder_name.split('_')
    if len(parts) > 1:
        return '_'.join(parts[:-1])
    
    return None

def extract_session_day(folder_path):
    """
    Extract session day from folder name
    Example: If folder is "e1_f9_Day1", extract 1
    """
    folder_name = os.path.basename(folder_path)
    match = re.search(r'Day(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return None

def extract_date(folder_path):
    """
    Extract date from folder name if available
    Example: If folder contains a date like "20250304", extract it
    """
    folder_name = os.path.basename(folder_path)
    match = re.search(r'(\d{8})', folder_name)
    if match:
        date_str = match.group(1)
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return None

def find_animal_sessions(animal_id, base_dir):
    """
    Find all session folders for a specific animal in the base directory.
    A session folder is identified by containing 'monitoring_data.mat',
    'monitoring_data_FP.mat', or 'test_data.mat'.
    
    Args:
        animal_id (str): Animal ID to search for.
        base_dir (str): Base directory to search in.
        
    Returns:
        list: List of session folder paths.
    """
    session_folders = []
    
    # Check if the base directory exists
    if not os.path.exists(base_dir):
        print(f"Base directory does not exist: {base_dir}")
        return session_folders

    print(f"Searching for session folders for '{animal_id}' in: {base_dir}")
    
    # Search only in the main branch of the base directory
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        
        # Skip any non-directory items or files
        if not os.path.isdir(item_path):
            continue
        
        # Check if the folder name contains the animal ID
        if animal_id.lower() in item.lower():
            # A folder is a session if it contains any of the possible data files.
            has_monitoring_data = os.path.exists(os.path.join(item_path, 'monitoring_data.mat'))
            has_monitoring_data_fp = os.path.exists(os.path.join(item_path, 'monitoring_data_FP.mat'))
            has_test_data = os.path.exists(os.path.join(item_path, 'test_data.mat'))
            
            if has_monitoring_data or has_monitoring_data_fp or has_test_data:
                session_folders.append(item_path)
    
    # Sort folders by session day if available, placing folders without a day at the end.
    session_folders.sort(key=lambda x: (extract_session_day(x) or 999))
    
    return session_folders

def select_session_folder(base_dir):
    """
    Open a directory dialog to select a folder containing session files
    Returns the selected folder path and extracted animal ID
    """
    import tkinter as tk
    from tkinter import filedialog

    # Create a root window and make it visible briefly to ensure dialog appears
    root = tk.Tk()
    root.withdraw()
    
    # Show directory selection dialog
    folder_path = filedialog.askdirectory(
        initialdir=base_dir,
        title="Select Session Folder"
    )
    
    if not folder_path:
        print("No folder selected.")
        return None, None

    # Extract animal ID from folder name
    animal_id = extract_animal_id_from_folder(folder_path)

    if not animal_id:
        print("Could not extract animal ID from folder name.")
        print("Folder name should be in format 'animalID_Day#'")
        return folder_path, None

    print(f"Selected folder: {folder_path}")
    print(f"Extracted animal ID: {animal_id}")

    return folder_path, animal_id

def load_test_data(folder_path):
    """
    Load test phase specific data from the session folder.
    It now checks for 'test_data_FP.mat' as a fallback.

    Args:
        folder_path (str): Path to the session folder containing test_data.mat.

    Returns:
        dict or None: A dictionary containing 'trials' and 'session_info'
                      if successful, otherwise None.
    """
    # Check for the new file name first, then fall back to the old one.
    test_data_file_fp = os.path.join(folder_path, "test_data_FP.mat")
    test_data_file_orig = os.path.join(folder_path, "test_data.mat")
    
    test_data_file_to_load = None
    if os.path.exists(test_data_file_fp):
        test_data_file_to_load = test_data_file_fp
    elif os.path.exists(test_data_file_orig):
        test_data_file_to_load = test_data_file_orig

    if test_data_file_to_load is None:
        print(f"Error: No test data file ('test_data.mat' or 'test_data_FP.mat') found in {folder_path}")
        return None

    try:
        # Load the .mat file
        print(f"Loading {test_data_file_to_load}...")
        test_data = loadmat(test_data_file_to_load)

        # Extract trial data and session info
        if 'trialData' not in test_data or 'sessionInfo' not in test_data:
             print("  ERROR: 'trialData' or 'sessionInfo' not found in MAT file!")
             return None

        trial_data = test_data['trialData'][0,0]  # Access the nested structure
        session_info = test_data['sessionInfo'][0,0]  # Access the nested structure

        # --- Helper function to safely extract numeric/boolean values ---
        # (Slightly simplified for clarity within the module)
        def safe_extract(data_field, field_name, default=None):
            try:
                # Check if the field exists in the dtype names
                if field_name in data_field.dtype.names:
                    field_value = data_field[field_name]
                    # Handle different MATLAB array structures
                    if field_value.size == 1:
                        # Extract scalar value, converting nested arrays if needed
                        val = field_value.item()
                        # Further extract if it's still an array (e.g., string array)
                        if isinstance(val, np.ndarray) and val.size == 1:
                           val = val.item()
                        # Convert boolean-like numerics to Python bool
                        if isinstance(val, (int, float)) and field_name in ['responded', 'injectionStatus', 'stressStatus']:
                            return bool(val)
                        return val
                    elif field_value.size > 0:
                         # For non-scalar arrays (like potentially lick times if added later)
                         # Return the array itself or handle specific cases
                         # For now, assume we only need scalars from trials here
                         # If it's a string array, maybe take the first element?
                         if field_value.dtype.kind in ('U', 'S'):
                             return field_value[0] if field_value.size > 0 else default
                         # print(f"Warning: Field {field_name} is array-like but not scalar. Returning default.") # Optional debug
                         return default # Or return the array if needed: field_value.flatten()
                    else: # Empty array
                        return default
                else: # Field name not found
                    # print(f"Debug: Field '{field_name}' not found in trial data.") # Optional debug
                    return default
            except Exception as e:
                # print(f"Debug: Error extracting '{field_name}': {e}. Returning default.") # Optional debug
                return default
        # --- End of safe_extract helper ---

        # Convert trial data
        trials = []
        # Check if 'trials' field exists and is iterable
        if 'trials' in trial_data.dtype.names and trial_data['trials'].size > 0:
            num_trials = len(trial_data['trials'][0])
            for i in range(num_trials):
                trial = trial_data['trials'][0][i]
                if not trial.dtype: # Skip if trial struct is empty/invalid
                    print(f"Warning: Skipping empty or invalid trial structure at index {i}")
                    continue

                trials.append({
                    'trial_num': safe_extract(trial, 'trialNum', default=i+1), # Use index as fallback
                    'cue_time': safe_extract(trial, 'cueTime'),
                    'reward_time': safe_extract(trial, 'rewardTime'),
                    'iti_duration': safe_extract(trial, 'ITIDuration'),
                    'responded': safe_extract(trial, 'responded', default=False),
                    'reward_size': safe_extract(trial, 'rewardSize')
                })
        else:
            print("Warning: 'trials' field missing or empty in trial_data.")


        # --- Helper function for session info ---
        def safe_extract_session_info(field_name, default_value=None, convert_func=None):
            try:
                if field_name in session_info.dtype.names:
                    field_data = session_info[field_name]

                    if not hasattr(field_data, 'size') or field_data.size == 0:
                        return default_value

                    value = None
                    # Try common MATLAB struct access patterns
                    try:
                        value = field_data[0,0]
                    except (IndexError, TypeError):
                        try:
                            value = field_data[0]
                        except (IndexError, TypeError):
                             value = field_data # Already scalar or string?

                    # Handle nested arrays (especially for strings)
                    while isinstance(value, np.ndarray) and value.size == 1:
                         value = value.item() # Extract scalar value

                    # Specific handling for string arrays
                    if isinstance(value, np.ndarray) and value.dtype.kind in ('U', 'S'):
                        value = value[0] if value.size > 0 else default_value

                    # Check for empty value after extraction attempts
                    if value is None or (hasattr(value, '__len__') and len(value) == 0):
                         return default_value

                    # Apply conversion function if provided
                    return convert_func(value) if convert_func else value

                return default_value
            except Exception as e:
                # print(f"  Debug: Error extracting '{field_name}': {e}. Using default.") # Optional Debug
                return default_value
        # --- End of safe_extract_session_info helper ---

        session_info_dict = {
            'animal_id': str(safe_extract_session_info('animalID', 'unknown', str)),
            'test_day': int(safe_extract_session_info('testDay', 1, int)),
            'timestamp': str(safe_extract_session_info('timestamp', '', str)),
            'date': str(safe_extract_session_info('date', '', str)),
            'session_type': str(safe_extract_session_info('sessionType', 'test', str)),
            'injection_status': bool(safe_extract_session_info('injectionStatus', False, bool)),
            'injection_type': str(safe_extract_session_info('injectionType', '', str)),
            'injection_time': str(safe_extract_session_info('injectionTime', '', str)),
            'stress_status': bool(safe_extract_session_info('stressStatus', False, bool))
        }

        # Add stress_day only if stress_status is True and field exists
        if session_info_dict['stress_status'] and 'stressDay' in session_info.dtype.names:
            session_info_dict['stress_day'] = safe_extract_session_info('stressDay', 0, int)

        # Final validation
        if not trials:
             print("Warning: No trials were successfully extracted.")
        if session_info_dict['animal_id'] == 'unknown':
             print("Warning: Animal ID extracted as 'unknown'. Check .mat file structure.")


        return {
            'trials': trials,
            'session_info': session_info_dict
        }

    except FileNotFoundError:
         print(f"Error: File not found at {test_data_file_to_load}")
         return None
    except KeyError as e:
         print(f"Error: Missing expected key in MAT file structure: {e}")
         return None
    except Exception as e:
        print(f"An unexpected error occurred loading test data from {folder_path}: {e}")
        import traceback
        traceback.print_exc()
        return None 