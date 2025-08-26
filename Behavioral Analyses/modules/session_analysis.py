"""
Module for analyzing individual behavioral sessions, particularly test sessions.
"""

import os
import numpy as np
from scipy.stats import sem
import traceback

# Import necessary functions from other modules
try:
    # Use relative imports within the package
    from .data_loader import load_test_data
    from .signal_processing import process_monitoring_data, calculate_cut_duration_in_window
except ImportError:
    # Fallback for running script directly or other import issues
    print("Warning: Could not perform relative imports. Trying absolute imports for session_analysis.")
    try:
        from modules.data_loader import load_test_data
        from modules.signal_processing import process_monitoring_data, calculate_cut_duration_in_window
    except ImportError as e:
        print(f"Fatal Error in session_analysis: Could not import required functions: {e}")
        raise

def analyze_test_session(folder_path):
    """
    Analyze a single test session, incorporating reward size information.
    """
    # Load test phase specific data
    test_data = load_test_data(folder_path)
    if test_data is None:
        return None # Error printed in load_test_data

    # Process monitoring data, but treat it as optional for test sessions.
    monitoring_data = process_monitoring_data(folder_path)
    if monitoring_data is None:
        # Don't fail. Create a placeholder and print a warning.
        # This allows analysis to continue with only test_data.mat if monitoring is absent.
        print(f"  Warning: Monitoring data not found or failed to process for {os.path.basename(folder_path)}. "
              f"Analysis will be based only on test_data.mat, some metrics will be missing.")
        monitoring_data = {} # Use an empty dict as a placeholder

    # --- Initialize Session Data Dictionary ---
    try:
        session_data = {
            'session_folder': folder_path,
            'session_day': test_data.get('session_info', {}).get('test_day', 'N/A'),
            'animal_id': test_data.get('session_info', {}).get('animal_id', 'N/A'),
            'trial_results': [],
            'reward_sizes': set(),
            'performance_by_reward_size': {},
            'first_lick_times_by_reward_size': {},
            'session_info': test_data.get('session_info', {}),
            'first_lick_times': monitoring_data.get('first_lick_times', []),
            'avg_first_lick_time': monitoring_data.get('avg_first_lick_time'),
            'std_first_lick_time': monitoring_data.get('std_first_lick_time'),
            'sem_first_lick_time': monitoring_data.get('sem_first_lick_time'),
            'lick_sensor_cut_analysis': monitoring_data.get('lick_sensor_cut_analysis', None)
        }
    except Exception as init_e:
         # Keep minimal error reporting in the module
         print(f"  Error initializing session_data for {os.path.basename(folder_path)}: {init_e}")
         return None

    # --- Process Each Trial ---
    processed_trial_count = 0
    skipped_monitoring_trials = 0
    try:
        test_trials = test_data.get('trials', [])
        if not isinstance(test_trials, (list, tuple)):
             print(f"  ERROR: test_data['trials'] is not a list/tuple in {os.path.basename(folder_path)}")
             return None

        for trial_idx, trial in enumerate(test_trials):
            if not isinstance(trial, dict): continue
            trial_num = trial.get('trial_num')
            if trial_num is None: continue

            monitoring_trial = next(
                (t for t in monitoring_data.get('trial_results', []) if isinstance(t, dict) and t.get('trial_num') == trial_num),
                None
            )

            if monitoring_trial:
                sound_start = trial.get('cue_time')
                if sound_start is None: continue

                first_lick_latency = monitoring_trial.get('first_lick_latency')
                responded = trial.get('responded', False)
                reward_size = trial.get('reward_size')

                trial_data = {
                    'trial_num': trial_num,
                    'sound_start': sound_start,
                    'sound_end': sound_start + monitoring_trial.get('sound_duration', 1.0),
                    'rewarded': responded,
                    'reward_size': reward_size,
                    'first_lick_time': (sound_start + first_lick_latency) if first_lick_latency is not None else None,
                    'lick_times': monitoring_trial.get('lick_times', []),
                    'first_lick_latency': first_lick_latency,
                    'lick_downs_relative': monitoring_trial.get('lick_downs_relative', []),
                    'lick_ups_relative': monitoring_trial.get('lick_ups_relative', []),
                    'reward_latency': monitoring_trial.get('reward_latency')
                }

                sound_duration = trial_data['sound_end'] - trial_data['sound_start']
                window_start_relative = 6.5
                window_end_relative = sound_duration + 17.0
                window_duration = window_end_relative - window_start_relative
                iti_cut_duration = 0.0
                if trial_data['lick_downs_relative'] and trial_data['lick_ups_relative']:
                    try:
                        iti_cut_duration = calculate_cut_duration_in_window(
                            trial_data['lick_downs_relative'],
                            trial_data['lick_ups_relative'],
                            window_start_relative,
                            window_duration
                        )
                    except Exception as iti_e:
                         print(f"    Error calculating ITI cut for trial {trial_num} in {os.path.basename(folder_path)}: {iti_e}")
                trial_data['ITI_lick_cut_duration'] = iti_cut_duration

                session_data['trial_results'].append(trial_data)
                processed_trial_count += 1

                if reward_size is not None:
                    session_data['reward_sizes'].add(reward_size)
                    if reward_size not in session_data['performance_by_reward_size']:
                        session_data['performance_by_reward_size'][reward_size] = {'total': 0, 'rewarded': 0}
                    session_data['performance_by_reward_size'][reward_size]['total'] += 1
                    if responded:
                        session_data['performance_by_reward_size'][reward_size]['rewarded'] += 1
                    if responded and first_lick_latency is not None:
                        if reward_size not in session_data['first_lick_times_by_reward_size']:
                            session_data['first_lick_times_by_reward_size'][reward_size] = []
                        session_data['first_lick_times_by_reward_size'][reward_size].append(first_lick_latency)
            else:
                skipped_monitoring_trials += 1

    except Exception as trial_loop_e:
         print(f"  ERROR during trial processing loop for {os.path.basename(folder_path)}: {trial_loop_e}")
         traceback.print_exc()
         return None

    # --- Calculate Overall Performance ---
    total_trials = len(session_data['trial_results'])
    if total_trials == 0:
         print(f"  Warning: No trials processed for {os.path.basename(folder_path)}.")
         session_data['total_trials'] = 0
         session_data['rewarded_trials'] = 0
         session_data['performance'] = 0
    else:
        rewarded_trials = sum(1 for t in session_data['trial_results'] if t.get('rewarded'))
        session_data['total_trials'] = total_trials
        session_data['rewarded_trials'] = rewarded_trials
        session_data['performance'] = (rewarded_trials / total_trials * 100)

        for reward_size, metrics in session_data['performance_by_reward_size'].items():
            metrics['performance'] = (metrics['rewarded'] / metrics['total'] * 100) if metrics['total'] > 0 else 0

        for reward_size, times in session_data['first_lick_times_by_reward_size'].items():
             valid_times = [t for t in times if t is not None and not np.isnan(t)] if times else []
             if valid_times:
                  session_data['first_lick_times_by_reward_size'][reward_size] = {
                      'mean': np.mean(valid_times), 'std': np.std(valid_times), 'n': len(valid_times)
                  }
             else:
                  session_data['first_lick_times_by_reward_size'][reward_size] = {
                       'mean': np.nan, 'std': np.nan, 'n': 0
                  }

    # <<< START: Post-Reward Lick Analysis >>>
    windows_to_analyze = [
        {'label': '0-1s',   'start': 0.0, 'end': 1.0},
        {'label': '1-2s',   'start': 1.0, 'end': 2.0},
        {'label': '5-15s',  'start': 5.0, 'end': 15.0}
    ]
    post_reward_cut_durations = {w['label']: {} for w in windows_to_analyze}
    try: # Wrap the whole post-reward section
        test_trials = test_data.get('trials', []) # Get trials again
        trial_info_map = {t.get('trial_num'): {'reward_size': t.get('reward_size'), 'cue_time': t.get('cue_time')}
                          for t in test_trials if isinstance(t, dict) and t.get('trial_num') is not None}

        monitoring_trials_for_post_reward = monitoring_data.get('trial_results', [])
        if not monitoring_trials_for_post_reward:
             print(f"    Warning: No monitoring trials for post-reward analysis in {os.path.basename(folder_path)}.")
        else:
            for trial_monitoring_info in monitoring_trials_for_post_reward:
                if not isinstance(trial_monitoring_info, dict): continue
                trial_num = trial_monitoring_info.get('trial_num')
                reward_latency = trial_monitoring_info.get('reward_latency')
                is_rewarded_monitoring = trial_monitoring_info.get('rewarded', False)

                if not is_rewarded_monitoring or trial_num is None or reward_latency is None: continue
                if trial_num not in trial_info_map: continue

                reward_size = trial_info_map[trial_num].get('reward_size')
                absolute_cue_time = trial_info_map[trial_num].get('cue_time')
                if reward_size is None or absolute_cue_time is None: continue

                trial_lick_downs_relative = trial_monitoring_info.get('lick_downs_relative', [])
                trial_lick_ups_relative = trial_monitoring_info.get('lick_ups_relative', [])
                if not trial_lick_downs_relative or not trial_lick_ups_relative: continue

                for window in windows_to_analyze:
                    window_label = window['label']
                    window_start_offset = window['start']
                    window_end_offset = window['end']
                    window_start_relative_to_trial = reward_latency + window_start_offset
                    window_duration = window_end_offset - window_start_offset
                    cut_duration = np.nan # Default to NaN
                    try:
                        cut_duration = calculate_cut_duration_in_window(
                            trial_lick_downs_relative, trial_lick_ups_relative,
                            window_start_relative_to_trial, window_duration
                        )
                    except Exception as post_reward_e:
                         print(f"    Error in post-reward cut duration calc for trial {trial_num}, window {window_label}: {post_reward_e}")

                    if reward_size not in post_reward_cut_durations[window_label]:
                        post_reward_cut_durations[window_label][reward_size] = []
                    post_reward_cut_durations[window_label][reward_size].append(cut_duration)

        # Aggregate results
        aggregated_results = { w['label']: {} for w in windows_to_analyze }
        for window_label, reward_data in post_reward_cut_durations.items():
            window_means, window_sems, window_ns = {}, {}, {}
            for reward_size, durations in reward_data.items():
                 try: reward_size_key = float(reward_size)
                 except (ValueError, TypeError): continue # Skip invalid keys

                 valid_durations = [d for d in durations if d is not None and not np.isnan(d)]
                 if valid_durations:
                     window_means[reward_size_key] = np.mean(valid_durations)
                     window_sems[reward_size_key] = sem(valid_durations) if len(valid_durations) > 1 else 0
                     window_ns[reward_size_key] = len(valid_durations)
                 else:
                     window_means[reward_size_key], window_sems[reward_size_key], window_ns[reward_size_key] = np.nan, np.nan, 0

            aggregated_results[window_label] = {'means': window_means, 'sems': window_sems, 'n': window_ns}
        session_data['post_reward_lick_analysis'] = aggregated_results

    except Exception as post_reward_outer_e:
         print(f"  ERROR during post-reward analysis block for {os.path.basename(folder_path)}: {post_reward_outer_e}")
         traceback.print_exc()
         session_data['post_reward_lick_analysis'] = {} # Ensure key exists but is empty

    # <<< END: Post-Reward Lick Analysis >>>

    # --- Final Check ---
    # Return data even if parts failed, calling code can check contents
    # if not session_data.get('trial_results') and not session_data.get('post_reward_lick_analysis',{}).get('means'):
    #      print(f"  Warning: Analysis resulted in no trial data and minimal post-reward data for {os.path.basename(folder_path)}. Returning None.")
    #      return None

    return session_data 