"""
Visualization utilities for behavioral analysis.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pandas as pd
from scipy.stats import sem
import matplotlib.colors as mcolors
import time # Add time import
# Use a non-interactive backend to prevent blocking
matplotlib.use('Agg')

# Set plot style
# --- MODIFICATION START: Temporarily disable style ---
# plt.style.use('seaborn-v0_8-whitegrid')
# --- MODIFICATION END ---
sns.set_context("talk")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.dpi'] = 100

def visualize_lick_sensor_cut_time(animal_sessions, animal_id, output_dir=None, 
                                   training_test_boundary_day=None, 
                                   pl37_vline_day=None, post_pl37_vline_day=None):
    """Create visualizations for lick sensor cut time analysis across sessions."""
    if not animal_sessions:
        print("No session data to visualize")
        return None
    
    # Filter sessions that have lick sensor cut analysis data
    sessions_with_analysis = [s for s in animal_sessions if 'lick_sensor_cut_analysis' in s]
    if not sessions_with_analysis:
        print("No lick sensor cut analysis data available")
        return None
    
    # Sort sessions by day
    sessions_with_analysis.sort(key=lambda x: x['session_day'])
    
    # Extract data for plotting
    days = [session['session_day'] for session in sessions_with_analysis]
    
    # Calculate average ITI_lick_cut_duration for each session (if available)
    iti_lick_durations = []
    iti_lick_sems = []  # Store SEM values
    
    for session in sessions_with_analysis:
        if 'trial_results' in session:
            # Check if trials already have the ITI_lick_cut_duration calculated (for test sessions)
            trial_with_iti_duration = next((t for t in session['trial_results'] if 'ITI_lick_cut_duration' in t), None)
            
            if trial_with_iti_duration:
                # Test sessions already have ITI_lick_cut_duration calculated
                iti_durations = [trial.get('ITI_lick_cut_duration', 0.0) for trial in session['trial_results'] 
                                if 'ITI_lick_cut_duration' in trial]
            else:
                # For training sessions, calculate ITI_lick_cut_duration
                iti_durations = []
                for trial in session['trial_results']:
                    if 'lick_downs_relative' in trial and 'lick_ups_relative' in trial:
                        # Calculate sound duration if available
                        if 'sound_start' in trial and 'sound_end' in trial:
                            sound_duration = trial['sound_end'] - trial['sound_start']
                        else:
                            # Default to 1.0 second if not available
                            sound_duration = 1.0
                        
                        # Use the same window parameters as in test_analysis.py
                        window_start_relative = 6.5  # 6.5 seconds after cue start
                        window_end_relative = sound_duration + 17.0  # cue end + 17 seconds
                        window_duration = window_end_relative - window_start_relative
                        
                        from modules.signal_processing import calculate_cut_duration_in_window
                        
                        iti_cut_duration = calculate_cut_duration_in_window(
                            trial['lick_downs_relative'],
                            trial['lick_ups_relative'],
                            window_start_relative,
                            window_duration
                        )
                        iti_durations.append(iti_cut_duration if iti_cut_duration is not None else 0.0)
            
            # Calculate mean and SEM if there are values
            if iti_durations:
                from scipy.stats import sem
                avg_iti_duration = np.mean(iti_durations)
                sem_iti_duration = sem(iti_durations) if len(iti_durations) > 1 else 0.0
            else:
                avg_iti_duration = 0.0
                sem_iti_duration = 0.0
                
            iti_lick_durations.append(avg_iti_duration)
            iti_lick_sems.append(sem_iti_duration)
        else:
            # If no trial_results available, use 0.0 as a placeholder
            iti_lick_durations.append(0.0)
            iti_lick_sems.append(0.0)
    
    # Extract metrics for pre-cue vs post-cue comparison (keep this part)
    pre_cue_pcts = [session['lick_sensor_cut_analysis']['avg_pct_cut_pre_cue'] for session in sessions_with_analysis]
    post_cue_pcts = [session['lick_sensor_cut_analysis']['avg_pct_cut_post_cue'] for session in sessions_with_analysis]
    
    # Calculate new normalized difference ratio: (post_cue - pre_cue)/(post_cue + pre_cue)
    # This produces values from -1 to 1, with 0 indicating equal values
    normalized_ratios = []
    for post, pre in zip(post_cue_pcts, pre_cue_pcts):
        if post + pre > 0:  # Avoid division by zero
            ratio = (post - pre) / (post + pre)
        else:
            ratio = 0  # Default to 0 if both are 0
        normalized_ratios.append(ratio)
    
    # --- FIX: Use passed arguments for boundary positions ---
    boundary_pos_cut = None
    if training_test_boundary_day is not None and training_test_boundary_day > 0:
        boundary_pos_cut = training_test_boundary_day + 0.5

    pl37_line_pos = pl37_vline_day
    post_pl37_boundary_pos = post_pl37_vline_day - 0.5 if post_pl37_vline_day is not None else None
    # --- End of FIX ---

    # --- MODIFICATION START: Find first stress day ---
    first_stress_day_cut = None
    stress_boundary_pos_cut = None
    for session in sessions_with_analysis:
        if session.get('session_info', {}).get('stress_status', False):
            first_stress_day_cut = session['session_day']
            break # Found the first one
    
    if first_stress_day_cut is not None:
        stress_boundary_pos_cut = first_stress_day_cut - 0.5
    
    # Create a figure with three panels stacked vertically (3x1 grid)
    fig, axs = plt.subplots(3, 1, figsize=(20, 18))
    
    # Plot 1: Average ITI Lick Cut Duration (in seconds) by session (top)
    axs[0].bar(days, iti_lick_durations, color='red', width=0.6, yerr=iti_lick_sems, capsize=5)
    
    axs[0].set_xlabel('Session Day', fontsize=12)
    axs[0].set_ylabel('ITI Lick Cut Duration (seconds)', fontsize=12)
    axs[0].set_title(f'{animal_id}: ITI Lick Cut Duration\n(Window: cue start + 6.5s to cue end + 17s)', fontsize=14)
    axs[0].set_xticks(days)
    axs[0].grid(True, alpha=0.3)
    
    boundary_label_added = False
    stress_label_added = False
    if boundary_pos_cut is not None:
        axs[0].axvline(x=boundary_pos_cut, color='red', linestyle='--', alpha=0.7, label='Training/Test Boundary')
        boundary_label_added = True
    if stress_boundary_pos_cut is not None:
        axs[0].axvline(x=stress_boundary_pos_cut, color='blue', linestyle=':', alpha=0.7, label='Stress Phase Start')
        stress_label_added = True
    
    # --- Add phase lines to the first plot ---
    if pl37_line_pos is not None:
        axs[0].axvline(x=pl37_line_pos, color='green', linestyle='-.', alpha=0.7, label='PL37')
    if post_pl37_boundary_pos is not None:
        axs[0].axvline(x=post_pl37_boundary_pos, color='purple', linestyle=':', alpha=0.7, label='Stress Phase End')
    # ----------------------------------------
    
    # Add data labels
    for i, (duration, sem_val) in enumerate(zip(iti_lick_durations, iti_lick_sems)):
        axs[0].annotate(f"{duration:.2f}s\n±{sem_val:.2f}", 
                    (days[i], duration), 
                    textcoords="offset points",
                    xytext=(0, 5), 
                    ha='center',
                    fontsize=9)
    
    # Add legends for boundary lines if needed
    if boundary_label_added or stress_label_added:
        axs[0].legend(loc='upper left')
    
    # Plot 2: Percentage cut time by session (pre-cue vs post-cue) - middle
    width = 0.35
    x = np.array(days)
    axs[1].bar(x - width/2, pre_cue_pcts, width, label='Pre-Cue Window', color='green')
    axs[1].bar(x + width/2, post_cue_pcts, width, label='Post-Cue Window', color='orange')
    
    axs[1].set_xlabel('Session Day', fontsize=12)
    axs[1].set_ylabel('% Time Lick Sensor Cut', fontsize=12)
    axs[1].set_title(f'{animal_id}: Lick Sensor Cut Time (Pre-Cue vs Post-Cue)', fontsize=14)
    axs[1].set_xticks(days)
    axs[1].legend(loc='upper left') # Adjust legend position
    axs[1].grid(True, alpha=0.3)
    
    if boundary_pos_cut is not None:
        axs[1].axvline(x=boundary_pos_cut, color='red', linestyle='--', alpha=0.7, label='_nolegend_') # No duplicate label
    if stress_boundary_pos_cut is not None:
        axs[1].axvline(x=stress_boundary_pos_cut, color='blue', linestyle=':', alpha=0.7, label='_nolegend_')
    
    # --- Add phase lines to the second plot ---
    if pl37_line_pos is not None:
        axs[1].axvline(x=pl37_line_pos, color='green', linestyle='-.', alpha=0.7, label='_nolegend_')
    if post_pl37_boundary_pos is not None:
        axs[1].axvline(x=post_pl37_boundary_pos, color='purple', linestyle=':', alpha=0.7, label='_nolegend_')
    # -----------------------------------------
    
    # Add data labels
    for i, (pre, post) in enumerate(zip(pre_cue_pcts, post_cue_pcts)):
        axs[1].annotate(f"{pre:.1f}%", 
                    (days[i] - width/2, pre), 
                    textcoords="offset points",
                    xytext=(0, 5), 
                    ha='center',
                    fontsize=9)
        axs[1].annotate(f"{post:.1f}%", 
                    (days[i] + width/2, post), 
                    textcoords="offset points",
                    xytext=(0, 5), 
                    ha='center',
                    fontsize=9)
    
    # Plot 3: Normalized difference ratio (post_cue - pre_cue)/(post_cue + pre_cue) - bottom
    axs[2].plot(days, normalized_ratios, 'o-', color='purple', markersize=8, linewidth=2)
    axs[2].set_xlabel('Session Day', fontsize=12)
    axs[2].set_ylabel('Normalized Difference Ratio', fontsize=12)
    axs[2].set_title(f'{animal_id}: Normalized Difference Ratio\n(post_cue - pre_cue)/(post_cue + pre_cue)', fontsize=14)
    axs[2].set_xticks(days)
    axs[2].grid(True, alpha=0.3)
    
    if boundary_pos_cut is not None:
        axs[2].axvline(x=boundary_pos_cut, color='red', linestyle='--', alpha=0.7, label='_nolegend_') # No duplicate label
    if stress_boundary_pos_cut is not None:
        axs[2].axvline(x=stress_boundary_pos_cut, color='blue', linestyle=':', alpha=0.7, label='_nolegend_')
    
    # --- Add phase lines to the third plot ---
    if pl37_line_pos is not None:
        axs[2].axvline(x=pl37_line_pos, color='green', linestyle='-.', alpha=0.7, label='_nolegend_')
    if post_pl37_boundary_pos is not None:
        axs[2].axvline(x=post_pl37_boundary_pos, color='purple', linestyle=':', alpha=0.7, label='_nolegend_')
    # ---------------------------------------
    
    # Add horizontal line at ratio = 0 (equal pre-cue and post-cue)
    axs[2].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Set y-axis limits to show the full -1 to 1 range
    axs[2].set_ylim(-1.1, 1.1)
    
    # Add data labels
    for i, ratio in enumerate(normalized_ratios):
        axs[2].annotate(f"{ratio:.2f}", 
                    (days[i], ratio), 
                    textcoords="offset points",
                    xytext=(0, 10), 
                    ha='center',
                    fontsize=10)
    
    # Consolidate legend
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    # Remove duplicates
    unique_tuples = list(dict.fromkeys(zip(labels, handles)))
    unique_labels, unique_handles = zip(*unique_tuples) if unique_tuples else ([], [])
    
    # Add a general legend
    fig.legend(unique_handles, unique_labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for legend
    
    # Add overall title
    fig.suptitle(f'{animal_id}: Lick Sensor Analysis Across Sessions', fontsize=16, y=0.98)
    
    # --- FIX: Save figure and return path ---
    if output_dir is None:
        output_dir = os.path.join('Figures', animal_id, 'session_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{animal_id}_lick_sensor_cut_analysis.png"
    filepath = os.path.join(output_dir, filename)
    
    try:
        plt.savefig(filepath, dpi=100)
        print(f"Saved lick sensor cut analysis figure: {filepath}")
    except Exception as e:
        print(f"Error saving lick sensor cut analysis figure: {e}")
        filepath = None
    finally:
        plt.close(fig)
        
    return filepath
    # --- End of FIX ---

def visualize_test_session_analysis(session_data, animal_id, output_dir):
    """Create visualizations for a single test session (latency/perf vs prev reward)."""
    # if not test_sessions:
    #     return
    
    # Get the last test session - NO LONGER NEEDED, use session_data directly
    # last_session = max(test_sessions, key=lambda x: x['session_day'])
    # --- MODIFICATION END ---

    # 1. Create figure with three panels for session analysis (now 3 panels instead of 2)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Extract trial data
    # --- MODIFICATION START: Use session_data --- 
    # trial_results = last_session['trial_results']
    trial_results = session_data['trial_results']
    # --- MODIFICATION END ---
    
    # 1.1 Response latency by previous reward size (new panel)
    # Group latencies by previous reward size, skipping first trial and trials after non-rewarded trials
    latencies_by_prev_reward = {}
    performance_by_prev_reward = {}  # For the second plot
    lick_cut_durations_by_prev_reward = {}  # For the third plot (new)
    
    for i in range(1, len(trial_results)):
        curr_trial = trial_results[i]
        prev_trial = trial_results[i-1]
        
        # Skip if previous trial wasn't rewarded
        if not prev_trial['rewarded']:
            continue
            
        if 'reward_size' in prev_trial and prev_trial['reward_size'] is not None:
            prev_reward_size = prev_trial['reward_size']
            
            # Get current trial's latency
            if 'first_lick_latency' in curr_trial and curr_trial['first_lick_latency'] is not None:
                if prev_reward_size not in latencies_by_prev_reward:
                    latencies_by_prev_reward[prev_reward_size] = []
                latencies_by_prev_reward[prev_reward_size].append(curr_trial['first_lick_latency'])
            
            # Track performance based on previous reward size
            if prev_reward_size not in performance_by_prev_reward:
                performance_by_prev_reward[prev_reward_size] = {
                    'total': 0,
                    'rewarded': 0
                }
            
            performance_by_prev_reward[prev_reward_size]['total'] += 1
            if curr_trial['rewarded']:
                performance_by_prev_reward[prev_reward_size]['rewarded'] += 1
            
            # NEW: Get current trial's lick cut duration (if available in lick_sensor_cut_analysis)
            if 'lick_sensor_cut_analysis' in session_data:
                # Find the corresponding trial metrics in lick_sensor_cut_analysis
                trial_metrics = session_data['lick_sensor_cut_analysis']['trial_metrics']
                current_trial_metric = next((t for t in trial_metrics if t['trial_num'] == curr_trial['trial_num']), None)
                
                if current_trial_metric and 'post_cue_cut_duration' in current_trial_metric:
                    if prev_reward_size not in lick_cut_durations_by_prev_reward:
                        lick_cut_durations_by_prev_reward[prev_reward_size] = []
                    lick_cut_durations_by_prev_reward[prev_reward_size].append(current_trial_metric['post_cue_cut_duration'])
    
    size_labels = ['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
    
    if latencies_by_prev_reward:
        # Sort reward sizes for consistent plotting
        prev_reward_sizes = sorted(latencies_by_prev_reward.keys())
        latency_data = [latencies_by_prev_reward[size] for size in prev_reward_sizes]
        
        # --- Calculate Mean and SEM --- 
        means = [np.mean(l) for l in latency_data]
        sems = [sem(l) if len(l) > 1 else 0 for l in latency_data]
        # ------------------------------
        
        # --- MODIFICATION: Replace Box Plot with Bar Plot --- 
        # Plot bars for mean latency with SEM error bars
        ax1.bar(prev_reward_sizes, means, yerr=sems, capsize=5, 
                color='lightgrey', alpha=0.7, ecolor='black', 
                label='Mean Latency +/- SEM') # Add label for legend if needed
        # --------------------------------------------------

        # --- Keep Individual data points (jittered) --- 
        cmap = plt.cm.viridis # Ensure cmap is defined
        norm = matplotlib.colors.Normalize(vmin=1, vmax=5) # Ensure norm is defined
        all_plotted_latencies = [] # Collect all latencies for robust ylim calculation
        for i, size in enumerate(prev_reward_sizes):
            latencies = latencies_by_prev_reward[size]
            if latencies:
                all_plotted_latencies.extend(latencies) # Add to combined list
                # Add jitter to x-position
                jitter = np.random.normal(0, 0.05, size=len(latencies))
                point_color = cmap(norm(size))
                ax1.scatter(np.array([size] * len(latencies)) + jitter, latencies, 
                        alpha=0.6, s=30, color=point_color, edgecolors='none')
        # --------------------------------------------
        
        ax1.set_xlabel('Previous Trial Reward Size', fontsize=12)
        ax1.set_ylabel('Response Latency (s)', fontsize=12)
        # --- MODIFICATION: Update title --- 
        # --- MODIFICATION START: Use session_data --- 
        # ax1.set_title(f'Mean Response Latency by Previous Reward Size (Test Day {last_session["session_day"]})', fontsize=14)
        ax1.set_title(f'Mean Response Latency by Previous Reward Size (Test Day {session_data["session_day"]})', fontsize=14)
        # --- MODIFICATION END ---
        # ----------------------------------
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(prev_reward_sizes)
        ax1.set_xticklabels([size_labels[size-1] for size in prev_reward_sizes if 1 <= size <= len(size_labels)])
        
        # --- Calculate Y-limits based on IQR of plotted data --- 
        if all_plotted_latencies:
            q1, q3 = np.percentile(all_plotted_latencies, [25, 75])
            iqr = q3 - q1
            upper_whisker = q3 + 1.5 * iqr
            # Set ylim, add some padding, ensure a minimum sensible upper limit (e.g., 1s or 2s)
            final_upper_limit = max(upper_whisker * 1.1, 1.0) # Add 10% padding, minimum 1.0s
            ax1.set_ylim(0, final_upper_limit)
        else:
             ax1.set_ylim(0, 1) # Default limit if no latency data
        # ------------------------------------------------------
        
        # Optional: Add legend if bar label was added
        # ax1.legend()
    else:
        # If no valid data, show message
        # --- MODIFICATION START: Use session_data --- 
        # ax1.text(0.5, 0.5, 'No valid response latency data after rewarded trials', ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.text(0.5, 0.5, 'No valid response latency data after rewarded trials', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        # --- MODIFICATION END ---
        ax1.set_title(f'Response Latency by Previous Reward Size (Test Day {session_data["session_day"]})', fontsize=14)
        ax1.set_xticks([])
        ax1.set_yticks([])
    
    # 1.2 Performance by previous reward size (middle panel)
    if performance_by_prev_reward:
        prev_reward_sizes = sorted(performance_by_prev_reward.keys())
        performances = []
        
        for size in prev_reward_sizes:
            metrics = performance_by_prev_reward[size]
            perf = (metrics['rewarded'] / metrics['total'] * 100) if metrics['total'] > 0 else 0
            performances.append(perf)
            
        # Bar plot of performance by previous reward size
        bars = ax2.bar(prev_reward_sizes, performances, color='lightgreen')
        
        # Add data labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            count_text = f"{performance_by_prev_reward[prev_reward_sizes[i]]['rewarded']}/{performance_by_prev_reward[prev_reward_sizes[i]]['total']}"
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f"{performances[i]:.1f}%\n({count_text})",
                    ha='center', va='bottom', fontsize=10)
        
        ax2.set_xlabel('Previous Trial Reward Size', fontsize=12)
        ax2.set_ylabel('Performance (%)', fontsize=12)
        # --- MODIFICATION START: Use session_data --- 
        # ax2.set_title(f'Performance by Previous Reward Size (Test Day {last_session["session_day"]})', fontsize=14)
        ax2.set_title(f'Performance by Previous Reward Size (Test Day {session_data["session_day"]})', fontsize=14)
        # --- MODIFICATION END ---
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3)
        
        # Add reward size labels for x-axis
        ax2.set_xticks(prev_reward_sizes)
        ax2.set_xticklabels([size_labels[size-1] for size in prev_reward_sizes if 1 <= size <= len(size_labels)])
    else:
        # If no valid data, show message
        # --- MODIFICATION START: Use session_data --- 
        # ax2.text(0.5, 0.5, 'No valid performance data after rewarded trials', ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.text(0.5, 0.5, 'No valid performance data after rewarded trials', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        # --- MODIFICATION END ---
        ax2.set_title(f'Performance by Previous Reward Size (Test Day {session_data["session_day"]})', fontsize=14)
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    # 1.3 NEW PANEL: Lick Cut Duration by previous reward size (bottom panel)
    if lick_cut_durations_by_prev_reward:
        # Sort reward sizes for consistent plotting
        prev_reward_sizes = sorted(lick_cut_durations_by_prev_reward.keys())
        duration_data = [lick_cut_durations_by_prev_reward[size] for size in prev_reward_sizes]
        
        # Calculate Mean and SEM for cut durations
        means = [np.mean(d) for d in duration_data]
        sems = [sem(d) if len(d) > 1 else 0 for d in duration_data]
        
        # Plot bars for mean cut duration with SEM error bars
        ax3.bar(prev_reward_sizes, means, yerr=sems, capsize=5, 
                color='lightblue', alpha=0.7, ecolor='black', 
                label='Mean Cut Duration +/- SEM')
                
        # Add individual data points (jittered)
        cmap = plt.cm.viridis
        norm = matplotlib.colors.Normalize(vmin=1, vmax=5)
        all_plotted_durations = []
        for i, size in enumerate(prev_reward_sizes):
            durations = lick_cut_durations_by_prev_reward[size]
            if durations:
                all_plotted_durations.extend(durations)
                # Add jitter to x-position
                jitter = np.random.normal(0, 0.05, size=len(durations))
                point_color = cmap(norm(size))
                ax3.scatter(np.array([size] * len(durations)) + jitter, durations, 
                        alpha=0.6, s=30, color=point_color, edgecolors='none')
        
        # Add a horizontal line at the maximum possible duration (1.5s)
        ax3.axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='Max Window (1.5s)')
        
        ax3.set_xlabel('Previous Trial Reward Size', fontsize=12)
        ax3.set_ylabel('Lick Sensor Cut Duration (s)', fontsize=12)
        ax3.set_title(f'Lick Sensor Cut Duration by Previous Reward Size (Test Day {session_data["session_day"]})', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(prev_reward_sizes)
        ax3.set_xticklabels([size_labels[size-1] for size in prev_reward_sizes if 1 <= size <= len(size_labels)])
        
        # Add data labels on top of bars
        for i, prev_size in enumerate(prev_reward_sizes):
            count = len(lick_cut_durations_by_prev_reward[prev_size])
            ax3.text(prev_size, means[i] + 0.05,
                    f"{means[i]:.2f}s\n(n={count})",
                    ha='center', va='bottom', fontsize=10)
        
        # Calculate Y-limits based on data range, leaving room for labels
        if all_plotted_durations:
            # Set ylim to accommodate the data with some padding
            max_duration = max(all_plotted_durations)
            # Make sure we show the 1.5s line and leave room for labels
            ax3.set_ylim(0, max(max_duration * 1.1, 1.6))
        else:
            ax3.set_ylim(0, 1.6)  # Default limit if no data
        
        # Add a legend
        ax3.legend(loc='upper right')
    else:
        # If no valid data, show message
        ax3.text(0.5, 0.5, 'No valid lick cut duration data after rewarded trials', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title(f'Lick Sensor Cut Duration by Previous Reward Size (Test Day {session_data["session_day"]})', fontsize=14)
        ax3.set_xticks([])
        ax3.set_yticks([])
    
    # Add overall title
    # --- MODIFICATION START: Use session_data --- 
    # plt.suptitle(f'Test Session Analysis for {animal_id} - Day {last_session["session_day"]}', fontsize=16, y=0.98)
    plt.suptitle(f'Test Session Analysis for {animal_id} - Day {session_data["session_day"]}', fontsize=16, y=0.98)
    # --- MODIFICATION END ---
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save session analysis figure
    # --- MODIFICATION START: Use session_data --- 
    # filename = f"{animal_id}_TEST_Day{last_session['session_day']}_analysis.png"
    filename = f"{animal_id}_TEST_Day{session_data['session_day']}_analysis.png"
    # --- MODIFICATION END ---
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved figure: {filepath}")
    
    plt.close(fig)
    
    # 2. Create lick sensor cut analysis figure if available
    if 'lick_sensor_cut_analysis' in session_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Extract trial data from lick sensor cut analysis
        trial_metrics = session_data['lick_sensor_cut_analysis']['trial_metrics']
        trial_nums = [t['trial_num'] for t in trial_metrics]
        pre_cue_pcts = [t['pct_cut_during_pre_cue'] for t in trial_metrics]
        post_cue_pcts = [t['pct_cut_during_post_cue'] for t in trial_metrics]
        
        # Extract ITI lick cut durations from the trial_results
        trial_results = session_data['trial_results']
        iti_cut_durations = []
        trial_nums_with_iti_data = []
        for trial_num in trial_nums:
            # Find the corresponding trial in trial_results
            matched_trial = next((t for t in trial_results if t['trial_num'] == trial_num), None)
            if matched_trial and 'ITI_lick_cut_duration' in matched_trial:
                iti_cut_durations.append(matched_trial['ITI_lick_cut_duration'])
                trial_nums_with_iti_data.append(trial_num)
        
        # Plot 1: ITI Lick Cut Duration (in seconds) by trial
        ax1.plot(trial_nums_with_iti_data, iti_cut_durations, 'o-', color='red', label='ITI Lick Duration (seconds)')
        ax1.set_xlabel('Trial Number', fontsize=12)
        ax1.set_ylabel('Lick Sensor Cut Duration (seconds)', fontsize=12)
        ax1.set_title(f'{animal_id}: ITI Lick Duration\n(Window: cue start + 6.5s to cue end + 17s)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add mean line for ITI duration
        if iti_cut_durations:
            mean_iti_duration = np.mean(iti_cut_durations)
            ax1.axhline(y=mean_iti_duration, color='black', linestyle='--', 
                      label=f'Mean: {mean_iti_duration:.2f}s', alpha=0.7)
            # Update legend to include the mean line
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles, labels)
        
        # Plot 2: Pre-cue vs Post-cue cut time by trial (unchanged)
        ax2.plot(trial_nums, pre_cue_pcts, 'o-', color='green', label='Pre-Cue Window')
        ax2.plot(trial_nums, post_cue_pcts, 'o-', color='orange', label='Post-Cue Window')
        ax2.set_xlabel('Trial Number', fontsize=12)
        ax2.set_ylabel('% Time Lick Sensor Cut', fontsize=12)
        ax2.set_title(f'{animal_id}: Pre-Cue vs Post-Cue Cut Time (Test Day {session_data["session_day"]})', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add overall title
        plt.suptitle(f'Lick Sensor Cut Analysis for {animal_id} - Test Day {session_data["session_day"]}', fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save lick sensor cut analysis figure
        filename = f"{animal_id}_TEST_Day{session_data['session_day']}_lick_sensor_cut_analysis.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved figure: {filepath}")
        
        plt.close(fig)

def visualize_animal_sessions(animal_sessions, animal_id, output_dir=None):
    """Generate visualizations for all sessions of an animal."""
    if output_dir is None:
        output_dir = os.path.join('Figures', animal_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure the input list is not empty
    if not animal_sessions:
        print("No session data to visualize")
        return

    # Sort ALL sessions by day to ensure chronological order
    animal_sessions.sort(key=lambda x: x['session_day'])

    # Extract data for plotting from ALL sessions
    days = [session['session_day'] for session in animal_sessions]
    performance = [session['performance'] for session in animal_sessions]
    total_trials = [session['total_trials'] for session in animal_sessions]
    rewarded_trials = [session['rewarded_trials'] for session in animal_sessions]

    # Extract first lick times for each session
    first_lick_times_by_session = []
    for session in animal_sessions:
        # Handle potential differences in key names or availability between training/test
        if 'first_lick_times' in session:
            first_lick_times_by_session.append(session['first_lick_times'])
        elif 'trial_results' in session: # Test sessions might have lick times within trial_results
            session_licks = []
            for trial in session['trial_results']:
                if 'first_lick_latency' in trial and trial['first_lick_latency'] is not None:
                    session_licks.append(trial['first_lick_latency'])
            first_lick_times_by_session.append(session_licks)
        else:
            first_lick_times_by_session.append([])

    # Extract average/std/sem first lick times, handling potential absence
    avg_lick_times = []
    std_lick_times = []
    sem_lick_times = []
    for session in animal_sessions:
        avg = session.get('avg_first_lick_time')
        # If avg_first_lick_time is missing (might happen in test sessions), try calculating it
        if avg is None and 'first_lick_times_by_reward_size' in session:
             # Calculate overall average from the 'first_lick_times' collected earlier
            session_licks = next((licks for d, licks in zip(days, first_lick_times_by_session) if d == session['session_day']), [])
            if session_licks:
                avg = np.mean(session_licks) if session_licks else None
                session['avg_first_lick_time'] = avg # Store it back for consistency if needed
                session['std_first_lick_time'] = np.std(session_licks) if session_licks else 0
                session['sem_first_lick_time'] = np.std(session_licks) / np.sqrt(len(session_licks)) if session_licks else 0
            else:
                 session['avg_first_lick_time'] = None
                 session['std_first_lick_time'] = 0
                 session['sem_first_lick_time'] = 0


        avg_lick_times.append(session.get('avg_first_lick_time')) # Use get again in case calculation failed
        std_lick_times.append(session.get('std_first_lick_time', 0))
        sem_lick_times.append(session.get('sem_first_lick_time', 0))

    # Ensure avg_lick_times contains numeric types or None for plotting
    avg_lick_times = [t if t is not None else np.nan for t in avg_lick_times]
    sem_lick_times = [t if t is not None else 0 for t in sem_lick_times] # error bars need numeric value

    # --- Find boundary between training and test sessions ---
    max_training_day = None
    min_test_day = None
    boundary_pos = None
    training_days = [s['session_day'] for s in animal_sessions if not s.get('is_test_session', False)]
    test_days = [s['session_day'] for s in animal_sessions if s.get('is_test_session', False)]
    if training_days:
        max_training_day = max(training_days)
    if test_days:
        min_test_day = min(test_days)
    if max_training_day is not None and min_test_day is not None and min_test_day > max_training_day:
        boundary_pos = max_training_day + 0.5

    # --- MODIFICATION START: Find first stress day ---
    first_stress_day = None
    stress_boundary_pos = None
    for session in animal_sessions:
        # Check stress status within session_info
        if session.get('session_info', {}).get('stress_status', False):
            first_stress_day = session['session_day']
            break # Found the first stress session
            
    if first_stress_day is not None:
        # Place the line just before the first stress day
        stress_boundary_pos = first_stress_day - 0.5

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Performance over days
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(days, performance, 'o-', color='blue', markersize=10, linewidth=2)
    ax1.set_xlabel('Session Day', fontsize=12)
    ax1.set_ylabel('Performance (%)', fontsize=12)
    ax1.set_title(f'{animal_id}: Performance Across Sessions', fontsize=14)
    ax1.set_ylim(0, 105)  # Give a little room above 100%
    ax1.set_xticks(days)
    ax1.grid(True, alpha=0.3)

    boundary_label_added = False
    stress_label_added = False # Track if stress label is added
    if boundary_pos is not None:
        ax1.axvline(x=boundary_pos, color='red', linestyle='--', alpha=0.7, label='Training/Test Boundary')
        boundary_label_added = True
    if stress_boundary_pos is not None:
        ax1.axvline(x=stress_boundary_pos, color='blue', linestyle=':', alpha=0.7, label='Stress Phase Start')
        stress_label_added = True

    # Add data labels
    for i, perf in enumerate(performance):
        ax1.annotate(f"{perf:.1f}%",
                    (days[i], performance[i]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=10)

    # 2. Trial counts over days
    ax2 = fig.add_subplot(gs[0, 1])
    width = 0.35
    x = np.array(days)
    ax2.bar(x - width/2, total_trials, width, label='Total Trials', color='skyblue')
    ax2.bar(x + width/2, rewarded_trials, width, label='Rewarded Trials', color='green')
    ax2.set_xlabel('Session Day', fontsize=12)
    ax2.set_ylabel('Number of Trials', fontsize=12)
    ax2.set_title(f'{animal_id}: Trial Counts Across Sessions', fontsize=14)
    ax2.set_xticks(days)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if boundary_pos is not None:
        ax2.axvline(x=boundary_pos, color='red', linestyle='--', alpha=0.7, label='_nolegend_') # Use label only once
    if stress_boundary_pos is not None:
        ax2.axvline(x=stress_boundary_pos, color='blue', linestyle=':', alpha=0.7, label='_nolegend_')

    # Add data labels
    for i, (total, rewarded) in enumerate(zip(total_trials, rewarded_trials)):
        ax2.annotate(f"{total}",
                    (days[i] - width/2, total),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha='center',
                    fontsize=9)
        ax2.annotate(f"{rewarded}",
                    (days[i] + width/2, rewarded),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha='center',
                    fontsize=9)

    # 3. Average first lick time over days
    ax3 = fig.add_subplot(gs[1, 0])

    # Plot with error bars (using SEM instead of STD)
    valid_indices = [i for i, t in enumerate(avg_lick_times) if not np.isnan(t)]
    if valid_indices:
         # Calculate SEM only for valid indices to avoid warnings
         sem_to_plot = np.array(sem_lick_times)[valid_indices]
         ax3.errorbar(np.array(days)[valid_indices], np.array(avg_lick_times)[valid_indices], yerr=sem_to_plot, fmt='o-', color='purple',
                     markersize=8, capsize=5, linewidth=2, elinewidth=1.5)
    # If there are points but none are valid (all NaN), plot them without error bars
    elif len(days) > 0:
         ax3.plot(days, avg_lick_times, 'o-', color='purple', markersize=8, linewidth=2) # Plot without error bars if no valid SEM or all NaN


    ax3.set_xlabel('Session Day', fontsize=12)
    ax3.set_ylabel('First Lick Time (s)', fontsize=12)
    ax3.set_title(f'{animal_id}: Average First Lick Time Across Sessions (Error bars: SEM)', fontsize=14)
    ax3.set_xticks(days)
    ax3.grid(True, alpha=0.3)

    if boundary_pos is not None:
        ax3.axvline(x=boundary_pos, color='red', linestyle='--', alpha=0.7, label='_nolegend_')
    if stress_boundary_pos is not None:
        ax3.axvline(x=stress_boundary_pos, color='blue', linestyle=':', alpha=0.7, label='_nolegend_')

    # Add data labels
    for i, avg in enumerate(avg_lick_times):
        if not np.isnan(avg):
            ax3.annotate(f"{avg:.2f}s",
                        (days[i], avg),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=10)

    # 4. Box plot of first lick times by session
    ax4 = fig.add_subplot(gs[1, 1])

    # Filter out sessions with no lick times for boxplot
    plot_lick_times = [times for times in first_lick_times_by_session if times]
    plot_days = [day for day, times in zip(days, first_lick_times_by_session) if times]

    if plot_lick_times:
        # Create box plot with individual points
        box_parts = ax4.boxplot(plot_lick_times, positions=plot_days, patch_artist=True,
                               widths=0.6, showfliers=False)

        # Customize box plot colors
        for box in box_parts['boxes']:
            box.set(facecolor='lightblue', alpha=0.8)
        for median in box_parts['medians']:
            median.set(color='navy', linewidth=2)

        # Add individual data points (jittered)
        for i, lick_times in enumerate(plot_lick_times):
            if lick_times:
                # Add jitter to x-position
                jitter = np.random.normal(0, 0.05, size=len(lick_times))
                ax4.scatter(np.array([plot_days[i]] * len(lick_times)) + jitter, lick_times,
                           alpha=0.5, s=30, color='darkblue')
    else:
        ax4.text(0.5, 0.5, 'No lick time data available', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)


    ax4.set_xlabel('Session Day', fontsize=12)
    ax4.set_ylabel('First Lick Time (s)', fontsize=12)
    ax4.set_title(f'{animal_id}: Distribution of First Lick Times', fontsize=14)
    ax4.set_xticks(days) # Set ticks for all days, even if no boxplot
    ax4.grid(True, alpha=0.3)

    if boundary_pos is not None:
        ax4.axvline(x=boundary_pos, color='red', linestyle='--', alpha=0.7, label='_nolegend_')
    if stress_boundary_pos is not None:
        ax4.axvline(x=stress_boundary_pos, color='blue', linestyle=':', alpha=0.7, label='_nolegend_')


    # 5. Heatmap of first lick time distribution by session
    ax5 = fig.add_subplot(gs[2, :])

    # Create bins for the heatmap
    all_licks = [lick for session_licks in first_lick_times_by_session for lick in session_licks]
    max_lick_time = max(all_licks) if all_licks else 1 # Avoid error if no licks, default max to 1s
    bins = np.linspace(0, min(max_lick_time, 10), 50)  # Cap at 10s for better visualization

    # Create a 2D histogram-like data structure
    heatmap_data = np.zeros((len(days), len(bins)-1))

    for i, lick_times in enumerate(first_lick_times_by_session):
        if lick_times:
            # Count licks in each bin
            hist, _ = np.histogram(lick_times, bins=bins)
            # Normalize by total licks in session
            heatmap_data[i, :] = hist / len(lick_times)

    # Plot heatmap
    im = ax5.imshow(heatmap_data, aspect='auto', cmap='viridis',
                   extent=[bins[0], bins[-1], days[-1] + 0.5, days[0] - 0.5]) # Y axis uses days directly

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label('Proportion of First Licks', fontsize=12)

    ax5.set_xlabel('First Lick Time (s)', fontsize=12)
    ax5.set_ylabel('Session Day', fontsize=12)
    ax5.set_title(f'{animal_id}: First Lick Time Distribution Across Sessions', fontsize=14)
    ax5.set_yticks(days)

    # Add vertical lines at 1s intervals
    for t in range(1, int(bins[-1]) + 1):
        ax5.axvline(x=t, color='white', linestyle='--', alpha=0.3)

    if boundary_pos is not None:
        # Calculate y-coordinate based on imshow extent and the boundary position
        # The y-axis runs from days[-1]+0.5 (bottom) to days[0]-0.5 (top)
        # We need the boundary between max_training_day and min_test_day
        # The boundary position corresponds to max_training_day + 0.5 on the 'days' scale
        
        # Find the index corresponding to the max_training_day
        try:
             max_train_idx = days.index(max_training_day)
             # Calculate y-coordinate. Since y runs top-to-bottom with indices,
             # the line should be between index max_train_idx and max_train_idx + 1.
             # The imshow extent maps indices to y-values.
             # y = y_max - (index + 0.5) * y_step (approximately)
             # Let's use the boundary_pos directly in the day scale and find corresponding y
             y_coord = boundary_pos # This might not align perfectly with imshow pixels, use index approach instead
             
             # Calculate y-coordinate based on imshow extent mapping from index
             y_max = days[-1] + 0.5
             y_min = days[0] - 0.5
             num_days = len(days)
             # The line should be between the row for max_training_day (index max_train_idx)
             # and the row for min_test_day (index max_train_idx + 1)
             # The y coordinate for the top of row max_train_idx is y_max - (max_train_idx / (num_days-1)) * (y_max - y_min) if num_days > 1 else y_max
             # The y coordinate for the bottom of row max_train_idx is y_max - ((max_train_idx+1) / (num_days-1)) * (y_max - y_min) if num_days > 1 else y_min
             
             # Simpler approach: Use the boundary_pos value directly if yticks are set to days
             # Since ax5.set_yticks(days) is used, the y-axis should correspond to day numbers.
             # However, imshow maps the *center* of the bins. A line at boundary_pos should work.
             ax5.axhline(y=boundary_pos, color='red', linestyle='--', alpha=0.7, label='_nolegend_')

        except ValueError:
             print(f"DEBUG: Max training day {max_training_day} not found in days list for heatmap line.")
             pass # Should not happen if boundary_pos is set

    # Consolidate legend
    handles, labels = [], []
    # Collect handles/labels from all subplots
    for ax in [ax1, ax2, ax3, ax4]: # Exclude ax5 for now
        h, l = ax.get_legend_handles_labels()
        for label, handle in zip(l, h):
             if label not in labels and label != '_nolegend_':
                 labels.append(label)
                 handles.append(handle)
    
    # Manually add boundary line handles/labels if they were created
    ordered_handles = []
    ordered_labels = []
    # Add plot-specific items first
    plot_items = [(h, l) for h, l in zip(handles, labels) if l not in ['Training/Test Boundary', 'Stress Phase Start']]
    ordered_handles.extend([item[0] for item in plot_items])
    ordered_labels.extend([item[1] for item in plot_items])
    # Add boundary lines in specific order
    if boundary_label_added:
        boundary_handle = next((h for h, l in zip(handles, labels) if l == 'Training/Test Boundary'), None)
        if boundary_handle: 
            ordered_handles.append(boundary_handle)
            ordered_labels.append('Training/Test Boundary')
    if stress_label_added:
        stress_handle = next((h for h, l in zip(handles, labels) if l == 'Stress Phase Start'), None)
        if stress_handle:
            ordered_handles.append(stress_handle)
            ordered_labels.append('Stress Phase Start')

    # Create a single legend for the figure using collected handles/labels
    if ordered_handles:
         # Place legend outside the plot area to avoid overlap
         fig.legend(ordered_handles, ordered_labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Add overall title
    plt.suptitle(f'Performance Analysis for {animal_id}', fontsize=16, y=0.98)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.96]) # Adjust right margin for figure legend

    # Save figure
    filename = f"{animal_id}_performance_analysis.png"
    if output_dir:
        filepath = os.path.join(output_dir, filename)
    else:
        filepath = filename

    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved figure: {filepath}")

    # Close the figure to free memory instead of showing it
    plt.close(fig)

    # Check if lick sensor analysis data exists in *any* session
    has_lick_cut_data = any('lick_sensor_cut_analysis' in s for s in animal_sessions)

    if has_lick_cut_data:
        # Pass the full sorted list of sessions to the cut time visualization
        visualize_lick_sensor_cut_time(animal_sessions, animal_id, output_dir)
    else:
        print("No lick sensor cut analysis data found in any session.")

def visualize_session_details(session_data, animal_id, output_dir=None, show_in_notebook=False):
    """
    Create detailed visualization for a single session with two panels
    
    Parameters:
    -----------
    session_data : dict
        Dictionary containing session data
    animal_id : str
        Animal ID (passed explicitly for consistency)
    output_dir : str, optional
        Directory to save the figure
    show_in_notebook : bool, optional
        Whether to display the figure in the notebook
    """
    
    session_day = session_data['session_day']
    trial_results = session_data['trial_results']
    
    # Skip if no trial results
    if not trial_results:
        print(f"No trial results for {animal_id} Day {session_day}")
        return
    
    # Create figure with two panels stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                  gridspec_kw={'height_ratios': [1, 2]})
    
    # Extract trial data
    trial_nums = [t['trial_num'] for t in trial_results]
    rewarded = [t['rewarded'] for t in trial_results]
    
    # Calculate first lick latencies for test session data
    # Test sessions have 'sound_start' and 'first_lick_time' instead of 'first_lick_latency'
    if 'is_test_session' in session_data and session_data['is_test_session']:
        first_lick_latencies = []
        for t in trial_results:
            if 'first_lick_latency' in t and t['first_lick_latency'] is not None:
                first_lick_latencies.append(t['first_lick_latency'])
            else:
                first_lick_latencies.append(np.nan)
    else:
        # For training sessions, first_lick_latency should be available
        first_lick_latencies = [t.get('first_lick_latency', np.nan) for t in trial_results]
    
    # Determine trial number range for consistent x-axis limits
    min_trial = min(trial_nums)
    max_trial = max(trial_nums)
    x_range = [min_trial - 1, max_trial + 1]  # Add padding for better visualization
    
    # 1. Trial outcomes (rewarded/not rewarded)
    ax1.bar(trial_nums, [1 if r else 0.3 for r in rewarded], 
           color=['green' if r else 'red' for r in rewarded])
    ax1.set_xlabel('', fontsize=12)  # Remove x-label since it will be covered by panel 2
    ax1.set_ylabel('Outcome', fontsize=12)
    ax1.set_title(f'{animal_id}: Trial Outcomes (Day {session_day})', fontsize=14)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Not Rewarded', 'Rewarded'])
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(x_range)  # Set x-axis limits to match between panels
    
    # Add text showing performance
    performance = sum(rewarded) / len(rewarded) * 100
    ax1.text(0.02, 0.85, f"Performance: {performance:.1f}% ({sum(rewarded)}/{len(rewarded)} trials)",
            transform=ax1.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # 2. First lick latencies by trial (colored by previous reward size)
    # Default color is gray for trials without previous reward size info
    colors = ['gray'] * len(trial_nums)
    sizes = [40] * len(trial_nums)  # Slightly larger base size
    size_labels = ['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
    
    # Define a colormap for different reward sizes
    cmap = plt.cm.viridis
    
    # Color based on previous reward size
    for i in range(1, len(trial_results)):
        curr_trial = trial_results[i]
        prev_trial = trial_results[i-1]
        
        # Skip if previous trial wasn't rewarded
        if not prev_trial['rewarded']:
            continue
            
        if 'reward_size' in prev_trial and prev_trial['reward_size'] is not None:
            prev_reward_size = prev_trial['reward_size']
            
            # Use color based on previous reward size (normalize to 0-1 range for colormap)
            max_reward_size = 5  # Assuming reward sizes are 1-5
            norm_size = (prev_reward_size - 1) / (max_reward_size - 1)
            colors[i] = cmap(norm_size)
            sizes[i] = 60  # Make these dots slightly larger
    
    # Plot first lick latencies with colors
    for i, (x, y, c, s) in enumerate(zip(trial_nums, first_lick_latencies, colors, sizes)):
        if not np.isnan(y):
            ax2.scatter(x, y, color=c, s=s, edgecolors='none', alpha=0.8)
    
    # Add connecting lines (colored gray)
    ax2.plot(trial_nums, first_lick_latencies, '-', color='gray', alpha=0.3, linewidth=1)
    
    ax2.set_xlabel('Trial Number', fontsize=12)
    ax2.set_ylabel('First Lick Latency (s)', fontsize=12)
    ax2.set_title(f'{animal_id}: First Lick Latencies by Trial (Day {session_day})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(x_range)  # Set x-axis limits to match between panels
    
    # Add horizontal line at median lick time
    median_lick = np.nanmedian(first_lick_latencies)
    ax2.axhline(y=median_lick, color='blue', linestyle='--', alpha=0.7)
    ax2.text(0.98, 0.05, f'Median: {median_lick:.2f}s', 
            transform=ax2.transAxes, ha='right', va='bottom', color='blue', fontsize=10)
    
    # Create inset legend for reward sizes (only for test sessions)
    if 'is_test_session' in session_data and session_data['is_test_session']:
        # Create inset legend instead of colorbar
        inset_ax = ax2.inset_axes([0.02, 0.65, 0.2, 0.3])
        inset_ax.set_axis_off()  # Turn off the inset axes
        
        # Create legend elements for each reward size
        reward_sizes = range(1, 6)  # Assuming 5 reward sizes
        legend_elements = []
        
        for size in reward_sizes:
            norm_size = (size - 1) / (len(reward_sizes) - 1)
            color = cmap(norm_size)
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor=color, markersize=10, 
                                             label=f'{size}: {size_labels[size-1]}'))
        
        # Add gray for trials without previous reward info
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor='gray', markersize=8, 
                                         label='No previous reward'))
        
        # Create the legend in the inset
        inset_ax.legend(handles=legend_elements, loc='center', 
                       title='Previous Trial Reward', framealpha=0.9,
                       fontsize=8, title_fontsize=9)
    
    # Add overall title
    plt.suptitle(f'Session Analysis for {animal_id} - Day {session_day}', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Determine the correct filename based on session type
    # --- REMOVE DEBUG PRINT --- 
    if 'is_test_session' in session_data and session_data['is_test_session']:
        filename = f"{animal_id}_TEST_Day{session_day}_session_analysis.png"
        # print(f"DEBUG - Test session filename: {filename}") # REMOVED
    else:
        filename = f"{animal_id}_Day{session_day}_session_analysis.png"
        # print(f"DEBUG - Training session filename: {filename}") # REMOVED (also removing this just in case)
    # ------------------------
    
    if output_dir:
        filepath = os.path.join(output_dir, filename)
    else:
        filepath = filename
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved figure: {filepath}")
    
    # Close the figure to free memory - don't call show() at all
    plt.close(fig)

def visualize_test_session(session_data, animal_id, figures_dir):
    """
    Create visualizations for a test session, including reward size analysis
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12)) # Increased height slightly for legend
    
    # --- MODIFICATION START: Define Colormap and Normalization --- 
    cmap = plt.cm.viridis
    max_reward_size = 5 # Assuming reward sizes 1-5
    norm = matplotlib.colors.Normalize(vmin=1, vmax=max_reward_size)
    size_labels = ['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
    # --- MODIFICATION END ---
    
    # 1. Response latency by previous reward size (now in top-left panel)
    ax1 = plt.subplot(2, 2, 1)
    
    latencies_by_prev_reward = {}
    trial_results = session_data['trial_results']
    
    for i in range(1, len(trial_results)):
        curr_trial = trial_results[i]
        prev_trial = trial_results[i-1]
        
        # Skip if previous trial wasn't rewarded
        if not prev_trial['rewarded']:
            continue
            
        if 'reward_size' in prev_trial and prev_trial['reward_size'] is not None:
            prev_reward_size = prev_trial['reward_size']
            
            # Get current trial's latency
            if 'first_lick_latency' in curr_trial and curr_trial['first_lick_latency'] is not None:
                if prev_reward_size not in latencies_by_prev_reward:
                    latencies_by_prev_reward[prev_reward_size] = []
                latencies_by_prev_reward[prev_reward_size].append(curr_trial['first_lick_latency'])
    
    if latencies_by_prev_reward:
        prev_reward_sizes = sorted(latencies_by_prev_reward.keys())
        latency_data = [latencies_by_prev_reward[size] for size in prev_reward_sizes]
        
        # --- Calculate Mean and SEM --- 
        means = [np.mean(l) for l in latency_data]
        sems = [sem(l) if len(l) > 1 else 0 for l in latency_data]
        # ------------------------------
        
        # --- MODIFICATION: Replace Box Plot with Bar Plot --- 
        # Plot bars for mean latency with SEM error bars
        ax1.bar(prev_reward_sizes, means, yerr=sems, capsize=5, 
                color='lightgrey', alpha=0.7, ecolor='black', 
                label='Mean Latency +/- SEM') # Add label for legend if needed
        # --------------------------------------------------

        # --- Keep Individual data points (jittered) --- 
        cmap = plt.cm.viridis # Ensure cmap is defined
        norm = matplotlib.colors.Normalize(vmin=1, vmax=5) # Ensure norm is defined
        all_plotted_latencies = [] # Collect all latencies for robust ylim calculation
        for i, size in enumerate(prev_reward_sizes):
            latencies = latencies_by_prev_reward[size]
            if latencies:
                all_plotted_latencies.extend(latencies) # Add to combined list
                # Add jitter to x-position
                jitter = np.random.normal(0, 0.05, size=len(latencies))
                point_color = cmap(norm(size))
                ax1.scatter(np.array([size] * len(latencies)) + jitter, latencies, 
                        alpha=0.6, s=30, color=point_color, edgecolors='none')
        # --------------------------------------------
        
        ax1.set_xlabel('Previous Trial Reward Size', fontsize=12)
        ax1.set_ylabel('Response Latency (s)', fontsize=12)
        # --- MODIFICATION: Update title --- 
        ax1.set_title(f'Mean Response Latency by Previous Reward Size (Test Day {session_data["session_day"]})', fontsize=14)
        # ----------------------------------
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(prev_reward_sizes)
        ax1.set_xticklabels([size_labels[size-1] for size in prev_reward_sizes if 1 <= size <= len(size_labels)])
        
        # --- Calculate Y-limits based on IQR of plotted data --- 
        if all_plotted_latencies:
            q1, q3 = np.percentile(all_plotted_latencies, [25, 75])
            iqr = q3 - q1
            upper_whisker = q3 + 1.5 * iqr
            # Set ylim, add some padding, ensure a minimum sensible upper limit (e.g., 1s or 2s)
            final_upper_limit = max(upper_whisker * 1.1, 1.0) # Add 10% padding, minimum 1.0s
            ax1.set_ylim(0, final_upper_limit)
        else:
             ax1.set_ylim(0, 1) # Default limit if no latency data
        # ------------------------------------------------------
        
        # Optional: Add legend if bar label was added
        # ax1.legend()
    else:
        # If no valid data, show message
        ax1.text(0.5, 0.5, 'No valid response latency data after rewarded trials', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Response Latency by Previous Reward Size', fontsize=14)
        ax1.set_xticks([])
        ax1.set_yticks([])
    
    # 2. Trial counts by reward size
    ax2 = plt.subplot(2, 2, 2)
    reward_sizes_counts = sorted(session_data['performance_by_reward_size'].keys())
    trial_counts = [session_data['performance_by_reward_size'][size]['total'] for size in reward_sizes_counts]
    bar_colors_counts = [cmap(norm(size)) for size in reward_sizes_counts]
    bars_container_counts = ax2.bar(reward_sizes_counts, trial_counts)
    ax2.set_xlabel('Reward Size')
    ax2.set_ylabel('Number of Trials')
    ax2.set_title('Trial Distribution by Reward Size')
    ax2.grid(axis='y', alpha=0.5)
    ax2.set_xticks(reward_sizes_counts)
    ax2.set_xticklabels([size_labels[size-1] for size in reward_sizes_counts if 1 <= size <= len(size_labels)], rotation=45, ha='right')
    
    # 3. Performance by reward size
    ax3 = plt.subplot(2, 2, 3)
    reward_sizes_perf = sorted(session_data['performance_by_reward_size'].keys())
    performances = [session_data['performance_by_reward_size'][size]['performance'] for size in reward_sizes_perf]
    bar_colors_perf = [cmap(norm(size)) for size in reward_sizes_perf]
    bars_container_perf = ax3.bar(reward_sizes_perf, performances)
    ax3.set_xlabel('Reward Size')
    ax3.set_ylabel('Performance (%)')
    ax3.set_title('Performance by Reward Size')
    ax3.set_ylim(0, 105)
    ax3.grid(axis='y', alpha=0.5)
    ax3.set_xticks(reward_sizes_perf)
    ax3.set_xticklabels([size_labels[size-1] for size in reward_sizes_perf if 1 <= size <= len(size_labels)], rotation=45, ha='right')
    
    # 4. Trial timeline (colors unchanged - based on current trial reward size)
    ax4 = plt.subplot(2, 2, 4)
    trial_numbers = [t['trial_num'] for t in session_data['trial_results']]
    current_reward_sizes = [t['reward_size'] for t in session_data['trial_results']]
    responses = [t['rewarded'] for t in session_data['trial_results']]
    
    # Map reward sizes to colors for this plot specifically
    timeline_colors = [cmap(norm(size)) if size is not None else 'gray' for size in current_reward_sizes]
    
    # Plot rewarded trials as circles with Viridis colors
    rewarded_indices = [i for i, r in enumerate(responses) if r]
    if rewarded_indices:
         ax4.scatter(np.array(trial_numbers)[rewarded_indices], np.array(current_reward_sizes)[rewarded_indices],
                     c=np.array(timeline_colors)[rewarded_indices], 
                     marker='o', s=100, label='Rewarded') # Added label

    # Add red X for unrewarded trials
    unrewarded_indices = [i for i, r in enumerate(responses) if not r]
    if unrewarded_indices:
        unrewarded_numbers = np.array(trial_numbers)[unrewarded_indices]
        unrewarded_sizes = np.array(current_reward_sizes)[unrewarded_indices]
        # Plot X markers with gray color if size is None, else use Viridis
        unrewarded_colors = np.array(timeline_colors)[unrewarded_indices]
        ax4.scatter(unrewarded_numbers, unrewarded_sizes, 
                   color='red', marker='x', s=100, label='Not Rewarded') # Added label
    
    ax4.set_xlabel('Trial Number')
    ax4.set_ylabel('Reward Size')
    ax4.set_title('Trial Timeline')
    ax4.grid(True, alpha=0.5)
    # Ensure unique reward sizes for ticks, handling potential None
    unique_sizes = sorted([s for s in set(current_reward_sizes) if s is not None])
    if unique_sizes:
        ax4.set_yticks(unique_sizes)
        ax4.set_yticklabels([size_labels[size-1] for size in unique_sizes if 1 <= size <= len(size_labels)])
    ax4.legend() # Add legend for rewarded/not rewarded markers
    
    # --- MODIFICATION START: Add shared colorbar --- 
    # Add a colorbar to the figure, linked to the normalization used
    # Place it slightly to the right
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7]) # x, y, width, height
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.set_label('Reward Size', fontsize=12)
    # Set ticks for the colorbar if reward sizes are known
    if unique_sizes:
        cb.set_ticks(unique_sizes)
        cb.set_ticklabels([size_labels[size-1] for size in unique_sizes if 1 <= size <= len(size_labels)])
    # --- MODIFICATION END ---
    
    # Add overall title
    plt.suptitle(f'Test Session Analysis for {animal_id} - Day {session_data["session_day"]}', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.85, 0.96]) # Adjust rect for colorbar and suptitle

    # --- MODIFICATION START: Apply facecolor AFTER tight_layout ---
    # Apply colors just before saving, hoping to override any style interference
    try:
        if 'bars_container_counts' in locals() and 'bar_colors_counts' in locals():
            print("Applying facecolor to ax2 bars (trial counts) before save...")
            for bar, color in zip(bars_container_counts, bar_colors_counts):
                bar.set_facecolor(color)
        else: 
             print("Skipping facecolor for ax2 bars (variables not found)")

        if 'bars_container_perf' in locals() and 'bar_colors_perf' in locals():
            print("Applying facecolor to ax3 bars (performance) before save...")
            for bar, color in zip(bars_container_perf, bar_colors_perf):
                bar.set_facecolor(color)
        else:
             print("Skipping facecolor for ax3 bars (variables not found)")
             
    except Exception as e:
        print(f"Error applying facecolor before save: {e}")
    # --- MODIFICATION END ---
    
    # Save figure
    test_day = session_data['session_day']
    fig_name = f"{animal_id}_TEST_Day{test_day}_analysis.png"
    fig_path = os.path.join(figures_dir, fig_name)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Test session visualization saved to: {fig_path}")
    
    # 2. Create lick sensor cut analysis figure if available
    if 'lick_sensor_cut_analysis' in session_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Extract trial data from lick sensor cut analysis
        trial_metrics = session_data['lick_sensor_cut_analysis']['trial_metrics']
        trial_nums = [t['trial_num'] for t in trial_metrics]
        pre_cue_pcts = [t['pct_cut_during_pre_cue'] for t in trial_metrics]
        post_cue_pcts = [t['pct_cut_during_post_cue'] for t in trial_metrics]
        
        # Extract ITI lick cut durations from the trial_results
        trial_results = session_data['trial_results']
        iti_cut_durations = []
        trial_nums_with_iti_data = []
        for trial_num in trial_nums:
            # Find the corresponding trial in trial_results
            matched_trial = next((t for t in trial_results if t['trial_num'] == trial_num), None)
            if matched_trial and 'ITI_lick_cut_duration' in matched_trial:
                iti_cut_durations.append(matched_trial['ITI_lick_cut_duration'])
                trial_nums_with_iti_data.append(trial_num)
        
        # Plot 1: ITI Lick Cut Duration (in seconds) by trial
        ax1.plot(trial_nums_with_iti_data, iti_cut_durations, 'o-', color='red', label='ITI Lick Duration (seconds)')
        ax1.set_xlabel('Trial Number', fontsize=12)
        ax1.set_ylabel('Lick Sensor Cut Duration (seconds)', fontsize=12)
        ax1.set_title(f'{animal_id}: ITI Lick Duration\n(Window: cue start + 6.5s to cue end + 17s)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add mean line for ITI duration
        if iti_cut_durations:
            mean_iti_duration = np.mean(iti_cut_durations)
            ax1.axhline(y=mean_iti_duration, color='black', linestyle='--', 
                      label=f'Mean: {mean_iti_duration:.2f}s', alpha=0.7)
            # Update legend to include the mean line
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles, labels)
        
        # Plot 2: Pre-cue vs Post-cue cut time by trial (unchanged)
        ax2.plot(trial_nums, pre_cue_pcts, 'o-', color='green', label='Pre-Cue Window')
        ax2.plot(trial_nums, post_cue_pcts, 'o-', color='orange', label='Post-Cue Window')
        ax2.set_xlabel('Trial Number', fontsize=12)
        ax2.set_ylabel('% Time Lick Sensor Cut', fontsize=12)
        ax2.set_title(f'{animal_id}: Pre-Cue vs Post-Cue Cut Time (Test Day {session_data["session_day"]})', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add overall title
        plt.suptitle(f'Lick Sensor Cut Analysis for {animal_id} - Test Day {session_data["session_day"]}', fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save lick sensor cut analysis figure
        filename = f"{animal_id}_TEST_Day{session_data['session_day']}_lick_sensor_cut_analysis.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved figure: {filepath}")
        
        plt.close(fig)

def visualize_test_sessions(animal_sessions, animal_id, figures_dir):
    """
    Create visualizations for all test sessions of an animal
    """
    if not animal_sessions:
        print("No sessions to visualize")
        return
    
    # Sort sessions by day
    animal_sessions.sort(key=lambda x: x['session_day'])
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Overall performance across sessions
    ax1 = plt.subplot(2, 2, 1)
    session_days = [s['session_day'] for s in animal_sessions]
    performances = [s['performance'] for s in animal_sessions]
    
    ax1.plot(session_days, performances, 'o-')
    ax1.set_xlabel('Test Day')
    ax1.set_ylabel('Overall Performance (%)')
    ax1.set_title('Performance Across Test Sessions')
    ax1.grid(True)
    ax1.set_ylim(0, 100)
    
    # 2. Performance by reward size across sessions
    ax2 = plt.subplot(2, 2, 2)
    reward_sizes = set()
    for session in animal_sessions:
        reward_sizes.update(session['performance_by_reward_size'].keys())
    reward_sizes = sorted(list(reward_sizes))
    
    for size in reward_sizes:
        size_performances = []
        for session in animal_sessions:
            if size in session['performance_by_reward_size']:
                size_performances.append(
                    session['performance_by_reward_size'][size]['performance']
                )
            else:
                size_performances.append(np.nan)
        
        ax2.plot(session_days, size_performances, 'o-', 
                label=f'Size {size}')
    
    ax2.set_xlabel('Test Day')
    ax2.set_ylabel('Performance (%)')
    ax2.set_title('Performance by Reward Size')
    ax2.grid(True)
    ax2.set_ylim(0, 100)
    ax2.legend()
    
    # 3. Response latency by reward size across sessions
    ax3 = plt.subplot(2, 2, 3)
    
    for size in reward_sizes:
        latencies = []
        for session in animal_sessions:
            if size in session['first_lick_times_by_reward_size']:
                stats = session['first_lick_times_by_reward_size'][size]
                if isinstance(stats, dict) and 'mean' in stats:
                    latencies.append(stats['mean'])
                else:
                    latencies.append(np.nan)
            else:
                latencies.append(np.nan)
        
        ax3.plot(session_days, latencies, 'o-', 
                label=f'Size {size}')
    
    ax3.set_xlabel('Test Day')
    ax3.set_ylabel('First Lick Latency (s)')
    ax3.set_title('Response Latency by Reward Size')
    ax3.grid(True)
    ax3.legend()
    
    # 4. Trial counts by reward size across sessions
    ax4 = plt.subplot(2, 2, 4)
    
    for size in reward_sizes:
        counts = []
        for session in animal_sessions:
            if size in session['performance_by_reward_size']:
                counts.append(
                    session['performance_by_reward_size'][size]['total']
                )
            else:
                counts.append(0)
        
        ax4.plot(session_days, counts, 'o-', 
                label=f'Size {size}')
    
    ax4.set_xlabel('Test Day')
    ax4.set_ylabel('Number of Trials')
    ax4.set_title('Trial Counts by Reward Size')
    ax4.grid(True)
    ax4.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save figure
    fig_name = f"{animal_id}_TEST_sessions_analysis.png"
    fig_path = os.path.join(figures_dir, fig_name)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Test sessions visualization saved to: {fig_path}")
    
    # Generate individual session visualizations
    for session in animal_sessions:
        visualize_test_session(session, animal_id, figures_dir)

def visualize_sessions(sessions, animal_id, figures_dir):
    """Generate visualizations for all sessions, including both training and test sessions."""
    # Sort sessions by day
    sessions.sort(key=lambda x: x['session_day'])
    
    # Create figure for all sessions
    plt.figure(figsize=(15, 10))
    
    # Plot performance over sessions
    plt.subplot(2, 2, 1)
    session_days = [s['session_day'] for s in sessions]
    performances = [s['performance'] for s in sessions]
    plt.plot(session_days, performances, 'o-', label='Performance')
    plt.xlabel('Session Day')
    plt.ylabel('Performance (%)')
    plt.title('Performance Over Sessions')
    plt.grid(True)
    
    # Plot reward size distribution (only for test sessions)
    plt.subplot(2, 2, 2)
    reward_sizes = []
    for session in sessions:
        if 'performance_by_reward_size' in session:
            for size, metrics in session['performance_by_reward_size'].items():
                reward_sizes.append(size)
    if reward_sizes:  # Only plot if we have reward size data
        plt.hist(reward_sizes, bins=range(1, 7), align='left', rwidth=0.8)
        plt.xlabel('Reward Size')
        plt.ylabel('Count')
        plt.title('Reward Size Distribution')
    else:
        plt.text(0.5, 0.5, 'No reward size data available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Reward Size Distribution')
    plt.grid(True)
    
    # Plot performance by reward size (only for test sessions)
    plt.subplot(2, 2, 3)
    test_sessions = [s for s in sessions if s.get('is_test_session', False)]
    if test_sessions:
        reward_sizes = sorted(set(size for session in test_sessions 
                                for size in session['performance_by_reward_size'].keys()))
        performances_by_size = {size: [] for size in reward_sizes}
        for session in sessions:
            for size in reward_sizes:
                if 'performance_by_reward_size' in session and size in session['performance_by_reward_size']:
                    performances_by_size[size].append(session['performance_by_reward_size'][size]['performance'])
                else:
                    performances_by_size[size].append(0)
        
        for size in reward_sizes:
            plt.plot(session_days, performances_by_size[size], 'o-', label=f'Size {size}')
        plt.xlabel('Session Day')
        plt.ylabel('Performance (%)')
        plt.title('Performance by Reward Size')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No reward size data available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Performance by Reward Size')
    plt.grid(True)
    
    # Add test session information if available
    test_sessions = [s for s in sessions if s.get('is_test_session', False)]
    if test_sessions:
        plt.subplot(2, 2, 4)
        test_days = [s['session_day'] for s in test_sessions]
        test_performances = [s['performance'] for s in test_sessions]
        plt.plot(test_days, test_performances, 'ro-', label='Test Sessions')
        plt.xlabel('Session Day')
        plt.ylabel('Performance (%)')
        plt.title('Test Session Performance')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'{animal_id}_sessions_analysis.png'))
    plt.close()
    
    # Create individual session plots
    for session in sessions:
        plt.figure(figsize=(12, 8))
        
        # Plot trial-by-trial performance
        plt.subplot(2, 1, 1)
        trials = range(1, session['total_trials'] + 1)
        # Handle both list and integer rewarded_trials
        if isinstance(session['rewarded_trials'], list):
            rewards = [1 if i in session['rewarded_trials'] else 0 for i in trials]
        else:
            # If rewarded_trials is an integer, create a list of 1s up to that number
            rewards = [1 if i <= session['rewarded_trials'] else 0 for i in trials]
        plt.plot(trials, rewards, 'o-', markersize=3)
        plt.xlabel('Trial Number')
        plt.ylabel('Reward (1) / No Reward (0)')
        plt.title(f'Session {session["session_day"]} - Trial-by-Trial Performance')
        plt.grid(True)
        
        # Plot performance by reward size (only for test sessions)
        plt.subplot(2, 1, 2)
        if 'performance_by_reward_size' in session:
            sizes = sorted(session['performance_by_reward_size'].keys())
            performances = [session['performance_by_reward_size'][size]['performance'] for size in sizes]
            plt.bar(sizes, performances)
            plt.xlabel('Reward Size')
            plt.ylabel('Performance (%)')
            plt.title(f'Session {session["session_day"]} - Performance by Reward Size')
        else:
            plt.text(0.5, 0.5, 'No reward size data available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'Session {session["session_day"]} - Performance by Reward Size')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'{animal_id}_Day{session["session_day"]}_analysis.png'))
        plt.close()

def visualize_post_reward_licking(aggregated_results, windows_to_analyze, animal_id, session_label, figures_dir_absolute):
    """
    Generates a multi-panel plot visualizing post-reward lick duration across different windows.

    Args:
        aggregated_results (dict): Dictionary with aggregated mean/SEM lick durations.
                                    Structure: {window_label: {'means': {rew_size: mean}, 'sems': {rew_size: sem}, 'n': {rew_size: count}}}
        windows_to_analyze (list): List of dictionaries defining the windows.
                                   Each dict: {'label': str, 'start': float, 'duration': float}
        animal_id (str): ID of the animal.
        session_label (str): Label for the session (e.g., 'TEST_Day1').
        figures_dir_absolute (str): Absolute path to the directory where the figure will be saved.
    """

    # --- Define Reward Size Labels --- 
    size_labels_map = {1: 'Very Small', 2: 'Small', 3: 'Medium', 4: 'Large', 5: 'Very Large'}
    # ---------------------------------

    num_windows = len(windows_to_analyze)
    if num_windows == 0:
        print("Warning: No windows provided for post-reward lick visualization.")
        return

    fig, axes = plt.subplots(num_windows, 1,
                             figsize=(6, 2 * num_windows + 1), # Adjust height based on panels
                             sharex=True)
    if num_windows == 1: axes = [axes] # Make it iterable if only one subplot

    # Find overall reward size range for consistent coloring
    all_reward_sizes = set()
    for window_data in aggregated_results.values():
        if 'means' in window_data:
            all_reward_sizes.update(window_data['means'].keys())

    if not all_reward_sizes:
        print(f"Warning: No data found in aggregated_results for {session_label}. Cannot generate plot.")
        plt.close(fig) # Close the empty figure
        return
        
    min_rs = min(all_reward_sizes)
    max_rs = max(all_reward_sizes)
    cmap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=min_rs, vmax=max_rs)

    plot_successful = False
    max_y_overall = 0.0 # Track max y across all panels for consistent ylim

    # Prepare list to store reward sizes present in the plot
    plotted_reward_sizes = set()

    # --- Plot data for each window --- 
    for idx, window_info in enumerate(windows_to_analyze):
        ax = axes[idx]
        window_label = window_info['label']
        window_data = aggregated_results.get(window_label, {})
        means = window_data.get('means', {})
        sems = window_data.get('sems', {})
        # ns = window_data.get('n', {}) # N value no longer needed for label

        if not means: # Check if there are means for this window
            ax.text(0.5, 0.5, 'No data for this window', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='gray')
            ax.set_title(f"Window: {window_label} Post-Reward", fontsize=10)
            ax.set_yticks([]) # Hide y-ticks if no data
            # --- Set fixed ylim for first two panels even if no data --- 
            if idx < 2:
                ax.set_ylim(0, 1)
            # ----------------------------------------------------------
            continue # Skip to the next window

        plot_successful = True # Mark that at least one panel has data
        sorted_reward_sizes = sorted(means.keys())
        plotted_reward_sizes.update(sorted_reward_sizes) # Add sizes plotted in this panel
        
        plot_mean_durations = [means[s] for s in sorted_reward_sizes]
        plot_sem_durations = [sems.get(s, 0) for s in sorted_reward_sizes] # Default SEM to 0 if missing
        # plot_ns = [ns.get(s, 0) for s in sorted_reward_sizes] # N value no longer needed
        bar_colors = [cmap(norm(size)) for size in sorted_reward_sizes]

        bars = ax.bar(sorted_reward_sizes, plot_mean_durations, yerr=plot_sem_durations, 
                      capsize=4, color=bar_colors, alpha=0.8, 
                      error_kw={'ecolor': 'gray', 'lw': 1.5})

        ax.set_title(f"Window: {window_label} Post-Reward", fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # --- Y-axis Limit Handling --- 
        if idx < 2: # First two panels (0-1s, 1-2s)
            y_upper_limit = 1.0
            ax.set_ylim(0, y_upper_limit)
        else: # Third panel (5-15s)
            # Determine max y value for this specific subplot to set ylim and position text
            max_y_this_plot = 0
            for mean_val, sem_val in zip(plot_mean_durations, plot_sem_durations):
                if not np.isnan(mean_val):
                    upper_bound = mean_val + (sem_val if not np.isnan(sem_val) else 0)
                    max_y_this_plot = max(max_y_this_plot, upper_bound)
            
            y_upper_limit = max(max_y_this_plot * 1.15, 0.05) # Add 15% padding, min 0.05
            ax.set_ylim(0, y_upper_limit)
            # max_y_overall = max(max_y_overall, y_upper_limit) # Not needed if axes aren't shared
        # ------------------------------

        # Add mean labels above bars
        for i, bar in enumerate(bars):
             yval = bar.get_height()
             yerr = plot_sem_durations[i]
             # n_val = plot_ns[i] # N value no longer needed
             text_offset = y_upper_limit * 0.02
             text_y = yval + (yerr if not np.isnan(yerr) else 0) + text_offset
             # Ensure text is within plot bounds
             text_y = min(text_y, y_upper_limit * 0.98)
             # Add Mean value label
             ax.text(bar.get_x() + bar.get_width()/2.0, text_y, f"{yval:.2f}", 
                      va='bottom', ha='center', fontsize=8, color='black') # Changed label and color
                          
        # --- Remove individual Y-axis label --- 
        # ax.set_ylabel("Mean Cut Duration (s)")
        # ------------------------------------

    # --- Apply overall figure adjustments --- 
    if plot_successful:
        # Add descriptive X labels to the last subplot
        final_sorted_sizes = sorted(list(plotted_reward_sizes))
        size_labels = [size_labels_map.get(s, str(s)) for s in final_sorted_sizes]
        axes[-1].set_xlabel("Reward Size")
        axes[-1].set_xticks(final_sorted_sizes)
        axes[-1].set_xticklabels(size_labels, rotation=45, ha='right', fontsize=9)
        
        # --- Add shared Y-axis label --- 
        fig.text(0.02, 0.5, "Mean Cut Duration (s)", va='center', rotation='vertical', fontsize=12)
        # ---------------------------------

        fig.suptitle(f"{animal_id} - {session_label}\nPost-Reward Lick Duration", y=0.99, fontsize=14)

        # Adjust layout to prevent overlap, accommodate shared label and rotated x-labels
        plt.tight_layout(rect=[0.06, 0.05, 1, 0.95]) # Left margin for shared Y, bottom for rotated X

    else:
        fig.suptitle(f"{animal_id} - {session_label}: No Post-Reward Data", y=0.98)
        plt.tight_layout()

    # --- Save the figure --- 
    filename = f"{animal_id}_{session_label}_post_reward_lick_duration.png"
    filepath = os.path.join(figures_dir_absolute, filename)
    try:
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved post-reward licking figure: {filepath}")
    except Exception as e:
        print(f"Error saving figure {filepath}: {e}")

    plt.close(fig)

def visualize_lick_sensor_cut_duration(session_data, animal_id, output_dir=None):
    """
    Create a plot showing the lick sensor cut duration during the 1.5 seconds after cue onset
    for each trial in a test session.
    
    Parameters:
    -----------
    session_data : dict
        Dictionary containing session data with lick sensor cut analysis
    animal_id : str
        Animal identifier
    output_dir : str, optional
        Directory to save the figure
        
    Returns:
    --------
    None
    """
    # Check if the session data contains lick sensor analysis
    if 'lick_sensor_cut_analysis' not in session_data:
        print("No lick sensor cut analysis data available for this session")
        return
    
    # Get session label
    session_label = f"TEST_Day{session_data['session_day']}"
    
    # Extract trial metrics
    lick_metrics = session_data['lick_sensor_cut_analysis']['trial_metrics']
    
    if not lick_metrics:
        print("No trial metrics available for this session")
        return
    
    # Extract trial numbers and cut durations
    trial_nums = [t['trial_num'] for t in lick_metrics]
    cut_durations = [t['post_cue_cut_duration'] for t in lick_metrics]
    rewarded = [t['rewarded'] for t in lick_metrics]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Set colors based on reward status
    colors = ['green' if r else 'red' for r in rewarded]
    
    # Create the bar plot
    bars = ax.bar(trial_nums, cut_durations, color=colors, alpha=0.7)
    
    # Add a horizontal line at the mean duration
    mean_duration = np.mean(cut_durations)
    ax.axhline(y=mean_duration, color='black', linestyle='--', alpha=0.7, label=f'Mean: {mean_duration:.2f}s')
    
    # Add rewarded/non-rewarded mean lines if both types exist
    if any(rewarded) and not all(rewarded):
        rewarded_mean = np.mean([d for d, r in zip(cut_durations, rewarded) if r])
        nonrewarded_mean = np.mean([d for d, r in zip(cut_durations, rewarded) if not r])
        ax.axhline(y=rewarded_mean, color='green', linestyle=':', alpha=0.7, 
                   label=f'Rewarded Mean: {rewarded_mean:.2f}s')
        ax.axhline(y=nonrewarded_mean, color='red', linestyle=':', alpha=0.7, 
                   label=f'Non-rewarded Mean: {nonrewarded_mean:.2f}s')
    
    # Set axis labels and title
    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Lick Sensor Cut Duration (seconds)', fontsize=12)
    ax.set_title(f'{animal_id} {session_label}: Lick Sensor Cut Duration\n(1.5s window after cue onset)', fontsize=14)
    
    # Set x-axis ticks to show all trial numbers
    ax.set_xticks(trial_nums)
    
    # Add a horizontal line at the maximum possible duration (1.5s)
    ax.axhline(y=1.5, color='blue', linestyle='-', alpha=0.3, label='Max Possible (1.5s)')
    
    # Add grid lines for better readability
    ax.grid(True, alpha=0.3)

    # --- Explicit Y-axis Tick Configuration ---
    ax.set_yticks([0, 0.5, 1.0]) 
    ax.set_yticklabels(['0.0', '0.5', '1.0']) 
    ax.tick_params(axis='both', which='major', labelsize=8) 
    ax.set_ylim(-0.05, 1.05) 
    # -----------------------------------------
    
    # Add a legend
    ax.legend(loc='upper right')
    
    # Add text labels with durations above each bar
    for i, (trial, duration) in enumerate(zip(trial_nums, cut_durations)):
        ax.text(trial, duration + 0.05, f"{duration:.2f}s", 
                ha='center', va='bottom', fontsize=8, 
                rotation=45 if len(trial_nums) > 20 else 0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output directory is provided
    if output_dir:
        filename = f"{animal_id}_{session_label}_lick_cut_duration.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved lick cut duration figure: {filepath}")
    
    # Close the figure to free memory
    plt.close(fig)
    return

# ------------------------------------------------------------------------------------
# --- NEW FUNCTION: Cross-Session Summary based on Previous Reward Size ---
# ------------------------------------------------------------------------------------

def visualize_cross_session_test_analysis(test_sessions, animal_id, output_dir):
    """
    Generates a 5x2 summary plot comparing latency and lick sensor cut duration 
    across test sessions, grouped by the reward size of the *previous* trial.
    Shows individual trial data points and mean+/-SEM per day.

    Parameters:
    -----------
    test_sessions : list
        A list of dictionaries, where each dictionary contains processed data 
        for a single test session, including 'trial_results', 'session_day', 
        and optionally 'session_info' with 'stress_status'.
    animal_id : str
        Identifier for the animal.
    output_dir : str
        Directory to save the output figure.
        
    Returns:
    --------
    str or None
        The absolute path to the saved figure file, or None if saving failed or no data.
    """
    if not test_sessions:
        print("No test sessions provided for cross-session summary.")
        return None

    # Sort sessions by day to ensure chronological plotting
    test_sessions.sort(key=lambda x: x['session_day'])
    session_days = [s['session_day'] for s in test_sessions]

    # Define reward size labels and colormap (consistent with other plots)
    size_labels = ['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
    reward_sizes = range(1, 6)
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=1, vmax=len(reward_sizes))

    # --- MODIFICATION START: Find first stress day and position --- 
    first_stress_day = None
    stress_boundary_pos = None
    for session in test_sessions:
        if session.get('session_info', {}).get('stress_status', False):
            first_stress_day = session['session_day']
            break # Found the first stress session
            
    if first_stress_day is not None:
        # Place the line just before the first stress day
        stress_boundary_pos = first_stress_day - 0.5
    # --- MODIFICATION END ---

    # --- Data Aggregation: Store INDIVIDUAL points per session/day --- 
    # plot_data[prev_reward_size][session_day] = {'latencies': [], 'durations': []}
    plot_data = {size: {day: {'latencies': [], 'durations': []} for day in session_days} 
                 for size in reward_sizes}
    all_latencies_overall = [] # For calculating global y-lim
    all_durations_overall = [] # For calculating global y-lim

    for session in test_sessions:
        session_day = session['session_day']
        trial_results = session.get('trial_results', [])
        if not trial_results:
            continue
        
        # Group data by previous reward size for this session
        for i in range(1, len(trial_results)):
            prev_trial = trial_results[i-1]
            current_trial = trial_results[i]

            # Check if previous trial was rewarded and get its size
            if prev_trial.get('rewarded', False) and 'reward_size' in prev_trial:
                prev_size = prev_trial['reward_size']
                if prev_size in reward_sizes:
                    # Get current trial's latency (handle missing/None)
                    latency = current_trial.get('first_lick_latency')
                    if latency is not None and not np.isnan(latency):
                         plot_data[prev_size][session_day]['latencies'].append(latency)
                         all_latencies_overall.append(latency)
                    
                    # Get current trial's lick cut duration
                    cut_duration = None
                    if 'lick_sensor_cut_analysis' in session and 'trial_metrics' in session['lick_sensor_cut_analysis']:
                        metrics = session['lick_sensor_cut_analysis']['trial_metrics']
                        current_trial_num = current_trial.get('trial_num')
                        metric = next((m for m in metrics if m.get('trial_num') == current_trial_num), None)
                        if metric and 'post_cue_cut_duration' in metric:
                            cut_duration = metric['post_cue_cut_duration']
                    elif 'post_cue_cut_duration' in current_trial:
                        cut_duration = current_trial['post_cue_cut_duration']

                    if cut_duration is not None and not np.isnan(cut_duration):
                        plot_data[prev_size][session_day]['durations'].append(cut_duration)
                        all_durations_overall.append(cut_duration)

    # --- Plotting ---
    fig, axes = plt.subplots(5, 2, figsize=(18, 22), sharex=True) 
    fig.suptitle(f'{animal_id} - Behavior Summary by Previous Reward Size (Test Sessions)', fontsize=16, y=0.98)

    # --- MODIFICATION START: Calculate shared Y-limits --- 
    lat_min, lat_max = (np.min(all_latencies_overall), np.max(all_latencies_overall)) if all_latencies_overall else (0, 1.5)
    dur_min, dur_max = (np.min(all_durations_overall), np.max(all_durations_overall)) if all_durations_overall else (0, 1.5)
    
    # Add padding to limits
    lat_padding = (lat_max - lat_min) * 0.1 if (lat_max - lat_min) > 0 else 0.1
    dur_padding = (dur_max - dur_min) * 0.1 if (dur_max - dur_min) > 0 else 0.1
    lat_ylim = (max(0, lat_min - lat_padding), lat_max + lat_padding)
    dur_ylim = (max(0, dur_min - dur_padding), min(1.5, dur_max + dur_padding)) # Cap duration at 1.5s max window
    # --- MODIFICATION END ---

    for i, size in enumerate(reward_sizes):
        row_index = i
        color = cmap(norm(size))
        label = size_labels[i]

        ax_lat = axes[row_index, 0]
        ax_dur = axes[row_index, 1]

        # --- MODIFICATION START: Plot individual points and mean/sem per day --- 
        for day_idx, day in enumerate(session_days):
            # Get data for this specific day and previous reward size
            day_latencies = plot_data[size][day]['latencies']
            day_durations = plot_data[size][day]['durations']

            # Latency Plot (Left Column)
            if day_latencies:
                # Individual points with jitter
                jitter_lat = np.random.normal(0, 0.08, size=len(day_latencies))
                ax_lat.scatter(np.array([day] * len(day_latencies)) + jitter_lat, day_latencies, 
                           # --- MODIFICATION: Increase size and alpha slightly --- 
                           alpha=0.4, s=45, color=color, edgecolors='none') 
                           # --------------------------------------------------
                # Mean +/- SEM marker
                mean_lat = np.mean(day_latencies)
                sem_lat = sem(day_latencies) if len(day_latencies) > 1 else 0
                ax_lat.errorbar(day, mean_lat, yerr=sem_lat, fmt='o', color='black', # Mean marker is black
                                markersize=6, capsize=4, elinewidth=1.5, markeredgecolor='white')
            else:
                # Optionally plot a marker indicating no data for this day
                ax_lat.plot(day, lat_ylim[0] - lat_padding*0.5, 'x', color='lightgray', markersize=4) # Small x at bottom

            # Duration Plot (Right Column)
            if day_durations:
                # Individual points with jitter
                jitter_dur = np.random.normal(0, 0.08, size=len(day_durations))
                ax_dur.scatter(np.array([day] * len(day_durations)) + jitter_dur, day_durations, 
                           # --- MODIFICATION: Increase size and alpha slightly --- 
                           alpha=0.4, s=45, color=color, edgecolors='none')
                           # --------------------------------------------------
                # Mean +/- SEM marker
                mean_dur = np.mean(day_durations)
                sem_dur = sem(day_durations) if len(day_durations) > 1 else 0
                ax_dur.errorbar(day, mean_dur, yerr=sem_dur, fmt='o', color='black', # Mean marker is black
                                markersize=6, capsize=4, elinewidth=1.5, markeredgecolor='white')
            else:
                # Optionally plot a marker indicating no data for this day
                ax_dur.plot(day, dur_ylim[0] - dur_padding*0.5, 'x', color='lightgray', markersize=4) # Small x at bottom
        # --- MODIFICATION END --- 

        ax_lat.set_title(f"Latency after '{label}' Reward")
        ax_lat.set_ylabel("First Lick Latency (s)")
        ax_lat.grid(True, alpha=0.5)
        # --- MODIFICATION: Set fixed y-limit for latency --- 
        # ax_lat.set_ylim(lat_ylim) # Apply shared y-limit
        ax_lat.set_ylim(0, 3)
        # ---------------------------------------------------

        ax_dur.set_title(f"Cut Duration after '{label}' Reward")
        ax_dur.set_ylabel("Lick Cut Duration (s)")
        ax_dur.grid(True, alpha=0.5)
        ax_dur.set_ylim(dur_ylim) # Apply shared y-limit
        ax_dur.axhline(y=1.5, color='red', linestyle=':', alpha=0.5, linewidth=1) # Add max duration line
        
        # --- MODIFICATION START: Add stress boundary line --- 
        if stress_boundary_pos is not None:
            ax_lat.axvline(x=stress_boundary_pos, color='blue', linestyle='--', alpha=0.6, linewidth=1.5)
            ax_dur.axvline(x=stress_boundary_pos, color='blue', linestyle='--', alpha=0.6, linewidth=1.5)
        # --- MODIFICATION END ---

        # Set x-axis label only for the bottom row
        if row_index == len(reward_sizes) - 1:
            ax_lat.set_xlabel("Session Day")
            ax_dur.set_xlabel("Session Day")
        
        # Set x-ticks for all subplots (since sharex=True, only need once technically)
        ax_lat.set_xticks(session_days)
        ax_dur.set_xticks(session_days)

    # Improve layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust rect for suptitle

    # Save the figure
    try:
        filename = f"{animal_id}_TEST_prev_reward_summary.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved cross-session summary figure: {filepath}")
        plt.close(fig)
        return filepath
    except Exception as e:
        print(f"Error saving cross-session summary figure: {e}")
        plt.close(fig)
        return None

# <<< NEW FUNCTION START >>>
def visualize_time_resolved_licking(session_data, animal_id, session_label, output_dir):
    """Generates a two-panel figure showing time-resolved licking grouped by
       current and previous reward size.
    """
    print(f"  Generating time-resolved licking plot for {session_label}...")
    start_func_time = time.time()

    # --- Analysis Parameters --- (Defined within function for encapsulation)
    ANALYSIS_START_TIME_REL_REWARD = -2.0
    TOTAL_DURATION_POST_REWARD = 8.0 # Keep duration reasonable for session plots
    ANALYSIS_END_TIME_REL_REWARD = TOTAL_DURATION_POST_REWARD
    WINDOW_SIZE = 0.1  # 100 ms window
    STEP_SIZE = 0.02   # 20 ms step
    TIME_RESOLUTION = 0.001 # 1 ms internal resolution
    size_labels_map_ts = {1: 'Very Small', 2: 'Small', 3: 'Medium', 4: 'Large', 5: 'Very Large'}

    # --- Extract required data from session_data --- 
    monitoring_trials = session_data.get('trial_results', [])
    if not monitoring_trials:
        print("  ERROR: No trial results found in session_data. Cannot generate time-resolved plot.")
        return None

    # <<< DEBUG: Print first few trial results >>>
    # print(f"  DEBUG: Total monitoring_trials found: {len(monitoring_trials)}")
    # if monitoring_trials:
    #     print(f"  DEBUG: First trial data sample: {monitoring_trials[0]}")
    # <<< END DEBUG >>>

    # Reconstruct reward_sizes_map (trial_num -> reward_size)
    # We need this mapping because the analysis logic uses it.
    reward_sizes_map = {}
    for trial in monitoring_trials:
        t_num = trial.get('trial_num')
        r_size = trial.get('reward_size') # Reward size should be here after analyze_test_session
        if t_num is not None and r_size is not None:
            reward_sizes_map[int(t_num)] = int(r_size)
            
    if not reward_sizes_map:
        print("  ERROR: Could not reconstruct reward sizes map. Cannot generate time-resolved plot.")
        return None
    
    # <<< DEBUG: Print reconstructed map sample >>>
    # if reward_sizes_map:
    #     print(f"  DEBUG: Sample reward_sizes_map: {list(reward_sizes_map.items())[:5]}")
    # <<< END DEBUG >>>

    # --- Shared Calculation Setup --- 
    time_points_relative_to_reward = np.arange(ANALYSIS_START_TIME_REL_REWARD + STEP_SIZE,
                                               ANALYSIS_END_TIME_REL_REWARD + STEP_SIZE,
                                               STEP_SIZE)
    # Ensure the last point doesn't exceed the end time due to float precision
    time_points_relative_to_reward = time_points_relative_to_reward[time_points_relative_to_reward <= ANALYSIS_END_TIME_REL_REWARD + 1e-9]
    num_target_time_points = len(time_points_relative_to_reward)
    hires_trace_duration_rel_reward = ANALYSIS_END_TIME_REL_REWARD - ANALYSIS_START_TIME_REL_REWARD
    num_hires_points_total_analysis = int(np.round(hires_trace_duration_rel_reward / TIME_RESOLUTION))
    # Calculate target indices carefully
    target_indices_in_hires = np.round((time_points_relative_to_reward - ANALYSIS_START_TIME_REL_REWARD) / TIME_RESOLUTION).astype(int)
    # Ensure indices are within bounds [0, num_hires_points_total_analysis - 1]
    target_indices_in_hires = np.clip(target_indices_in_hires -1, 0, num_hires_points_total_analysis - 1) 

    win_samples = int(np.round(WINDOW_SIZE / TIME_RESOLUTION))
    win_samples = max(1, win_samples) # Ensure window size is at least 1 sample
    
    # Helper function for calculating the time series for one trial
    def calculate_single_trial_ts(mtrial, reward_latency):
        lick_downs = mtrial.get('lick_downs_relative', [])
        lick_ups = mtrial.get('lick_ups_relative', [])
        
        # --- Generate High-Resolution Binary Lick Trace --- 
        analysis_start_rel_trial = reward_latency + ANALYSIS_START_TIME_REL_REWARD
        analysis_end_rel_trial = reward_latency + ANALYSIS_END_TIME_REL_REWARD
        max_time_rel_trial_needed = analysis_end_rel_trial + TIME_RESOLUTION # Include endpoint
        num_hires_points_trial = int(np.ceil(max_time_rel_trial_needed / TIME_RESOLUTION))
        lick_trace_trial = np.zeros(num_hires_points_trial, dtype=np.int8)
        lick_downs_sorted = sorted([ld for ld in lick_downs if ld is not None])
        lick_ups_sorted = sorted([lu for lu in lick_ups if lu is not None])
        num_bouts = min(len(lick_downs_sorted), len(lick_ups_sorted))
        for i in range(num_bouts):
            start_idx = int(np.floor((lick_downs_sorted[i] + 1e-9) / TIME_RESOLUTION)) # Add epsilon for floor
            end_idx = int(np.ceil(lick_ups_sorted[i] / TIME_RESOLUTION))
            start_idx = max(0, start_idx)
            end_idx = min(num_hires_points_trial, end_idx)
            if start_idx < end_idx: 
                lick_trace_trial[start_idx:end_idx] = 1
        
        # --- Extract Analysis Window Trace --- 
        idx_analysis_start_in_trial_trace = int(np.round(analysis_start_rel_trial / TIME_RESOLUTION))
        idx_analysis_end_in_trial_trace = idx_analysis_start_in_trial_trace + num_hires_points_total_analysis
        actual_start_idx = max(0, idx_analysis_start_in_trial_trace)
        actual_end_idx = min(len(lick_trace_trial), idx_analysis_end_in_trial_trace)
        extracted_trace = lick_trace_trial[actual_start_idx:actual_end_idx]
        pad_before = max(0, -idx_analysis_start_in_trial_trace) # Pad if analysis starts before trial trace
        pad_after = max(0, num_hires_points_total_analysis - (len(extracted_trace) + pad_before))
        analysis_trace_hires = np.pad(extracted_trace, (pad_before, pad_after), 'constant')
        
        if len(analysis_trace_hires) != num_hires_points_total_analysis:
             # print(f"Warning: Trial {mtrial.get('trial_num')}: Length mismatch in hires trace ({len(analysis_trace_hires)} vs {num_hires_points_total_analysis}). Returning NaNs.")
             return np.full(num_target_time_points, np.nan)
             
        # --- Calculate Sliding Window Proportion --- 
        # Use pandas rolling mean for efficiency and handling edges
        rolling_proportions = pd.Series(analysis_trace_hires).rolling(window=win_samples, min_periods=1, center=True).mean().to_numpy()
        
        # --- Sample Proportions at Target Time Points --- 
        # Ensure indices are valid for the rolling_proportions array
        current_target_indices = target_indices_in_hires[target_indices_in_hires < len(rolling_proportions)]
        sampled_proportions = np.full(num_target_time_points, np.nan)
        valid_indices_mask = target_indices_in_hires < len(rolling_proportions)
        valid_target_indices_for_sampling = target_indices_in_hires[valid_indices_mask]
        sampled_proportions[valid_indices_mask] = rolling_proportions[valid_target_indices_for_sampling]

        return sampled_proportions

    # --- Part A: Analysis Grouped by CURRENT Reward Size --- 
    licking_ts_by_reward_size = {}
    processed_trials_curr = 0
    first_trial_ts_output_curr = None # <<< DEBUG VARIABLE >>>
    for trial_idx, mtrial in enumerate(monitoring_trials):
        trial_num = mtrial.get('trial_num')
        is_rewarded = mtrial.get('rewarded', False)
        reward_latency = mtrial.get('reward_latency')
        if trial_num is None or not is_rewarded or reward_latency is None: continue
        reward_size = reward_sizes_map.get(int(trial_num))
        if reward_size is None: continue
        
        try:
            sampled_proportions = calculate_single_trial_ts(mtrial, reward_latency)
            # <<< DEBUG: Check output of calculate_single_trial_ts for first processed trial >>>
            # if processed_trials_curr == 0 and sampled_proportions is not None:
            #     first_trial_ts_output_curr = sampled_proportions
            #     print(f"  DEBUG (Current Size): First calculated TS (Trial {trial_num}) - First 5 points: {first_trial_ts_output_curr[:5]}")
            #     print(f"  DEBUG (Current Size): First calculated TS (Trial {trial_num}) - Contains NaNs: {np.isnan(first_trial_ts_output_curr).any()}")
            # <<< END DEBUG >>>
            
            current_rew_size_key = int(reward_size)
            if current_rew_size_key not in licking_ts_by_reward_size: licking_ts_by_reward_size[current_rew_size_key] = []
            licking_ts_by_reward_size[current_rew_size_key].append(sampled_proportions)
            processed_trials_curr += 1
        except Exception as e:
            print(f"  Warning: Error processing trial {trial_num} (Current Size Grouping): {e}")
            
    # --- Part B: Analysis Grouped by PREVIOUS Reward Size --- 
    licking_ts_by_prev_reward_size = {}
    processed_trials_prev = 0
    if len(monitoring_trials) >= 2:
        for idx in range(1, len(monitoring_trials)):
            mtrial_curr = monitoring_trials[idx]
            mtrial_prev = monitoring_trials[idx-1]
            trial_num_curr = mtrial_curr.get('trial_num')
            is_rewarded_curr = mtrial_curr.get('rewarded', False)
            reward_latency_curr = mtrial_curr.get('reward_latency')
            if trial_num_curr is None or not is_rewarded_curr or reward_latency_curr is None: continue

            is_rewarded_prev = mtrial_prev.get('rewarded', False)
            trial_num_prev = mtrial_prev.get('trial_num')
            if not is_rewarded_prev or trial_num_prev is None: continue
            reward_size_prev = reward_sizes_map.get(int(trial_num_prev))
            if reward_size_prev is None: continue

            try:
                sampled_proportions = calculate_single_trial_ts(mtrial_curr, reward_latency_curr)
                prev_rew_size_key = int(reward_size_prev)
                if prev_rew_size_key not in licking_ts_by_prev_reward_size: licking_ts_by_prev_reward_size[prev_rew_size_key] = []
                licking_ts_by_prev_reward_size[prev_rew_size_key].append(sampled_proportions)
                processed_trials_prev += 1
            except Exception as e:
                 print(f"  Warning: Error processing trial {trial_num_curr} (Prev Size Grouping): {e}")
                 
    # --- Aggregate Results --- 
    def aggregate_ts_data(licking_ts_dict):
        mean_ts_agg, sem_ts_agg, counts_agg = {}, {}, {}
        reward_sizes_present = sorted(licking_ts_dict.keys())
        for r_size in reward_sizes_present:
            valid_trials_ts = [ts for ts in licking_ts_dict[r_size] if ts is not None and not np.all(np.isnan(ts))]
            n_trials = len(valid_trials_ts)
            counts_agg[r_size] = n_trials
            # <<< DEBUG: Print aggregation counts >>>
            # print(f"  DEBUG (Aggregate): Size {r_size} - Found {len(licking_ts_dict.get(r_size, []))} raw trials, {n_trials} valid (non-NaN) trials.")
            # <<< END DEBUG >>>
            if n_trials > 0:
                # Ensure all arrays have the same length before stacking
                expected_len = num_target_time_points
                valid_trials_ts = [ts for ts in valid_trials_ts if len(ts) == expected_len]
                n_trials = len(valid_trials_ts) # Update count after filtering
                counts_agg[r_size] = n_trials
                if n_trials > 0:
                    trial_ts_stack = np.array(valid_trials_ts)
                    mean_ts = np.nanmean(trial_ts_stack, axis=0)
                    if n_trials > 1:
                         with np.errstate(invalid='ignore'): # Ignore warnings from all-NaN slices
                             sem_val = sem(trial_ts_stack, axis=0, nan_policy='omit')
                         sem_val = np.nan_to_num(sem_val, nan=0.0) # Replace NaN SEM with 0
                    else: 
                        sem_val = np.zeros_like(mean_ts)
                    mean_ts_agg[r_size] = mean_ts
                    sem_ts_agg[r_size] = sem_val
        return mean_ts_agg, sem_ts_agg, counts_agg

    mean_licking_ts_curr, sem_licking_ts_curr, trial_counts_agg_curr = aggregate_ts_data(licking_ts_by_reward_size)
    mean_licking_ts_prev, sem_licking_ts_prev, trial_counts_agg_prev = aggregate_ts_data(licking_ts_by_prev_reward_size)

    # --- Plotting --- 
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True) # Two panels sharing x-axis
    cmap = plt.cm.viridis 

    # Panel 0: Grouped by Current Reward Size
    ax = axs[0]
    all_plot_sizes_curr = sorted(mean_licking_ts_curr.keys())
    if not all_plot_sizes_curr:
        ax.text(0.5, 0.5, 'No data (Current Size)', ha='center', va='center', transform=ax.transAxes)
    else:
        min_rs_curr = min(all_plot_sizes_curr); max_rs_curr = max(all_plot_sizes_curr)
        norm_ts_curr = mcolors.Normalize(vmin=min_rs_curr, vmax=max_rs_curr) if max_rs_curr > min_rs_curr else mcolors.Normalize(vmin=min_rs_curr-1, vmax=max_rs_curr+1)
        for r_size in all_plot_sizes_curr:
            mean_ts = mean_licking_ts_curr.get(r_size)
            sem_ts = sem_licking_ts_curr.get(r_size)
            if mean_ts is not None and sem_ts is not None and len(time_points_relative_to_reward) == len(mean_ts):
                color = cmap(norm_ts_curr(r_size))
                n_val = trial_counts_agg_curr.get(r_size, 0)
                label = f"Size {r_size} ({size_labels_map_ts.get(r_size, '')}, N={n_val})"
                ax.plot(time_points_relative_to_reward, mean_ts, color=color, label=label, linewidth=2)
                ax.fill_between(time_points_relative_to_reward, mean_ts - sem_ts, mean_ts + sem_ts, color=color, alpha=0.2)
            else:
                print(f"  Warning: Plotting issue for Current Size {r_size}. Mean/SEM None or length mismatch.")
        ax.set_ylabel("Mean Lick Proportion")
        ax.set_title(f"Time-Resolved Licking by CURRENT Reward Size")
        ax.legend(title="Current Reward Size", loc='upper right')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Reward Delivery')
        # Ensure legend includes reward line
        handles, labels = ax.get_legend_handles_labels()
        if not any(l=='Reward Delivery' for l in labels):
            line_handle = next((line for line in ax.get_lines() if line.get_color()=='red'), None)
            if line_handle: handles.append(line_handle); labels.append('Reward Delivery')
        ax.legend(handles=handles, labels=labels, title="Current Reward Size", loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.5)
        max_y_curr = max([np.nanmax(m + s) for m, s in zip(mean_licking_ts_curr.values(), sem_licking_ts_curr.values()) if m is not None and s is not None] + [0])
        ax.set_ylim(bottom=-0.05, top=max(1.05, max_y_curr * 1.1))

    # Panel 1: Grouped by Previous Reward Size
    ax = axs[1]
    all_plot_sizes_prev = sorted(mean_licking_ts_prev.keys())
    if not all_plot_sizes_prev:
        ax.text(0.5, 0.5, 'No data (Previous Size)', ha='center', va='center', transform=ax.transAxes)
    else:
        min_rs_prev = min(all_plot_sizes_prev); max_rs_prev = max(all_plot_sizes_prev)
        norm_ts_prev = mcolors.Normalize(vmin=min_rs_prev, vmax=max_rs_prev) if max_rs_prev > min_rs_prev else mcolors.Normalize(vmin=min_rs_prev-1, vmax=max_rs_prev+1)
        for prev_r_size in all_plot_sizes_prev:
            mean_ts = mean_licking_ts_prev.get(prev_r_size)
            sem_ts = sem_licking_ts_prev.get(prev_r_size)
            if mean_ts is not None and sem_ts is not None and len(time_points_relative_to_reward) == len(mean_ts):
                color = cmap(norm_ts_prev(prev_r_size))
                n_val = trial_counts_agg_prev.get(prev_r_size, 0)
                label = f"Prev Size {prev_r_size} ({size_labels_map_ts.get(prev_r_size, '')}, N={n_val})"
                ax.plot(time_points_relative_to_reward, mean_ts, color=color, label=label, linewidth=2)
                ax.fill_between(time_points_relative_to_reward, mean_ts - sem_ts, mean_ts + sem_ts, color=color, alpha=0.2)
            else:
                 print(f"  Warning: Plotting issue for Previous Size {prev_r_size}. Mean/SEM None or length mismatch.")
        ax.set_xlabel("Time Relative to Current Reward Delivery (s)")
        ax.set_ylabel("Mean Lick Proportion")
        ax.set_title(f"Time-Resolved Licking by PREVIOUS Reward Size")
        ax.legend(title="Previous Reward Size", loc='upper right')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Current Reward Delivery')
        # Ensure legend includes reward line
        handles, labels = ax.get_legend_handles_labels()
        if not any(l=='Current Reward Delivery' for l in labels):
            line_handle = next((line for line in ax.get_lines() if line.get_color()=='red'), None)
            if line_handle: handles.append(line_handle); labels.append('Current Reward Delivery')
        ax.legend(handles=handles, labels=labels, title="Previous Reward Size", loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.5)
        ax.set_xlim(ANALYSIS_START_TIME_REL_REWARD, ANALYSIS_END_TIME_REL_REWARD) # Ensure x-lim is set
        max_y_prev = max([np.nanmax(m + s) for m, s in zip(mean_licking_ts_prev.values(), sem_licking_ts_prev.values()) if m is not None and s is not None] + [0])
        ax.set_ylim(bottom=-0.05, top=max(1.05, max_y_prev * 1.1))

    # Overall Figure Title & Layout
    fig.suptitle(f"{animal_id} - {session_label}: Time-Resolved Licking Analysis\n(Window: {WINDOW_SIZE*1000:.0f}ms, Step: {STEP_SIZE*1000:.0f}ms, Resolution: {TIME_RESOLUTION*1000:.0f}ms)", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # --- Save Figure --- 
    output_filename = os.path.join(output_dir, f"{animal_id}_{session_label}_time_resolved_licking.png")
    try:
        plt.savefig(output_filename)
        print(f"  Saved time-resolved licking figure: {output_filename}")
    except Exception as e:
        print(f"  ERROR saving time-resolved licking figure: {e}")
        output_filename = None # Indicate failure
    finally:
        plt.close(fig) # Ensure figure is closed
        
    print(f"  Finished time-resolved licking plot generation ({time.time() - start_func_time:.2f} seconds).")
    return output_filename
# <<< NEW FUNCTION END >>>

# <<< NEW FUNCTION START: Licking by Previous Extremes >>>
def visualize_licking_by_prev_extremes(session_data, animal_id, session_label, output_dir):
    """Generates a 5-panel figure showing time-resolved licking for each current reward size,
       grouped by whether the previous reward was Very Small/Small vs Large/Very Large.
    """
    print(f"  Generating licking by prev extremes plot for {session_label}...")
    start_func_time = time.time()

    # --- Analysis Parameters --- 
    ANALYSIS_START_TIME_REL_REWARD = -2.0 # <<< CHANGED FROM -1.0 >>>
    TOTAL_DURATION_POST_REWARD = 8.0
    ANALYSIS_END_TIME_REL_REWARD = TOTAL_DURATION_POST_REWARD
    WINDOW_SIZE = 0.1
    STEP_SIZE = 0.02
    TIME_RESOLUTION = 0.001
    ALL_REWARD_SIZES = [1, 2, 3, 4, 5]
    size_labels_map = {1: 'VS', 2: 'S', 3: 'M', 4: 'L', 5: 'VL'} # Shorter labels for title
    PREV_VS_S_SIZES = [1, 2]
    PREV_L_VL_SIZES = [4, 5]
    GROUP1_LABEL = "Prev VS/S"
    GROUP2_LABEL = "Prev L/VL"
    GROUP_COLORS = {GROUP1_LABEL: 'blue', GROUP2_LABEL: 'orange'}

    # --- Extract required data from session_data --- 
    monitoring_trials = session_data.get('trial_results', [])
    if not monitoring_trials or len(monitoring_trials) < 2:
        print("  ERROR: Not enough trial results (<2) found. Cannot generate plot.")
        return None

    # Reconstruct reward_sizes_map (trial_num -> reward_size)
    reward_sizes_map = {}
    for trial in monitoring_trials:
        t_num = trial.get('trial_num')
        r_size = trial.get('reward_size')
        if t_num is not None and r_size is not None:
            reward_sizes_map[int(t_num)] = int(r_size)
            
    if not reward_sizes_map:
        print("  ERROR: Could not reconstruct reward sizes map. Cannot generate plot.")
        return None

    # --- Shared Calculation Setup --- 
    time_points_relative_to_reward = np.arange(ANALYSIS_START_TIME_REL_REWARD + STEP_SIZE,
                                               ANALYSIS_END_TIME_REL_REWARD + STEP_SIZE,
                                               STEP_SIZE)
    time_points_relative_to_reward = time_points_relative_to_reward[time_points_relative_to_reward <= ANALYSIS_END_TIME_REL_REWARD + 1e-9]
    num_target_time_points = len(time_points_relative_to_reward)
    hires_trace_duration_rel_reward = ANALYSIS_END_TIME_REL_REWARD - ANALYSIS_START_TIME_REL_REWARD
    num_hires_points_total_analysis = int(np.round(hires_trace_duration_rel_reward / TIME_RESOLUTION))
    target_indices_in_hires = np.round((time_points_relative_to_reward - ANALYSIS_START_TIME_REL_REWARD) / TIME_RESOLUTION).astype(int) - 1
    target_indices_in_hires = np.clip(target_indices_in_hires, 0, num_hires_points_total_analysis - 1)
    win_samples = int(np.round(WINDOW_SIZE / TIME_RESOLUTION))
    win_samples = max(1, win_samples)
    
    # Re-use the helper function from visualize_time_resolved_licking if available,
    # otherwise define it locally (simplified version)
    try:
        # Check if the helper exists (it should if the previous function was added)
        from .visualization import calculate_single_trial_ts # Use relative import if possible
    except ImportError:
        # Define locally if import fails (copy simplified logic)
        def calculate_single_trial_ts(mtrial, reward_latency):
            # ... (Simplified or copied calculation logic here - see previous function) ...
            # For brevity, assume the function is available from the previous step.
            # If running this standalone, the full helper function needs to be copied here.
            # Placeholder return:
            # return np.full(num_target_time_points, np.nan)
            lick_downs = mtrial.get('lick_downs_relative', [])
            lick_ups = mtrial.get('lick_ups_relative', [])
            analysis_start_rel_trial = reward_latency + ANALYSIS_START_TIME_REL_REWARD
            analysis_end_rel_trial = reward_latency + ANALYSIS_END_TIME_REL_REWARD
            max_time_rel_trial_needed = analysis_end_rel_trial + TIME_RESOLUTION
            num_hires_points_trial = int(np.ceil(max_time_rel_trial_needed / TIME_RESOLUTION))
            lick_trace_trial = np.zeros(num_hires_points_trial, dtype=np.int8)
            lick_downs_sorted = sorted([ld for ld in lick_downs if ld is not None])
            lick_ups_sorted = sorted([lu for lu in lick_ups if lu is not None])
            num_bouts = min(len(lick_downs_sorted), len(lick_ups_sorted))
            for i in range(num_bouts):
                start_idx = int(np.floor((lick_downs_sorted[i] + 1e-9) / TIME_RESOLUTION))
                end_idx = int(np.ceil(lick_ups_sorted[i] / TIME_RESOLUTION))
                start_idx = max(0, start_idx); end_idx = min(num_hires_points_trial, end_idx)
                if start_idx < end_idx: lick_trace_trial[start_idx:end_idx] = 1
            idx_analysis_start_in_trial_trace = int(np.round(analysis_start_rel_trial / TIME_RESOLUTION))
            idx_analysis_end_in_trial_trace = idx_analysis_start_in_trial_trace + num_hires_points_total_analysis
            actual_start_idx = max(0, idx_analysis_start_in_trial_trace)
            actual_end_idx = min(len(lick_trace_trial), idx_analysis_end_in_trial_trace)
            extracted_trace = lick_trace_trial[actual_start_idx:actual_end_idx]
            pad_before = max(0, -idx_analysis_start_in_trial_trace)
            pad_after = max(0, num_hires_points_total_analysis - (len(extracted_trace) + pad_before))
            analysis_trace_hires = np.pad(extracted_trace, (pad_before, pad_after), 'constant')
            if len(analysis_trace_hires) != num_hires_points_total_analysis:
                return np.full(num_target_time_points, np.nan)
            rolling_proportions = pd.Series(analysis_trace_hires).rolling(window=win_samples, min_periods=1, center=True).mean().to_numpy()
            current_target_indices = target_indices_in_hires[target_indices_in_hires < len(rolling_proportions)]
            sampled_proportions = np.full(num_target_time_points, np.nan)
            valid_indices_mask = target_indices_in_hires < len(rolling_proportions)
            valid_target_indices_for_sampling = target_indices_in_hires[valid_indices_mask]
            sampled_proportions[valid_indices_mask] = rolling_proportions[valid_target_indices_for_sampling]
            return sampled_proportions

    # --- Data Collection Loop --- 
    # Structure: collected_data[current_size][group_label] = [ts1, ts2, ...]
    collected_data = {curr_size: {GROUP1_LABEL: [], GROUP2_LABEL: []} for curr_size in ALL_REWARD_SIZES}
    
    for idx in range(1, len(monitoring_trials)):
        mtrial_curr = monitoring_trials[idx]
        mtrial_prev = monitoring_trials[idx-1]
        trial_num_curr = mtrial_curr.get('trial_num')
        try:
            reward_size_curr = reward_sizes_map.get(int(trial_num_curr))
            if reward_size_curr not in ALL_REWARD_SIZES: continue
            is_rewarded_curr = mtrial_curr.get('rewarded', False)
            reward_latency_curr = mtrial_curr.get('reward_latency')
            if trial_num_curr is None or not is_rewarded_curr or reward_latency_curr is None: continue
            is_rewarded_prev = mtrial_prev.get('rewarded', False)
            trial_num_prev = mtrial_prev.get('trial_num')
            if not is_rewarded_prev or trial_num_prev is None: continue
            reward_size_prev = reward_sizes_map.get(int(trial_num_prev))
            if reward_size_prev is None: continue

            group_key = None
            if reward_size_prev in PREV_VS_S_SIZES: group_key = GROUP1_LABEL
            elif reward_size_prev in PREV_L_VL_SIZES: group_key = GROUP2_LABEL
            else: continue

            sampled_proportions = calculate_single_trial_ts(mtrial_curr, reward_latency_curr)
            if sampled_proportions is not None and not np.all(np.isnan(sampled_proportions)):
                 collected_data[reward_size_curr][group_key].append(sampled_proportions)
                 
        except Exception as e:
            # print(f"  Warning: Error processing trial {trial_num_curr} for prev extremes plot: {e}")
            pass # Continue processing other trials

    # --- Aggregation --- 
    # aggregated_data[current_size][group_label] = {'mean': mean_ts, 'sem': sem_ts, 'n': n_trials}
    aggregated_data = {curr_size: {} for curr_size in ALL_REWARD_SIZES}
    
    for curr_size in ALL_REWARD_SIZES:
        for group_label in [GROUP1_LABEL, GROUP2_LABEL]:
            trial_list = collected_data[curr_size][group_label]
            n_trials = len(trial_list)
            mean_ts, sem_ts = None, None
            if n_trials > 0:
                trial_ts_stack = np.array(trial_list)
                mean_ts = np.nanmean(trial_ts_stack, axis=0)
                if n_trials > 1:
                     with np.errstate(invalid='ignore'): sem_ts = sem(trial_ts_stack, axis=0, nan_policy='omit')
                     sem_ts = np.nan_to_num(sem_ts, nan=0.0)
                else: sem_ts = np.zeros_like(mean_ts)
            aggregated_data[curr_size][group_label] = {'mean': mean_ts, 'sem': sem_ts, 'n': n_trials}

    # --- Plotting --- 
    fig, axs = plt.subplots(len(ALL_REWARD_SIZES), 1, figsize=(8, 12), sharex=True, sharey=True)
    if len(ALL_REWARD_SIZES) == 1: axs = [axs] # Make iterable
    fig.suptitle(f"{animal_id} - {session_label}\nLicking Response Grouped by Previous Reward Extremes", fontsize=14, y=0.99)

    max_y_overall = 0 # For shared Y limit
    plot_generated = False
    
    for i, curr_size in enumerate(ALL_REWARD_SIZES):
        ax = axs[i]
        has_data_for_panel = False
        # Get N values for title
        n_group1 = aggregated_data[curr_size].get(GROUP1_LABEL, {}).get('n', 0)
        n_group2 = aggregated_data[curr_size].get(GROUP2_LABEL, {}).get('n', 0)

        for group_label, group_data in aggregated_data[curr_size].items():
            mean_ts = group_data['mean']
            sem_ts = group_data['sem']
            n_val = group_data['n']
            if mean_ts is not None and sem_ts is not None and n_val > 0:
                has_data_for_panel = True
                plot_generated = True
                color = GROUP_COLORS[group_label]
                label = f"{group_label} (N={n_val})"
                if len(time_points_relative_to_reward) == len(mean_ts):
                     ax.plot(time_points_relative_to_reward, mean_ts, color=color, label=label, linewidth=1.5)
                     ax.fill_between(time_points_relative_to_reward, mean_ts - sem_ts, mean_ts + sem_ts, color=color, alpha=0.2)
                     max_y_overall = max(max_y_overall, np.nanmax(mean_ts + sem_ts))
                else: print(f"  Warning: Length mismatch plotting Curr={curr_size}, Group='{group_label}'")
        
        ax.axvline(x=0, color='red', linestyle=':', linewidth=1, alpha=0.6)
        ax.grid(True, alpha=0.4, linestyle='--')
        # <<< UPDATED TITLE WITH N VALUES >>>
        title_text = f"Current Reward: {size_labels_map[curr_size]} ({curr_size})    (N: Prev VS/S={n_group1}, Prev L/VL={n_group2})"
        ax.set_title(title_text, fontsize=9, loc='left') # Reduced font size slightly
        if not has_data_for_panel:
             ax.text(0.5, 0.5, 'No data for this grouping', ha='center', va='center', transform=ax.transAxes, color='gray')
        if i == 0: ax.legend(loc='upper right', fontsize='small') # Legend only on top plot
        if i == len(ALL_REWARD_SIZES) - 1: ax.set_xlabel("Time Relative to Current Reward Delivery (s)")

    # Apply shared settings
    fig.text(0.02, 0.5, "Mean Proportion Time Lick Sensor Cut", va='center', rotation='vertical', fontsize=12)
    plt.setp(axs, xlim=(ANALYSIS_START_TIME_REL_REWARD, ANALYSIS_END_TIME_REL_REWARD)) 
    # plt.setp(axs, ylim=(-0.05, max(1.05, max_y_overall * 1.1))) # Set shared Y limit
    plt.tight_layout(rect=[0.05, 0.03, 1, 0.96]) # Adjust layout for titles/labels

    # --- Save Figure --- 
    if not plot_generated:
         print("  No data generated for any panel. Skipping save.")
         plt.close(fig)
         return None
         
    output_filename = os.path.join(output_dir, f"{animal_id}_{session_label}_licking_by_prev_extremes.png")
    try:
        plt.savefig(output_filename)
        print(f"  Saved licking by prev extremes figure: {output_filename}")
    except Exception as e:
        print(f"  ERROR saving licking by prev extremes figure: {e}")
        output_filename = None
    finally:
        plt.close(fig)
        
    print(f"  Finished licking by prev extremes plot generation ({time.time() - start_func_time:.2f} seconds).")
    return output_filename
# <<< NEW FUNCTION END >>>

# <<< NEW FUNCTION START: Stress Comparison >>>
from scipy.stats import mannwhitneyu # Import statistical test
import seaborn as sns # Import seaborn for better plots

def visualize_stress_comparison_by_prev_reward(test_sessions, animal_id, output_dir):
    """Generates a 5x2 plot comparing Pre- vs Post-Stress latency and duration,
       grouped by the previous trial's reward size, using violin plots.
    """
    if not test_sessions:
        print("No test sessions provided for stress comparison.")
        return None

    print(f"\nGenerating Pre vs Post-Stress comparison plot for {animal_id}...")
    start_func_time = time.time()

    # --- Find first stress day --- 
    first_stress_day = None
    for session in test_sessions:
        if session.get('session_info', {}).get('stress_status', False):
            first_stress_day = session['session_day']
            print(f"  First stress day identified: {first_stress_day}")
            break
            
    if first_stress_day is None:
        print("  No stress sessions found for this animal. Cannot generate comparison plot.")
        return None

    # --- Analysis Parameters & Setup --- 
    ALL_REWARD_SIZES = [1, 2, 3, 4, 5]
    size_labels_map = {1: 'VS', 2: 'S', 3: 'M', 4: 'L', 5: 'VL'}
    stress_periods = ['Pre-Stress', 'Post-Stress']
    plot_palette = {'Pre-Stress': 'skyblue', 'Post-Stress': 'salmon'}

    # --- Data Pooling --- 
    # Structure: pooled_data[prev_reward_size][stress_period][metric] = [value1, value2,...]
    pooled_data = {prev_size: {'Pre-Stress': {'latencies': [], 'durations': []}, 
                                'Post-Stress': {'latencies': [], 'durations': []}} 
                   for prev_size in ALL_REWARD_SIZES}
    
    all_latencies_overall = []
    all_durations_overall = []

    # --- Convert data to DataFrame for easier plotting with Seaborn --- 
    plot_df_list = []
    for session in test_sessions:
        session_day = session['session_day']
        trial_results = session.get('trial_results', [])
        stress_period = 'Post-Stress' if session_day >= first_stress_day else 'Pre-Stress'
        
        if not trial_results or len(trial_results) < 2:
            continue
            
        for i in range(1, len(trial_results)):
            prev_trial = trial_results[i-1]
            current_trial = trial_results[i]

            if prev_trial.get('rewarded', False):
                prev_size = prev_trial.get('reward_size')
                if prev_size in ALL_REWARD_SIZES:
                    latency = current_trial.get('first_lick_latency')
                    cut_duration = None # Initialize
                    # Get duration (handle potential nested structure)
                    lick_analysis_data = session.get('lick_sensor_cut_analysis', {})
                    if 'trial_metrics' in lick_analysis_data:
                        metrics = lick_analysis_data['trial_metrics']
                        current_trial_num = current_trial.get('trial_num')
                        metric = next((m for m in metrics if m.get('trial_num') == current_trial_num), None)
                        if metric and 'post_cue_cut_duration' in metric:
                            cut_duration = metric['post_cue_cut_duration']
                    elif 'post_cue_cut_duration' in current_trial: 
                        cut_duration = current_trial['post_cue_cut_duration']
                    
                    latency_val = latency if (latency is not None and not np.isnan(latency)) else np.nan
                    duration_val = cut_duration if (cut_duration is not None and not np.isnan(cut_duration)) else np.nan
                    
                    # Add to overall lists for limit calculation BEFORE adding to dataframe
                    if not np.isnan(latency_val): all_latencies_overall.append(latency_val)
                    if not np.isnan(duration_val): all_durations_overall.append(duration_val)
                    
                    plot_df_list.append({
                        'prev_reward_size': prev_size,
                        'stress_period': stress_period,
                        'latency': latency_val,
                        'duration': duration_val
                    })
                    
    if not plot_df_list:
         print("  No valid data points found after pooling. Cannot generate plot.")
         return None
         
    plot_df = pd.DataFrame(plot_df_list)

    # --- Plotting --- 
    fig, axs = plt.subplots(len(ALL_REWARD_SIZES), 2, figsize=(9, 12), sharex='col', sharey='col') # Adjusted figsize
    if len(ALL_REWARD_SIZES) == 1: axs = np.array([axs]) # Ensure axs is 2D if only 1 row
        
    fig.suptitle(f"{animal_id} - Pre vs Post-Stress Behavior by Previous Reward Size", fontsize=14, y=0.99)

    plot_generated = False # Flag to check if any data was plotted
    min_data_points_for_test = 3 # Minimum data points needed in EACH group for stat test
    stats_results = {} # Store stats results for annotation after setting ylim

    for i, prev_size in enumerate(ALL_REWARD_SIZES):
        ax_lat = axs[i, 0]
        ax_dur = axs[i, 1]
        prev_label = size_labels_map[prev_size]
        row_title = f"Previous Reward: {prev_label} ({prev_size})"
        # <<< Set Row title on the left axis >>>
        ax_lat.set_ylabel(row_title, fontsize=9) # Use ylabel for row identification
        ax_dur.set_ylabel("duration (s)", fontsize=9) # Clear duration ylabel
        
        # <<< Remove top/right spines and increase tick size >>>
        for ax in [ax_lat, ax_dur]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', size=6, width=1.5, labelsize=9)

        # Filter DataFrame for the current previous reward size
        df_subset = plot_df[plot_df['prev_reward_size'] == prev_size]

        # Data for stats
        lat_pre = df_subset[df_subset['stress_period'] == 'Pre-Stress']['latency'].dropna().tolist()
        lat_post = df_subset[df_subset['stress_period'] == 'Post-Stress']['latency'].dropna().tolist()
        dur_pre = df_subset[df_subset['stress_period'] == 'Pre-Stress']['duration'].dropna().tolist()
        dur_post = df_subset[df_subset['stress_period'] == 'Post-Stress']['duration'].dropna().tolist()

        # --- Latency Plot (Left Column) --- 
        if not df_subset['latency'].dropna().empty:
            plot_generated = True
            # Violin plot - using hue and specified palette
            sns.violinplot(x='stress_period', y='latency', data=df_subset, order=stress_periods, 
                           hue='stress_period', palette=plot_palette, # Correct palette usage
                           inner=None, linewidth=1.5, ax=ax_lat)
            # Strip plot (individual points)
            sns.stripplot(x='stress_period', y='latency', data=df_subset, order=stress_periods, 
                          color='.25', size=3, alpha=0.5, jitter=0.15, ax=ax_lat)
            # Perform Stats
            p_val_lat_num = np.nan
            p_val_lat_str = 'N/A'
            if len(lat_pre) >= min_data_points_for_test and len(lat_post) >= min_data_points_for_test:
                 try:
                     stat, p = mannwhitneyu(lat_pre, lat_post, alternative='two-sided')
                     p_val_lat_num = p
                     # <<< Format p-value scientifically >>>
                     p_val_lat_str = f"p={p:.2e}" if p < 0.001 else f"p={p:.3f}"
                 except ValueError:
                     p_val_lat_str = "p=1.000" # Handle potential identical data error
                     p_val_lat_num = 1.0
            else:
                 p_val_lat_str = "n.s. (n<3)"
            
            # Store results for later annotation
            stats_results[(i, 0)] = {'p_num': p_val_lat_num, 'p_str': p_val_lat_str, 
                                     'data_pre': lat_pre, 'data_post': lat_post} # Store data for mean calc
        else:
            ax_lat.text(0.5, 0.5, 'No latency data', ha='center', va='center', transform=ax_lat.transAxes, color='gray', fontsize=9)
            # ax_lat.set_yticks([]) # Keep ticks for shared axis
            
        # ax_lat.set_ylabel("") # Removed, using for row title
        ax_lat.set_xlabel("") # Remove individual x-label (shared)

        # --- Duration Plot (Right Column) --- 
        if not df_subset['duration'].dropna().empty:
            plot_generated = True
            # Violin plot - using hue and specified palette
            sns.violinplot(x='stress_period', y='duration', data=df_subset, order=stress_periods,
                           hue='stress_period', palette=plot_palette, # Correct palette usage
                           inner=None, linewidth=1.5, ax=ax_dur)
            # Strip plot
            sns.stripplot(x='stress_period', y='duration', data=df_subset, order=stress_periods,
                          color='.25', size=3, alpha=0.5, jitter=0.15, ax=ax_dur)
            ax_dur.axhline(y=1.5, color='gray', linestyle=':', alpha=0.7, linewidth=1) # Max duration line
            # Perform Stats
            p_val_dur_num = np.nan
            p_val_dur_str = 'N/A'
            if len(dur_pre) >= min_data_points_for_test and len(dur_post) >= min_data_points_for_test:
                 try:
                     stat, p = mannwhitneyu(dur_pre, dur_post, alternative='two-sided')
                     p_val_dur_num = p
                     # <<< Format p-value scientifically >>>
                     p_val_dur_str = f"p={p:.2e}" if p < 0.001 else f"p={p:.3f}"
                 except ValueError:
                     p_val_dur_str = "p=1.000" # Handle potential identical data error
                     p_val_dur_num = 1.0
            else:
                 p_val_dur_str = "n.s. (n<3)"
            
            # Store results for later annotation
            stats_results[(i, 1)] = {'p_num': p_val_dur_num, 'p_str': p_val_dur_str,
                                     'data_pre': dur_pre, 'data_post': dur_post} # Store data for mean calc
        else:
            ax_dur.text(0.5, 0.5, 'No duration data', ha='center', va='center', transform=ax_dur.transAxes, color='gray', fontsize=9)
            # ax_dur.set_yticks([]) # Keep ticks for shared axis
            
        # ax_dur.set_ylabel("") # Already cleared
        ax_dur.set_xlabel("") # Remove individual x-label (shared)
            
        # Apply grid
        ax_lat.grid(True, axis='y', alpha=0.5, linestyle='--')
        ax_dur.grid(True, axis='y', alpha=0.5, linestyle='--')

        # Set x-label only for the bottom row
        # if i == len(ALL_REWARD_SIZES) - 1:
        #     ax_lat.set_xlabel("Stress Period")
        #     ax_dur.set_xlabel("Stress Period")
            
        # <<< Set Column Titles on Top Row >>>
        if i == 0:
            ax_lat.set_title("First Lick Latency (s)", fontsize=11)
            ax_dur.set_title("Lick Cut Duration (s)", fontsize=11)
        else:
            # Clear default titles for other rows
            ax_lat.set_title("")
            ax_dur.set_title("")

    # --- Set shared Y-limits & Final Touches --- 
    if all_latencies_overall:
        lat_min, lat_max = np.percentile(all_latencies_overall, [1, 99])
        lat_pad = (lat_max - lat_min) * 0.1 if lat_max > lat_min else 0.1
        final_lat_max = max(1.0, lat_max + lat_pad)
        plt.setp(axs[:, 0], ylim=(max(0, lat_min - lat_pad * 2), final_lat_max * 1.1)) # Add more bottom/top padding for stats line
    else: 
        plt.setp(axs[:, 0], ylim=(0, 1.5)) 
        
    if all_durations_overall:
        dur_min, dur_max = np.percentile(all_durations_overall, [1, 99])
        dur_pad = (dur_max - dur_min) * 0.1 if dur_max > dur_min else 0.1
        final_dur_max = min(1.55, max(0.1, dur_max + dur_pad))
        plt.setp(axs[:, 1], ylim=(max(0, dur_min - dur_pad * 2), final_dur_max * 1.1)) # Add more bottom/top padding
    else: 
        plt.setp(axs[:, 1], ylim=(0, 1.55))
        
    # --- Add Significance Annotations AFTER setting Y limits --- 
    for (row, col), stats in stats_results.items():
        ax = axs[row, col]
        p_num = stats['p_num']
        p_str = stats['p_str']
        # <<< Get data for mean calculation >>>
        data_pre = stats['data_pre']
        data_post = stats['data_post']
        
        if not np.isnan(p_num):
             ylim = ax.get_ylim()
             # Position line higher, e.g., at 90% of the final axis range
             y_line = ylim[1] * 0.9 
             y_line = min(y_line, ylim[1] * 0.95) # Ensure it's below the top boundary
             
             # <<< Calculate means for tick length >>>
             mean_pre = np.mean(data_pre) if data_pre else 0
             mean_post = np.mean(data_post) if data_post else 0
             
             # <<< Calculate max data value in this specific plot for scaling >>>
             y_max_in_plot = 0
             if data_pre: y_max_in_plot = max(y_max_in_plot, np.max(data_pre))
             if data_post: y_max_in_plot = max(y_max_in_plot, np.max(data_post))
             
             # <<< Calculate variable tick lengths >>>
             max_tick_height_abs = (ylim[1] - ylim[0]) * 0.05 # Max height is 5% of range
             scale_factor = max_tick_height_abs / y_max_in_plot if y_max_in_plot > 0 else 0
             tick_len_pre = mean_pre * scale_factor
             tick_len_post = mean_post * scale_factor
             
             # Ensure ticks are not excessively long if means are very high relative to range
             tick_len_pre = min(tick_len_pre, max_tick_height_abs * 1.5)
             tick_len_post = min(tick_len_post, max_tick_height_abs * 1.5)

             # ax.plot([0, 1], [y_line, y_line], color='black', lw=1) # Horizontal line - REMOVED
             # <<< Plot ticks with variable length >>>
             ax.plot([0, 0], [y_line, y_line - tick_len_pre], color='black', lw=1) # Tick 1 (Pre)
             ax.plot([1, 1], [y_line, y_line - tick_len_post], color='black', lw=1) # Tick 2 (Post)
             # <<< Position p-value text slightly above the line >>>
             text_y_offset = (ylim[1] - ylim[0]) * 0.01 # Adjust offset as needed
             ax.text(0.5, y_line + text_y_offset, p_str, ha='center', va='bottom', fontsize=8, fontweight='bold')

             # <<< Add mean values above ticks >>>
             ax.text(0, y_line, f'{mean_pre:.2f}', ha='center', va='bottom', fontsize=7, color='black', fontweight='bold')
             ax.text(1, y_line, f'{mean_post:.2f}', ha='center', va='bottom', fontsize=7, color='black', fontweight='bold')

    # Add shared Y-axis labels (Removed as requested)
    # fig.text(0.01, 0.5, 'First Lick Latency (s)', va='center', ha='left', rotation='vertical', fontsize=12)
    # fig.text(0.51, 0.5, 'Lick Cut Duration (s)', va='center', ha='left', rotation='vertical', fontsize=12)

    plt.tight_layout(rect=[0.04, 0.03, 1, 0.97]) # Adjust layout for titles/labels

    # --- Save Figure --- 
    if not plot_generated:
        print("  No data generated for any comparison panel. Skipping save.")
        plt.close(fig)
        return None
        
    output_filename = os.path.join(output_dir, f"{animal_id}_stress_comparison_by_prev_reward.png")
    try:
        plt.savefig(output_filename, dpi=150) # Use default bbox_inches, might be better with tight_layout
        print(f"  Saved stress comparison figure: {output_filename}")
    except Exception as e:
        print(f"  ERROR saving stress comparison figure: {e}")
        output_filename = None
    finally:
        plt.close(fig)
        
    print(f"  Finished stress comparison plot generation ({time.time() - start_func_time:.2f} seconds).")
    return output_filename
# <<< NEW FUNCTION END >>>

# <<< NEW FUNCTION START: Pooled Stress Comparison >>>

def visualize_stress_comparison_pooled(test_sessions, animal_id, output_dir):
    """Generates a 1x2 plot comparing Pre- vs Post-Stress latency and duration,
       pooling data across all previous reward sizes.
    """
    if not test_sessions:
        print("No test sessions provided for pooled stress comparison.")
        return None

    print(f"\nGenerating Pooled Pre vs Post-Stress comparison plot for {animal_id}...")
    start_func_time = time.time()

    # --- Find first stress day --- 
    first_stress_day = None
    for session in test_sessions:
        if session.get('session_info', {}).get('stress_status', False):
            first_stress_day = session['session_day']
            print(f"  First stress day identified: {first_stress_day}")
            break
            
    if first_stress_day is None:
        print("  No stress sessions found for this animal. Cannot generate pooled comparison plot.")
        return None

    # --- Analysis Parameters & Setup --- 
    stress_periods = ['Pre-Stress', 'Post-Stress']
    plot_palette = {'Pre-Stress': 'skyblue', 'Post-Stress': 'salmon'}
    min_data_points_for_test = 3 # Minimum data points needed in EACH group for stat test

    # --- Data Pooling (Simplified from previous function) --- 
    plot_df_list = []
    all_latencies_overall = []
    all_durations_overall = []

    for session in test_sessions:
        session_day = session['session_day']
        trial_results = session.get('trial_results', [])
        stress_period = 'Post-Stress' if session_day >= first_stress_day else 'Pre-Stress'
        
        if not trial_results or len(trial_results) < 2:
            continue
            
        for i in range(1, len(trial_results)): # Iterate through trials that have a previous trial
            prev_trial = trial_results[i-1]
            current_trial = trial_results[i]

            # Only consider trials following a rewarded trial (standard condition)
            if prev_trial.get('rewarded', False):
                latency = current_trial.get('first_lick_latency')
                cut_duration = None # Initialize
                # Get duration (handle potential nested structure)
                lick_analysis_data = session.get('lick_sensor_cut_analysis', {})
                if 'trial_metrics' in lick_analysis_data:
                    metrics = lick_analysis_data['trial_metrics']
                    current_trial_num = current_trial.get('trial_num')
                    metric = next((m for m in metrics if m.get('trial_num') == current_trial_num), None)
                    if metric and 'post_cue_cut_duration' in metric:
                        cut_duration = metric['post_cue_cut_duration']
                elif 'post_cue_cut_duration' in current_trial: 
                    cut_duration = current_trial['post_cue_cut_duration']
                
                latency_val = latency if (latency is not None and not np.isnan(latency)) else np.nan
                duration_val = cut_duration if (cut_duration is not None and not np.isnan(cut_duration)) else np.nan
                
                if not np.isnan(latency_val): all_latencies_overall.append(latency_val)
                if not np.isnan(duration_val): all_durations_overall.append(duration_val)
                
                # Add to list regardless of previous reward size
                plot_df_list.append({
                    'stress_period': stress_period,
                    'latency': latency_val,
                    'duration': duration_val
                })
                    
    if not plot_df_list:
         print("  No valid data points found after pooling. Cannot generate plot.")
         return None
         
    plot_df = pd.DataFrame(plot_df_list)

    # --- Plotting (1x2 Grid) --- 
    fig, axs = plt.subplots(1, 2, figsize=(8, 5)) # Adjusted figsize for 1x2
    fig.suptitle(f"{animal_id} - Pooled Pre vs Post-Stress Behavior", fontsize=14, y=0.99)

    plot_generated = False
    stats_results = {} # Store stats for annotation

    # --- Panel 0: Latency --- 
    ax_lat = axs[0]
    ax_lat.set_title("First Lick Latency (s)", fontsize=11)
    ax_lat.spines['top'].set_visible(False)
    ax_lat.spines['right'].set_visible(False)
    ax_lat.tick_params(axis='both', which='major', size=6, width=1.5, labelsize=9)
    
    lat_pre = plot_df[plot_df['stress_period'] == 'Pre-Stress']['latency'].dropna().tolist()
    lat_post = plot_df[plot_df['stress_period'] == 'Post-Stress']['latency'].dropna().tolist()

    if not plot_df['latency'].dropna().empty:
        plot_generated = True
        sns.violinplot(x='stress_period', y='latency', data=plot_df, order=stress_periods, 
                       hue='stress_period', palette=plot_palette, inner=None, linewidth=1.5, ax=ax_lat)
        sns.stripplot(x='stress_period', y='latency', data=plot_df, order=stress_periods, 
                      color='.25', size=3, alpha=0.5, jitter=0.15, ax=ax_lat)
        # Stats
        p_val_lat_num = np.nan
        p_val_lat_str = 'N/A'
        if len(lat_pre) >= min_data_points_for_test and len(lat_post) >= min_data_points_for_test:
            try:
                stat, p = mannwhitneyu(lat_pre, lat_post, alternative='two-sided')
                p_val_lat_num = p
                p_val_lat_str = f"p={p:.2e}" if p < 0.001 else f"p={p:.3f}"
            except ValueError:
                p_val_lat_str = "p=1.000"
                p_val_lat_num = 1.0
        else:
            p_val_lat_str = "n.s. (n<3)"
        stats_results[0] = {'p_num': p_val_lat_num, 'p_str': p_val_lat_str, 'data_pre': lat_pre, 'data_post': lat_post}
    else:
        ax_lat.text(0.5, 0.5, 'No latency data', ha='center', va='center', transform=ax_lat.transAxes, color='gray', fontsize=9)

    ax_lat.set_ylabel("Latency (s)", fontsize=10)
    # ax_lat.set_xlabel("Stress Period")
    ax_lat.grid(True, axis='y', alpha=0.5, linestyle='--')

    # --- Panel 1: Duration --- 
    ax_dur = axs[1]
    ax_dur.set_title("Lick Cut Duration (s)", fontsize=11)
    ax_dur.spines['top'].set_visible(False)
    ax_dur.spines['right'].set_visible(False)
    ax_dur.tick_params(axis='both', which='major', size=6, width=1.5, labelsize=9)
    
    dur_pre = plot_df[plot_df['stress_period'] == 'Pre-Stress']['duration'].dropna().tolist()
    dur_post = plot_df[plot_df['stress_period'] == 'Post-Stress']['duration'].dropna().tolist()

    if not plot_df['duration'].dropna().empty:
        plot_generated = True
        sns.violinplot(x='stress_period', y='duration', data=plot_df, order=stress_periods,
                       hue='stress_period', palette=plot_palette, inner=None, linewidth=1.5, ax=ax_dur)
        sns.stripplot(x='stress_period', y='duration', data=plot_df, order=stress_periods,
                      color='.25', size=3, alpha=0.5, jitter=0.15, ax=ax_dur)
        ax_dur.axhline(y=1.5, color='gray', linestyle=':', alpha=0.7, linewidth=1) # Max duration line
        # Stats
        p_val_dur_num = np.nan
        p_val_dur_str = 'N/A'
        if len(dur_pre) >= min_data_points_for_test and len(dur_post) >= min_data_points_for_test:
            try:
                stat, p = mannwhitneyu(dur_pre, dur_post, alternative='two-sided')
                p_val_dur_num = p
                p_val_dur_str = f"p={p:.2e}" if p < 0.001 else f"p={p:.3f}"
            except ValueError:
                p_val_dur_str = "p=1.000"
                p_val_dur_num = 1.0
        else:
            p_val_dur_str = "n.s. (n<3)"
        stats_results[1] = {'p_num': p_val_dur_num, 'p_str': p_val_dur_str, 'data_pre': dur_pre, 'data_post': dur_post}
    else:
        ax_dur.text(0.5, 0.5, 'No duration data', ha='center', va='center', transform=ax_dur.transAxes, color='gray', fontsize=9)
        
    ax_dur.set_ylabel("Duration (s)", fontsize=10)
    # ax_dur.set_xlabel("Stress Period")
    ax_dur.grid(True, axis='y', alpha=0.5, linestyle='--')

    # --- Set shared Y-limits --- 
    if all_latencies_overall:
        lat_min, lat_max = np.percentile(all_latencies_overall, [1, 99])
        lat_pad = (lat_max - lat_min) * 0.1 if lat_max > lat_min else 0.1
        final_lat_max = max(1.0, lat_max + lat_pad)
        ax_lat.set_ylim(max(0, lat_min - lat_pad * 2), final_lat_max * 1.1)
    else: 
        ax_lat.set_ylim(0, 1.5) 
        
    if all_durations_overall:
        dur_min, dur_max = np.percentile(all_durations_overall, [1, 99])
        dur_pad = (dur_max - dur_min) * 0.1 if dur_max > dur_min else 0.1
        final_dur_max = min(1.55, max(0.1, dur_max + dur_pad))
        ax_dur.set_ylim(max(0, dur_min - dur_pad * 2), final_dur_max * 1.1)
    else: 
        ax_dur.set_ylim(0, 1.55)
        
    # --- Add Significance Annotations --- 
    for col, stats in stats_results.items():
        ax = axs[col]
        p_num = stats['p_num']
        p_str = stats['p_str']
        data_pre = stats['data_pre']
        data_post = stats['data_post']
        
        if not np.isnan(p_num):
            ylim = ax.get_ylim()
            y_line = ylim[1] * 0.9 
            y_line = min(y_line, ylim[1] * 0.95)
            
            mean_pre = np.mean(data_pre) if data_pre else 0
            mean_post = np.mean(data_post) if data_post else 0
            
            y_max_in_plot = 0
            if data_pre: y_max_in_plot = max(y_max_in_plot, np.max(data_pre))
            if data_post: y_max_in_plot = max(y_max_in_plot, np.max(data_post))
            
            max_tick_height_abs = (ylim[1] - ylim[0]) * 0.05
            scale_factor = max_tick_height_abs / y_max_in_plot if y_max_in_plot > 0 else 0
            tick_len_pre = mean_pre * scale_factor
            tick_len_post = mean_post * scale_factor
            
            tick_len_pre = min(tick_len_pre, max_tick_height_abs * 1.5)
            tick_len_post = min(tick_len_post, max_tick_height_abs * 1.5)

            # No horizontal line
            ax.plot([0, 0], [y_line, y_line - tick_len_pre], color='black', lw=1)
            ax.plot([1, 1], [y_line, y_line - tick_len_post], color='black', lw=1)
            
            text_y_offset = (ylim[1] - ylim[0]) * 0.01
            ax.text(0.5, y_line + text_y_offset, p_str, ha='center', va='bottom', fontsize=8, fontweight='bold')

            ax.text(0, y_line, f'{mean_pre:.2f}', ha='center', va='bottom', fontsize=7, color='black', fontweight='bold')
            ax.text(1, y_line, f'{mean_post:.2f}', ha='center', va='bottom', fontsize=7, color='black', fontweight='bold')

    plt.tight_layout(rect=[0.04, 0.03, 1, 0.95])

    # --- Save Figure --- 
    if not plot_generated:
        print("  No data generated for pooled comparison. Skipping save.")
        plt.close(fig)
        return None
        
    output_filename = os.path.join(output_dir, f"{animal_id}_stress_comparison_pooled.png")
    try:
        plt.savefig(output_filename, dpi=150)
        print(f"  Saved pooled stress comparison figure: {output_filename}")
    except Exception as e:
        print(f"  ERROR saving pooled stress comparison figure: {e}")
        output_filename = None
    finally:
        plt.close(fig)
        
    print(f"  Finished pooled stress comparison plot generation ({time.time() - start_func_time:.2f} seconds).")
    return output_filename
# <<< NEW FUNCTION END >>>

# <<< NEW FUNCTION START: Cross-Session Time-Resolved Licking Summary >>>
def visualize_cross_session_time_resolved_licking(test_sessions, animal_id, output_dir):
    """Generates a 2x2 summary plot comparing time-resolved licking 
       (Pre vs Post-Stress), grouped by current or previous reward size.
    """
    if not test_sessions:
        print("No test sessions provided for cross-session time-resolved summary.")
        return None

    print(f"\nGenerating Cross-Session Time-Resolved Licking Summary plot for {animal_id}...")
    start_func_time = time.time()

    # --- Find first stress day --- 
    first_stress_day = None
    for session in test_sessions:
        if session.get('session_info', {}).get('stress_status', False):
            first_stress_day = session['session_day']
            print(f"  First stress day identified: {first_stress_day}")
            break
            
    if first_stress_day is None:
        print("  No stress sessions found. Cannot generate cross-session time-resolved plot.")
        return None

    # --- Analysis Parameters (Copied from visualize_time_resolved_licking) ---
    ANALYSIS_START_TIME_REL_REWARD = -2.0
    TOTAL_DURATION_POST_REWARD = 8.0 
    ANALYSIS_END_TIME_REL_REWARD = TOTAL_DURATION_POST_REWARD
    WINDOW_SIZE = 0.1  
    STEP_SIZE = 0.02   
    TIME_RESOLUTION = 0.001 
    size_labels_map_ts = {1: 'Very Small', 2: 'Small', 3: 'Medium', 4: 'Large', 5: 'Very Large'}
    ALL_REWARD_SIZES = list(size_labels_map_ts.keys())

    # --- Shared Calculation Setup ---
    time_points_relative_to_reward = np.arange(ANALYSIS_START_TIME_REL_REWARD + STEP_SIZE,
                                               ANALYSIS_END_TIME_REL_REWARD + STEP_SIZE,
                                               STEP_SIZE)
    time_points_relative_to_reward = time_points_relative_to_reward[time_points_relative_to_reward <= ANALYSIS_END_TIME_REL_REWARD + 1e-9]
    num_target_time_points = len(time_points_relative_to_reward)
    hires_trace_duration_rel_reward = ANALYSIS_END_TIME_REL_REWARD - ANALYSIS_START_TIME_REL_REWARD
    num_hires_points_total_analysis = int(np.round(hires_trace_duration_rel_reward / TIME_RESOLUTION))
    target_indices_in_hires = np.round((time_points_relative_to_reward - ANALYSIS_START_TIME_REL_REWARD) / TIME_RESOLUTION).astype(int) - 1
    target_indices_in_hires = np.clip(target_indices_in_hires, 0, num_hires_points_total_analysis - 1)
    win_samples = int(np.round(WINDOW_SIZE / TIME_RESOLUTION))
    win_samples = max(1, win_samples)

    # --- Helper function (Copied from visualize_time_resolved_licking) ---
    # Ideally, this would be in a shared module
    def calculate_single_trial_ts(mtrial, reward_latency):
        lick_downs = mtrial.get('lick_downs_relative', [])
        lick_ups = mtrial.get('lick_ups_relative', [])
        analysis_start_rel_trial = reward_latency + ANALYSIS_START_TIME_REL_REWARD
        analysis_end_rel_trial = reward_latency + ANALYSIS_END_TIME_REL_REWARD
        max_time_rel_trial_needed = analysis_end_rel_trial + TIME_RESOLUTION
        num_hires_points_trial = int(np.ceil(max_time_rel_trial_needed / TIME_RESOLUTION))
        lick_trace_trial = np.zeros(num_hires_points_trial, dtype=np.int8)
        lick_downs_sorted = sorted([ld for ld in lick_downs if ld is not None])
        lick_ups_sorted = sorted([lu for lu in lick_ups if lu is not None])
        num_bouts = min(len(lick_downs_sorted), len(lick_ups_sorted))
        for i in range(num_bouts):
            start_idx = int(np.floor((lick_downs_sorted[i] + 1e-9) / TIME_RESOLUTION))
            end_idx = int(np.ceil(lick_ups_sorted[i] / TIME_RESOLUTION))
            start_idx = max(0, start_idx); end_idx = min(num_hires_points_trial, end_idx)
            if start_idx < end_idx: lick_trace_trial[start_idx:end_idx] = 1
        idx_analysis_start_in_trial_trace = int(np.round(analysis_start_rel_trial / TIME_RESOLUTION))
        idx_analysis_end_in_trial_trace = idx_analysis_start_in_trial_trace + num_hires_points_total_analysis
        actual_start_idx = max(0, idx_analysis_start_in_trial_trace)
        actual_end_idx = min(len(lick_trace_trial), idx_analysis_end_in_trial_trace)
        extracted_trace = lick_trace_trial[actual_start_idx:actual_end_idx]
        pad_before = max(0, -idx_analysis_start_in_trial_trace)
        pad_after = max(0, num_hires_points_total_analysis - (len(extracted_trace) + pad_before))
        analysis_trace_hires = np.pad(extracted_trace, (pad_before, pad_after), 'constant')
        if len(analysis_trace_hires) != num_hires_points_total_analysis:
            return np.full(num_target_time_points, np.nan)
        rolling_proportions = pd.Series(analysis_trace_hires).rolling(window=win_samples, min_periods=1, center=True).mean().to_numpy()
        current_target_indices = target_indices_in_hires[target_indices_in_hires < len(rolling_proportions)]
        sampled_proportions = np.full(num_target_time_points, np.nan)
        valid_indices_mask = target_indices_in_hires < len(rolling_proportions)
        valid_target_indices_for_sampling = target_indices_in_hires[valid_indices_mask]
        sampled_proportions[valid_indices_mask] = rolling_proportions[valid_target_indices_for_sampling]
        return sampled_proportions

    # --- Data Pooling Structure --- 
    pooled_ts_data = {
        'Pre-Stress': {
            'current_size': {size: [] for size in ALL_REWARD_SIZES},
            'previous_size': {size: [] for size in ALL_REWARD_SIZES}
        },
        'Post-Stress': {
            'current_size': {size: [] for size in ALL_REWARD_SIZES},
            'previous_size': {size: [] for size in ALL_REWARD_SIZES}
        }
    }

    # --- Data Processing Loop --- 
    print("  Pooling time series data across sessions...")
    sessions_processed = 0
    for session in test_sessions:
        session_day = session['session_day']
        stress_period = 'Post-Stress' if session_day >= first_stress_day else 'Pre-Stress'
        monitoring_trials = session.get('trial_results', [])
        if not monitoring_trials:
            continue
        
        # Reconstruct reward map for this session
        reward_sizes_map_session = {}
        for trial in monitoring_trials:
            t_num = trial.get('trial_num')
            r_size = trial.get('reward_size')
            if t_num is not None and r_size is not None:
                reward_sizes_map_session[int(t_num)] = int(r_size)
        if not reward_sizes_map_session:
            continue # Skip session if no reward map
            
        sessions_processed += 1
        # --- Group by Current Reward Size ---
        for trial_idx, mtrial in enumerate(monitoring_trials):
            trial_num = mtrial.get('trial_num')
            is_rewarded = mtrial.get('rewarded', False)
            reward_latency = mtrial.get('reward_latency')
            if trial_num is None or not is_rewarded or reward_latency is None: continue
            reward_size = reward_sizes_map_session.get(int(trial_num))
            if reward_size not in ALL_REWARD_SIZES: continue
            
            try:
                sampled_proportions = calculate_single_trial_ts(mtrial, reward_latency)
                if sampled_proportions is not None and not np.all(np.isnan(sampled_proportions)):
                     pooled_ts_data[stress_period]['current_size'][reward_size].append(sampled_proportions)
            except Exception as e:
                 # print(f"Warning: Error processing trial {trial_num} (Current Size): {e}")
                 pass
                 
        # --- Group by Previous Reward Size ---
        if len(monitoring_trials) >= 2:
            for idx in range(1, len(monitoring_trials)):
                mtrial_curr = monitoring_trials[idx]
                mtrial_prev = monitoring_trials[idx-1]
                trial_num_curr = mtrial_curr.get('trial_num')
                is_rewarded_curr = mtrial_curr.get('rewarded', False)
                reward_latency_curr = mtrial_curr.get('reward_latency')
                if trial_num_curr is None or not is_rewarded_curr or reward_latency_curr is None: continue

                is_rewarded_prev = mtrial_prev.get('rewarded', False)
                trial_num_prev = mtrial_prev.get('trial_num')
                if not is_rewarded_prev or trial_num_prev is None: continue
                reward_size_prev = reward_sizes_map_session.get(int(trial_num_prev))
                if reward_size_prev not in ALL_REWARD_SIZES: continue

                try:
                    sampled_proportions = calculate_single_trial_ts(mtrial_curr, reward_latency_curr)
                    if sampled_proportions is not None and not np.all(np.isnan(sampled_proportions)):
                         pooled_ts_data[stress_period]['previous_size'][reward_size_prev].append(sampled_proportions)
                except Exception as e:
                     # print(f"Warning: Error processing trial {trial_num_curr} (Prev Size): {e}")
                     pass
                     
    print(f"  Processed {sessions_processed} sessions for pooling.")

    # --- Aggregation Function (similar to visualize_time_resolved_licking) ---
    def aggregate_pooled_data(pooled_dict):
        aggregated_results = {}
        for stress_p in pooled_dict:
            aggregated_results[stress_p] = {}
            for group_type in pooled_dict[stress_p]:
                aggregated_results[stress_p][group_type] = {}
                for r_size in pooled_dict[stress_p][group_type]:
                    trial_list = pooled_dict[stress_p][group_type][r_size]
                    # Filter out trials that are None or all NaNs
                    valid_trials_ts = [ts for ts in trial_list if ts is not None and not np.all(np.isnan(ts))]
                    # Further filter based on expected length
                    valid_trials_ts = [ts for ts in valid_trials_ts if len(ts) == num_target_time_points]
                    n_trials = len(valid_trials_ts)
                    mean_ts, sem_ts = None, None
                    if n_trials > 0:
                        trial_ts_stack = np.array(valid_trials_ts)
                        mean_ts = np.nanmean(trial_ts_stack, axis=0)
                        if n_trials > 1:
                            with np.errstate(invalid='ignore'): sem_ts = sem(trial_ts_stack, axis=0, nan_policy='omit')
                            sem_ts = np.nan_to_num(sem_ts, nan=0.0)
                        else: sem_ts = np.zeros_like(mean_ts)
                    aggregated_results[stress_p][group_type][r_size] = {'mean': mean_ts, 'sem': sem_ts, 'n': n_trials}
        return aggregated_results

    aggregated_data = aggregate_pooled_data(pooled_ts_data)

    # --- Plotting --- 
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey='row') # Share Y axis per row
    fig.suptitle(f"{animal_id} - Cross-Session Time-Resolved Licking (Pre vs Post-Stress)", fontsize=16, y=0.99)
    cmap = plt.cm.viridis 
    norm = mcolors.Normalize(vmin=min(ALL_REWARD_SIZES), vmax=max(ALL_REWARD_SIZES))

    plot_successful = False # Track if any data is plotted at all

    grouping_types = ['current_size', 'previous_size']
    row_titles = ['Grouped by Current Reward Size', 'Grouped by Previous Reward Size']
    stress_periods_plot = ['Pre-Stress', 'Post-Stress']

    for row_idx, group_type in enumerate(grouping_types):
        max_y_row = 0 # Track max Y for this row to set shared ylim
        for col_idx, stress_p in enumerate(stress_periods_plot):
            ax = axs[row_idx, col_idx]
            data_to_plot = aggregated_data[stress_p][group_type]
            panel_has_data = False
            
            for r_size in ALL_REWARD_SIZES:
                agg_res = data_to_plot.get(r_size, {})
                mean_ts = agg_res.get('mean')
                sem_ts = agg_res.get('sem')
                n_val = agg_res.get('n', 0)
                
                if mean_ts is not None and sem_ts is not None and n_val > 0:
                    panel_has_data = True
                    plot_successful = True
                    color = cmap(norm(r_size))
                    label = f"Size {r_size} ({size_labels_map_ts.get(r_size, '')}, N={n_val})" 
                    ax.plot(time_points_relative_to_reward, mean_ts, color=color, label=label, linewidth=1.5)
                    ax.fill_between(time_points_relative_to_reward, mean_ts - sem_ts, mean_ts + sem_ts, color=color, alpha=0.2)
                    max_y_row = max(max_y_row, np.nanmax(mean_ts + sem_ts))
            
            ax.axvline(x=0, color='red', linestyle=':', linewidth=1, alpha=0.7, label='_nolegend_') # Add reward line
            ax.grid(True, alpha=0.5, linestyle='--')
            title = f"{stress_p}"
            ax.set_title(title, fontsize=12)

            if not panel_has_data:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, color='gray')

            # Labels and Legend
            if col_idx == 0: # Left column
                ax.set_ylabel("Mean Lick Proportion", fontsize=10)
                ax.annotate(row_titles[row_idx], xy=(0.1, 0.5), xycoords='axes fraction', # Adjusted x position slightly left
                            fontsize=12, rotation=90, va='center', ha='right')
            if row_idx == 1: # Bottom row
                ax.set_xlabel("Time Relative to Reward Delivery (s)", fontsize=10)
            
            # <<< MODIFICATION: Add legend to BOTH columns >>>
            # if col_idx == 1: # Right column (add legend here) -- REMOVED CONDITION
            handles, labels = ax.get_legend_handles_labels()
            # Add reward line handle/label if not present
            if not any(line.get_color() == 'red' for line in handles):
                red_line = plt.Line2D([0], [0], color='red', linestyle=':', linewidth=1, alpha=0.7)
                handles.append(red_line)
                labels.append('Reward Delivery')
            # Add legend to the current axis (will run for both col_idx 0 and 1)
            ax.legend(handles=handles, labels=labels, title="Reward Size", loc='upper right', fontsize='x-small')
            # <<< MODIFICATION END >>>

        # Set shared Y limits for the row
        axs[row_idx, 0].set_ylim(bottom=-0.05, top=max(0.1, max_y_row * 1.1)) # Ensure some minimum height
        axs[row_idx, 1].set_ylim(bottom=-0.05, top=max(0.1, max_y_row * 1.1))
        
    # Final layout adjustment
    plt.tight_layout(rect=[0.05, 0.03, 1, 0.96]) # Adjust left margin for row titles

    # --- Save Figure --- 
    if not plot_successful:
        print("  No data plotted for any panel. Skipping save.")
        plt.close(fig)
        return None
        
    output_filename = os.path.join(output_dir, f"{animal_id}_time_resolved_licking_stress_summary.png")
    try:
        plt.savefig(output_filename, dpi=150)
        print(f"  Saved cross-session time-resolved licking summary: {output_filename}")
    except Exception as e:
        print(f"  ERROR saving cross-session time-resolved licking summary: {e}")
        output_filename = None
    finally:
        plt.close(fig)
        
    print(f"  Finished cross-session time-resolved summary plot generation ({time.time() - start_func_time:.2f} seconds).")
    return output_filename
# <<< NEW FUNCTION END >>>

# <<< NEW FUNCTION START: Cross-Session Summary of Licking by Previous Extremes (Pre/Post Stress) >>>
def visualize_cross_session_prev_extremes_stress_summary(test_sessions, animal_id, output_dir):
    """Generates a 5x2 summary plot comparing time-resolved licking (Pre vs Post-Stress),
       for each current reward size, grouped by previous reward extremes (VS/S vs L/VL).
    """
    if not test_sessions:
        print("No test sessions provided for cross-session prev extremes summary.")
        return None

    print(f"\nGenerating Cross-Session Previous Extremes Summary plot for {animal_id}...")
    start_func_time = time.time()

    # --- Find first stress day --- 
    first_stress_day = None
    for session in test_sessions:
        if session.get('session_info', {}).get('stress_status', False):
            first_stress_day = session['session_day']
            print(f"  First stress day identified: {first_stress_day}")
            break
            
    if first_stress_day is None:
        print("  No stress sessions found. Cannot generate cross-session prev extremes plot.")
        return None

    # --- Analysis Parameters (Copied from visualize_licking_by_prev_extremes) ---
    ANALYSIS_START_TIME_REL_REWARD = -2.0
    TOTAL_DURATION_POST_REWARD = 8.0 
    ANALYSIS_END_TIME_REL_REWARD = TOTAL_DURATION_POST_REWARD
    WINDOW_SIZE = 0.1  
    STEP_SIZE = 0.02   
    TIME_RESOLUTION = 0.001 
    ALL_REWARD_SIZES = [1, 2, 3, 4, 5]
    size_labels_map = {1: 'VS', 2: 'S', 3: 'M', 4: 'L', 5: 'VL'}
    PREV_VS_S_SIZES = [1, 2]
    PREV_L_VL_SIZES = [4, 5]
    GROUP1_LABEL = "Prev VS/S"
    GROUP2_LABEL = "Prev L/VL"
    GROUP_COLORS = {GROUP1_LABEL: 'blue', GROUP2_LABEL: 'orange'}
    STRESS_PERIODS = ['Pre-Stress', 'Post-Stress']

    # --- Shared Calculation Setup --- (Copied)
    time_points_relative_to_reward = np.arange(ANALYSIS_START_TIME_REL_REWARD + STEP_SIZE,
                                               ANALYSIS_END_TIME_REL_REWARD + STEP_SIZE,
                                               STEP_SIZE)
    time_points_relative_to_reward = time_points_relative_to_reward[time_points_relative_to_reward <= ANALYSIS_END_TIME_REL_REWARD + 1e-9]
    num_target_time_points = len(time_points_relative_to_reward)
    hires_trace_duration_rel_reward = ANALYSIS_END_TIME_REL_REWARD - ANALYSIS_START_TIME_REL_REWARD
    num_hires_points_total_analysis = int(np.round(hires_trace_duration_rel_reward / TIME_RESOLUTION))
    target_indices_in_hires = np.round((time_points_relative_to_reward - ANALYSIS_START_TIME_REL_REWARD) / TIME_RESOLUTION).astype(int) - 1
    target_indices_in_hires = np.clip(target_indices_in_hires, 0, num_hires_points_total_analysis - 1)
    win_samples = int(np.round(WINDOW_SIZE / TIME_RESOLUTION))
    win_samples = max(1, win_samples)

    # --- Helper function (Assume available from previous steps or copy here) ---
    # Simplified placeholder - requires the actual function from visualize_time_resolved_licking
    def calculate_single_trial_ts(mtrial, reward_latency):
        lick_downs = mtrial.get('lick_downs_relative', [])
        lick_ups = mtrial.get('lick_ups_relative', [])
        analysis_start_rel_trial = reward_latency + ANALYSIS_START_TIME_REL_REWARD
        analysis_end_rel_trial = reward_latency + ANALYSIS_END_TIME_REL_REWARD
        max_time_rel_trial_needed = analysis_end_rel_trial + TIME_RESOLUTION
        num_hires_points_trial = int(np.ceil(max_time_rel_trial_needed / TIME_RESOLUTION))
        lick_trace_trial = np.zeros(num_hires_points_trial, dtype=np.int8)
        lick_downs_sorted = sorted([ld for ld in lick_downs if ld is not None])
        lick_ups_sorted = sorted([lu for lu in lick_ups if lu is not None])
        num_bouts = min(len(lick_downs_sorted), len(lick_ups_sorted))
        for i in range(num_bouts):
            start_idx = int(np.floor((lick_downs_sorted[i] + 1e-9) / TIME_RESOLUTION))
            end_idx = int(np.ceil(lick_ups_sorted[i] / TIME_RESOLUTION))
            start_idx = max(0, start_idx); end_idx = min(num_hires_points_trial, end_idx)
            if start_idx < end_idx: lick_trace_trial[start_idx:end_idx] = 1
        idx_analysis_start_in_trial_trace = int(np.round(analysis_start_rel_trial / TIME_RESOLUTION))
        idx_analysis_end_in_trial_trace = idx_analysis_start_in_trial_trace + num_hires_points_total_analysis
        actual_start_idx = max(0, idx_analysis_start_in_trial_trace)
        actual_end_idx = min(len(lick_trace_trial), idx_analysis_end_in_trial_trace)
        extracted_trace = lick_trace_trial[actual_start_idx:actual_end_idx]
        pad_before = max(0, -idx_analysis_start_in_trial_trace)
        pad_after = max(0, num_hires_points_total_analysis - (len(extracted_trace) + pad_before))
        analysis_trace_hires = np.pad(extracted_trace, (pad_before, pad_after), 'constant')
        if len(analysis_trace_hires) != num_hires_points_total_analysis:
            return np.full(num_target_time_points, np.nan)
        rolling_proportions = pd.Series(analysis_trace_hires).rolling(window=win_samples, min_periods=1, center=True).mean().to_numpy()
        current_target_indices = target_indices_in_hires[target_indices_in_hires < len(rolling_proportions)]
        sampled_proportions = np.full(num_target_time_points, np.nan)
        valid_indices_mask = target_indices_in_hires < len(rolling_proportions)
        valid_target_indices_for_sampling = target_indices_in_hires[valid_indices_mask]
        sampled_proportions[valid_indices_mask] = rolling_proportions[valid_target_indices_for_sampling]
        return sampled_proportions

    # --- Data Pooling Structure --- 
    # pooled_data[stress_period][current_reward_size][prev_extreme_group] = [trace1, trace2, ...]
    pooled_data = {stress_p: {curr_size: {GROUP1_LABEL: [], GROUP2_LABEL: []} 
                              for curr_size in ALL_REWARD_SIZES} 
                   for stress_p in STRESS_PERIODS}

    # --- Data Processing Loop --- 
    print("  Pooling time series data across sessions for prev extremes summary...")
    sessions_processed = 0
    for session in test_sessions:
        session_day = session['session_day']
        stress_period = 'Post-Stress' if session_day >= first_stress_day else 'Pre-Stress'
        monitoring_trials = session.get('trial_results', [])
        if not monitoring_trials or len(monitoring_trials) < 2:
            continue
        
        # Reconstruct reward map for this session
        reward_sizes_map_session = {}
        for trial in monitoring_trials:
            t_num = trial.get('trial_num'); r_size = trial.get('reward_size')
            if t_num is not None and r_size is not None: reward_sizes_map_session[int(t_num)] = int(r_size)
        if not reward_sizes_map_session: continue
            
        sessions_processed += 1
        # Iterate through trials that have a preceding trial
        for idx in range(1, len(monitoring_trials)):
            mtrial_curr = monitoring_trials[idx]
            mtrial_prev = monitoring_trials[idx-1]
            trial_num_curr = mtrial_curr.get('trial_num')
            
            try:
                # Check current trial validity
                reward_size_curr = reward_sizes_map_session.get(int(trial_num_curr))
                if reward_size_curr not in ALL_REWARD_SIZES: continue
                is_rewarded_curr = mtrial_curr.get('rewarded', False)
                reward_latency_curr = mtrial_curr.get('reward_latency')
                if not is_rewarded_curr or reward_latency_curr is None: continue
                
                # Check previous trial validity and reward size group
                is_rewarded_prev = mtrial_prev.get('rewarded', False)
                trial_num_prev = mtrial_prev.get('trial_num')
                if not is_rewarded_prev or trial_num_prev is None: continue
                reward_size_prev = reward_sizes_map_session.get(int(trial_num_prev))
                if reward_size_prev is None: continue

                prev_group_key = None
                if reward_size_prev in PREV_VS_S_SIZES: prev_group_key = GROUP1_LABEL
                elif reward_size_prev in PREV_L_VL_SIZES: prev_group_key = GROUP2_LABEL
                else: continue # Skip if previous reward not in extreme groups
                
                # Calculate and store the trace
                sampled_proportions = calculate_single_trial_ts(mtrial_curr, reward_latency_curr)
                if sampled_proportions is not None and not np.all(np.isnan(sampled_proportions)):
                     pooled_data[stress_period][reward_size_curr][prev_group_key].append(sampled_proportions)
                     
            except Exception as e:
                 # print(f"Warning: Error processing trial {trial_num_curr} for prev extremes summary: {e}")
                 pass
                     
    print(f"  Processed {sessions_processed} sessions for prev extremes pooling.")

    # --- Aggregation Function --- (Similar to previous functions)
    def aggregate_pooled_extremes_data(data_dict):
        aggregated_results = {stress_p: {curr_size: {} for curr_size in ALL_REWARD_SIZES} 
                               for stress_p in STRESS_PERIODS}
        for stress_p in data_dict:
            for curr_size in data_dict[stress_p]:
                for group_label in data_dict[stress_p][curr_size]:
                    trial_list = data_dict[stress_p][curr_size][group_label]
                    valid_trials_ts = [ts for ts in trial_list if ts is not None 
                                       and not np.all(np.isnan(ts)) 
                                       and len(ts) == num_target_time_points]
                    n_trials = len(valid_trials_ts)
                    mean_ts, sem_ts = None, None
                    if n_trials > 0:
                        trial_ts_stack = np.array(valid_trials_ts)
                        mean_ts = np.nanmean(trial_ts_stack, axis=0)
                        if n_trials > 1:
                            with np.errstate(invalid='ignore'): sem_ts = sem(trial_ts_stack, axis=0, nan_policy='omit')
                            sem_ts = np.nan_to_num(sem_ts, nan=0.0)
                        else: sem_ts = np.zeros_like(mean_ts)
                    aggregated_results[stress_p][curr_size][group_label] = {'mean': mean_ts, 'sem': sem_ts, 'n': n_trials}
        return aggregated_results

    aggregated_data = aggregate_pooled_extremes_data(pooled_data)

    # --- Plotting --- 
    fig, axs = plt.subplots(len(ALL_REWARD_SIZES), 2, 
                             figsize=(10, 12), # Adjusted figsize
                             sharex=True, sharey=True) # Share both axes
    if len(ALL_REWARD_SIZES) == 1: axs = np.array([axs]) # Ensure axs is 2D
        
    fig.suptitle(f"{animal_id} - Licking by Previous Extremes (Pre vs Post-Stress)", fontsize=14, y=0.99)

    plot_successful = False # Track if any data is plotted at all
    max_y_overall = 0

    for row_idx, curr_size in enumerate(ALL_REWARD_SIZES):
        for col_idx, stress_p in enumerate(STRESS_PERIODS):
            ax = axs[row_idx, col_idx]
            data_to_plot = aggregated_data[stress_p][curr_size]
            panel_has_data = False
            legend_handles = []
            legend_labels = []
            
            for group_label in [GROUP1_LABEL, GROUP2_LABEL]:
                agg_res = data_to_plot.get(group_label, {})
                mean_ts = agg_res.get('mean')
                sem_ts = agg_res.get('sem')
                n_val = agg_res.get('n', 0)
                
                if mean_ts is not None and sem_ts is not None and n_val > 0:
                    panel_has_data = True
                    plot_successful = True
                    color = GROUP_COLORS[group_label]
                    label = f"{group_label} (N={n_val})" 
                    line, = ax.plot(time_points_relative_to_reward, mean_ts, color=color, label=label, linewidth=1.5)
                    ax.fill_between(time_points_relative_to_reward, mean_ts - sem_ts, mean_ts + sem_ts, color=color, alpha=0.2)
                    max_y_overall = max(max_y_overall, np.nanmax(mean_ts + sem_ts))
                    legend_handles.append(line)
                    legend_labels.append(label)
            
            ax.axvline(x=0, color='red', linestyle=':', linewidth=1, alpha=0.7, label='_nolegend_')
            ax.grid(True, alpha=0.5, linestyle='--')

            if not panel_has_data:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, color='gray')
            else:
                # Add legend to this specific subplot
                ax.legend(handles=legend_handles, labels=legend_labels, loc='upper right', fontsize='x-small')

            # Column Titles (Top Row Only)
            if row_idx == 0:
                ax.set_title(stress_p, fontsize=11)
            
            # Row Titles (Left Column Only)
            if col_idx == 0:
                row_title = f"Current: {size_labels_map[curr_size]} ({curr_size})"
                ax.set_ylabel(row_title, fontsize=9)
            
            # X-axis Label (Bottom Row Only)
            if row_idx == len(ALL_REWARD_SIZES) - 1:
                ax.set_xlabel("Time Relative to Reward (s)", fontsize=10)
                
            # Remove redundant y-axis labels for inner plots
            if col_idx > 0:
                 plt.setp(ax.get_yticklabels(), visible=False)
            if row_idx < len(ALL_REWARD_SIZES) - 1:
                 plt.setp(ax.get_xticklabels(), visible=False)

    # Set shared Y limits after plotting all data
    plt.setp(axs, ylim=(-0.05, max(1.05, max_y_overall * 1.1))) 
    plt.setp(axs, xlim=(ANALYSIS_START_TIME_REL_REWARD, ANALYSIS_END_TIME_REL_REWARD))
        
    # Shared Y-axis label for the entire figure
    fig.text(0.01, 0.5, "Mean Lick Proportion", va='center', rotation='vertical', fontsize=12)

    # Final layout adjustment
    plt.tight_layout(rect=[0.05, 0.03, 1, 0.96]) # Adjust margins

    # --- Save Figure --- 
    if not plot_successful:
        print("  No data plotted for any panel. Skipping save.")
        plt.close(fig)
        return None
        
    output_filename = os.path.join(output_dir, f"{animal_id}_prev_extremes_stress_summary.png")
    try:
        plt.savefig(output_filename, dpi=150)
        print(f"  Saved cross-session prev extremes summary: {output_filename}")
    except Exception as e:
        print(f"  ERROR saving cross-session prev extremes summary: {e}")
        output_filename = None
    finally:
        plt.close(fig)
        
    print(f"  Finished cross-session prev extremes summary plot generation ({time.time() - start_func_time:.2f} seconds).")
    return output_filename
# <<< NEW FUNCTION END >>>