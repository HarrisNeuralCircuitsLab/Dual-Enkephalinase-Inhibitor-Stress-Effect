"""
Phase-aware visualization utilities for behavioral analysis.

Contains functions adapted from visualization.py to handle distinct experimental phases.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pandas as pd
from scipy.stats import sem, mannwhitneyu, kruskal
# Potential need for post-hoc test if using Kruskal-Wallis
# from scikit_posthocs import posthoc_dunn 
import matplotlib.colors as mcolors
import time
from matplotlib.ticker import FixedLocator, ScalarFormatter # Import for explicit tick control

# Use a non-interactive backend to prevent blocking
matplotlib.use('Agg')

# Set plot style (matching visualization.py)
sns.set_context("talk")
plt.rcParams['figure.figsize'] = (12, 8) # Default, individual functions may override
plt.rcParams['savefig.dpi'] = 150 # Increase default save DPI
plt.rcParams['figure.dpi'] = 100

# Define phase order and colors for consistency
PHASE_ORDER = ["Initial Test", "Stress Test (Pre-PL37)", "PL37 Test", "Post-PL37 Test"]
PHASE_ORDER_COMPARISON = ["Initial Test", "Stress Test (Pre-PL37)", "Post-PL37 Test"] # For 3-way comparison plots

# Define a color palette (adjust as needed)
# Using seaborn's 'colorblind' palette and manually mapping
cb_palette = sns.color_palette('colorblind', n_colors=4)
PHASE_COLORS = {
    "Initial Test": cb_palette[0],
    "Stress Test (Pre-PL37)": cb_palette[1],
    "PL37 Test": cb_palette[2], # Often excluded from direct comparisons
    "Post-PL37 Test": cb_palette[3] 
}
PHASE_COLORS_COMPARISON = {p: PHASE_COLORS[p] for p in PHASE_ORDER_COMPARISON}


# --- Functions adapted from visualization.py --- 

def visualize_animal_sessions_phase_aware(animal_sessions, animal_id, output_dir=None,
                                          training_test_boundary_day=None, # <-- ADD NEW ARGUMENT
                                          pl37_vline_day=None, post_pl37_vline_day=None):
    """Generate visualizations for all sessions of an animal, adding phase lines.
    
    Adapted from visualize_animal_sessions to include vertical lines indicating
    the start of specific phases (PL37, Post-PL37).
    """
    if output_dir is None:
        output_dir = os.path.join('Figures', animal_id)
    os.makedirs(output_dir, exist_ok=True)
    
    if not animal_sessions:
        print("No session data to visualize")
        return None # Return None if no data

    # Sort ALL sessions by day 
    animal_sessions.sort(key=lambda x: x['session_day'])

    # Extract data
    days = [session['session_day'] for session in animal_sessions]
    performance = [session['performance'] for session in animal_sessions]
    total_trials = [session['total_trials'] for session in animal_sessions]
    rewarded_trials = [session['rewarded_trials'] for session in animal_sessions]

    first_lick_times_by_session = []
    for session in animal_sessions:
        if 'first_lick_times' in session:
            first_lick_times_by_session.append(session['first_lick_times'])
        elif 'trial_results' in session: 
            session_licks = []
            for trial in session['trial_results']:
                if 'first_lick_latency' in trial and trial['first_lick_latency'] is not None:
                    session_licks.append(trial['first_lick_latency'])
            first_lick_times_by_session.append(session_licks)
        else:
            first_lick_times_by_session.append([])

    avg_lick_times = []
    std_lick_times = []
    sem_lick_times = []
    for session in animal_sessions:
        avg = session.get('avg_first_lick_time')
        if avg is None and 'first_lick_times_by_reward_size' in session:
            session_licks = next((licks for d, licks in zip(days, first_lick_times_by_session) if d == session['session_day']), [])
            if session_licks:
                avg = np.mean(session_licks) if session_licks else None
                session['avg_first_lick_time'] = avg
                session['std_first_lick_time'] = np.std(session_licks) if session_licks else 0
                session['sem_first_lick_time'] = sem(session_licks) if len(session_licks) > 1 else 0
            else:
                 session['avg_first_lick_time'] = None
                 session['std_first_lick_time'] = 0
                 session['sem_first_lick_time'] = 0
        avg_lick_times.append(session.get('avg_first_lick_time'))
        std_lick_times.append(session.get('std_first_lick_time', 0))
        sem_lick_times.append(session.get('sem_first_lick_time', 0))

    avg_lick_times = [t if t is not None else np.nan for t in avg_lick_times]
    sem_lick_times = [t if t is not None else 0 for t in sem_lick_times]

    # --- FIX: Use the passed argument directly for the boundary ---
    # The old logic is unreliable. If the day is provided, its position is just day + 0.5
    boundary_pos = None
    if training_test_boundary_day is not None and training_test_boundary_day > 0:
        boundary_pos = training_test_boundary_day + 0.5
    # --- End of FIX ---

    first_stress_day = None
    stress_boundary_pos = None
    for session in animal_sessions:
        if session.get('session_info', {}).get('stress_status', False):
            first_stress_day = session['session_day']
            break
    if first_stress_day is not None:
        stress_boundary_pos = first_stress_day - 0.5
        
    # --- Calculate Phase Boundary Positions --- 
    pl37_line_pos = pl37_vline_day if pl37_vline_day is not None else None
    post_pl37_boundary_pos = post_pl37_vline_day - 0.5 if post_pl37_vline_day is not None else None
    # --------------------------------------------------------

    # Create figure
    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Performance over days (ax1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(days, performance, 'o-', color='blue', markersize=10, linewidth=2)
    ax1.set_xlabel('Session Day', fontsize=12)
    ax1.set_ylabel('Performance (%)', fontsize=12)
    ax1.set_title(f'{animal_id}: Performance Across Sessions', fontsize=14)
    ax1.set_ylim(0, 105)
    ax1.set_xticks(days)
    ax1.tick_params(axis='x', labelsize=8) # Reduce x-tick label size
    ax1.grid(True, alpha=0.3)

    boundary_label_added = False
    stress_label_added = False
    pl37_label_added = False # NEW
    post_pl37_label_added = False # NEW
    
    # Draw boundary lines
    if boundary_pos is not None:
        ax1.axvline(x=boundary_pos, color='red', linestyle='--', alpha=0.7, label='Training/Test Boundary')
        boundary_label_added = True
    if stress_boundary_pos is not None:
        ax1.axvline(x=stress_boundary_pos, color='blue', linestyle=':', alpha=0.7, label='Stress Phase Start')
        stress_label_added = True
    # --- NEW: Draw phase boundary lines --- 
    if pl37_line_pos is not None:
        ax1.axvline(x=pl37_line_pos, color='green', linestyle='-.', alpha=0.7, label='PL37')
        pl37_label_added = True
    if post_pl37_boundary_pos is not None:
        ax1.axvline(x=post_pl37_boundary_pos, color='purple', linestyle=':', alpha=0.7, label='Stress Phase End')
        post_pl37_label_added = True
    # -------------------------------------

    # Comment out performance annotations
    # for i, perf in enumerate(performance):
    #     ax1.annotate(f"{perf:.1f}%", (days[i], performance[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)

    # --- Plots 2, 3, 4, 5 remain the same as original function --- 
    # 2. Trial counts over days (ax2)
    ax2 = fig.add_subplot(gs[0, 1])
    width = 0.35
    x = np.array(days)
    ax2.bar(x - width/2, total_trials, width, label='Total Trials', color='skyblue')
    ax2.bar(x + width/2, rewarded_trials, width, label='Rewarded Trials', color='green')
    ax2.set_xlabel('Session Day', fontsize=12)
    ax2.set_ylabel('Number of Trials', fontsize=12)
    ax2.set_title(f'{animal_id}: Trial Counts Across Sessions', fontsize=14)
    ax2.set_xticks(days)
    ax2.tick_params(axis='x', labelsize=8) # Reduce x-tick label size
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    if boundary_pos is not None: ax2.axvline(x=boundary_pos, color='red', linestyle='--', alpha=0.7, label='_nolegend_')
    if stress_boundary_pos is not None: ax2.axvline(x=stress_boundary_pos, color='blue', linestyle=':', alpha=0.7, label='_nolegend_')
    # --- NEW: Draw phase boundary lines (no legend) --- 
    if pl37_line_pos is not None: ax2.axvline(x=pl37_line_pos, color='green', linestyle='-.', alpha=0.7, label='_nolegend_')
    if post_pl37_boundary_pos is not None: ax2.axvline(x=post_pl37_boundary_pos, color='purple', linestyle=':', alpha=0.7, label='_nolegend_')
    # --------------------------------------------------
    
    # Comment out trial count annotations
    # for i, (total, rewarded) in enumerate(zip(total_trials, rewarded_trials)):
    #     ax2.annotate(f"{total}", (days[i] - width/2, total), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=9)
    #     ax2.annotate(f"{rewarded}", (days[i] + width/2, rewarded), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=9)

    # 3. Average first lick time over days (ax3)
    ax3 = fig.add_subplot(gs[1, 0])
    valid_indices = [i for i, t in enumerate(avg_lick_times) if not np.isnan(t)]
    if valid_indices:
         sem_to_plot = np.array(sem_lick_times)[valid_indices]
         ax3.errorbar(np.array(days)[valid_indices], np.array(avg_lick_times)[valid_indices], yerr=sem_to_plot, fmt='o-', color='purple',
                     markersize=8, capsize=5, linewidth=2, elinewidth=1.5)
    elif len(days) > 0:
         ax3.plot(days, avg_lick_times, 'o-', color='purple', markersize=8, linewidth=2)
    ax3.set_xlabel('Session Day', fontsize=12)
    ax3.set_ylabel('First Lick Time (s)', fontsize=12)
    ax3.set_title(f'{animal_id}: Average First Lick Time Across Sessions (Error bars: SEM)', fontsize=14)
    ax3.set_xticks(days)
    ax3.tick_params(axis='x', labelsize=8) # Reduce x-tick label size
    ax3.grid(True, alpha=0.3)
    if boundary_pos is not None: ax3.axvline(x=boundary_pos, color='red', linestyle='--', alpha=0.7, label='_nolegend_')
    if stress_boundary_pos is not None: ax3.axvline(x=stress_boundary_pos, color='blue', linestyle=':', alpha=0.7, label='_nolegend_')
    # --- NEW: Draw phase boundary lines (no legend) --- 
    if pl37_line_pos is not None: ax3.axvline(x=pl37_line_pos, color='green', linestyle='-.', alpha=0.7, label='_nolegend_')
    if post_pl37_boundary_pos is not None: ax3.axvline(x=post_pl37_boundary_pos, color='purple', linestyle=':', alpha=0.7, label='_nolegend_')
    # --------------------------------------------------
    
    # Comment out average lick time annotations
    # for i, avg in enumerate(avg_lick_times):
    #     if not np.isnan(avg):
    #         ax3.annotate(f"{avg:.2f}s", (days[i], avg), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)

    # 4. Box plot of first lick times by session (ax4)
    ax4 = fig.add_subplot(gs[1, 1])
    plot_lick_times = [times for times in first_lick_times_by_session if times]
    plot_days = [day for day, times in zip(days, first_lick_times_by_session) if times]
    if plot_lick_times:
        box_parts = ax4.boxplot(plot_lick_times, positions=plot_days, patch_artist=True, widths=0.6, showfliers=False)
        for box in box_parts['boxes']: box.set(facecolor='lightblue', alpha=0.8)
        for median in box_parts['medians']: median.set(color='navy', linewidth=2)
        for i, lick_times in enumerate(plot_lick_times):
            if lick_times:
                jitter = np.random.normal(0, 0.05, size=len(lick_times))
                ax4.scatter(np.array([plot_days[i]] * len(lick_times)) + jitter, lick_times, alpha=0.5, s=30, color='darkblue')
    else:
        ax4.text(0.5, 0.5, 'No lick time data available', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
    ax4.set_xlabel('Session Day', fontsize=12)
    ax4.set_ylabel('First Lick Time (s)', fontsize=12)
    ax4.set_title(f'{animal_id}: Distribution of First Lick Times', fontsize=14)
    ax4.set_xticks(days) # Set ticks for all days, even if no boxplot
    ax4.tick_params(axis='x', labelsize=8) # Reduce x-tick label size
    ax4.grid(True, alpha=0.3)
    if boundary_pos is not None: ax4.axvline(x=boundary_pos, color='red', linestyle='--', alpha=0.7, label='_nolegend_')
    if stress_boundary_pos is not None: ax4.axvline(x=stress_boundary_pos, color='blue', linestyle=':', alpha=0.7, label='_nolegend_')
    # --- NEW: Draw phase boundary lines (no legend) --- 
    if pl37_line_pos is not None: ax4.axvline(x=pl37_line_pos, color='green', linestyle='-.', alpha=0.7, label='_nolegend_')
    if post_pl37_boundary_pos is not None: ax4.axvline(x=post_pl37_boundary_pos, color='purple', linestyle=':', alpha=0.7, label='_nolegend_')
    # --------------------------------------------------

    # 5. Heatmap of first lick time distribution by session (ax5)
    ax5 = fig.add_subplot(gs[2, :])
    all_licks = [lick for session_licks in first_lick_times_by_session for lick in session_licks]
    max_lick_time = max(all_licks) if all_licks else 1
    bins = np.linspace(0, min(max_lick_time, 10), 50)
    heatmap_data = np.zeros((len(days), len(bins)-1))
    for i, lick_times in enumerate(first_lick_times_by_session):
        if lick_times:
            hist, _ = np.histogram(lick_times, bins=bins)
            if len(lick_times) > 0: heatmap_data[i, :] = hist / len(lick_times)
    im = ax5.imshow(heatmap_data, aspect='auto', cmap='viridis', extent=[bins[0], bins[-1], days[-1] + 0.5, days[0] - 0.5])
    cbar = plt.colorbar(im, ax=ax5); cbar.set_label('Proportion of First Licks', fontsize=12)
    ax5.set_xlabel('First Lick Time (s)', fontsize=12)
    ax5.set_ylabel('Session Day', fontsize=12)
    ax5.set_title(f'{animal_id}: First Lick Time Distribution Across Sessions', fontsize=14)
    ax5.set_yticks(days)
    ax5.tick_params(axis='y', labelsize=8) # Reduce y-tick label size (days)
    
    # Remove vertical white lines
    # for t in range(1, int(bins[-1]) + 1): ax5.axvline(x=t, color='white', linestyle='--', alpha=0.3)
    
    ax5.grid(False) # Explicitly disable grid lines for the heatmap
    
    # --- Draw horizontal boundary lines on heatmap --- 
    if boundary_pos is not None: ax5.axhline(y=boundary_pos, color='red', linestyle='--', alpha=0.7, label='_nolegend_')
    if stress_boundary_pos is not None: ax5.axhline(y=stress_boundary_pos, color='blue', linestyle=':', alpha=0.7, label='_nolegend_')
    if pl37_line_pos is not None: ax5.axhline(y=pl37_line_pos, color='green', linestyle='-.', alpha=0.7, label='_nolegend_')
    if post_pl37_boundary_pos is not None: ax5.axhline(y=post_pl37_boundary_pos, color='purple', linestyle=':', alpha=0.7, label='_nolegend_')
    # -------------------------------------------------

    # --- Consolidate legend (ONLY for boundary lines) --- 
    handles, labels = [], []
    # Get handles/labels ONLY from ax1 where boundary lines were defined with labels
    h, l = ax1.get_legend_handles_labels()
    boundary_labels_ordered = ['Training/Test Boundary', 'Stress Phase Start', 'PL37', 'Stress Phase End'] # Updated labels
    # Filter handles and labels to keep only the desired boundary lines in the correct order
    ordered_handles = []
    ordered_labels = []
    for label_name in boundary_labels_ordered:
        try:
            idx = l.index(label_name)
            ordered_handles.append(h[idx])
            ordered_labels.append(l[idx])
        except ValueError:
            pass # Label not found (e.g., if a boundary wasn't plotted)

    # --- Create legend with smaller font --- 
    if ordered_handles:
         fig.legend(ordered_handles, ordered_labels, 
                    loc='upper right', 
                    bbox_to_anchor=(0.98, 0.98), # Keep position for now
                    fontsize='small') # Reduce font size
    
    plt.suptitle(f'Performance Analysis for {animal_id}', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96]) # Adjust right margin if needed

    filename = f"{animal_id}_performance_analysis_phase_aware.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved phase-aware performance figure: {filepath}")
    plt.close(fig)
    
    # --- Call lick sensor cut time visualization (unchanged, uses original function) --- 
    # This assumes the original visualization module is still available for this specific plot
    try:
        from .visualization import visualize_lick_sensor_cut_time # Using relative import
        has_lick_cut_data = any('lick_sensor_cut_analysis' in s for s in animal_sessions)
        if has_lick_cut_data:
            print("  Calling visualize_lick_sensor_cut_time with phase boundaries...")
            lick_cut_plot_path = visualize_lick_sensor_cut_time(
                animal_sessions,
                animal_id,
                output_dir,
                training_test_boundary_day=training_test_boundary_day,
                pl37_vline_day=pl37_vline_day,
                post_pl37_vline_day=post_pl37_vline_day
            )
        else:
            print("  No lick sensor cut analysis data found in any session for original plot.")
            lick_cut_plot_path = None
    except ImportError:
        print("Warning: Could not import original visualize_lick_sensor_cut_time from visualization.py.")
        lick_cut_plot_path = None
    except Exception as e:
        print(f"Error calling original visualize_lick_sensor_cut_time: {e}")
        lick_cut_plot_path = None
    # -------------------------------------------------------------------------------------
    
    # --- FIX: Return a list of all generated plot paths ---
    generated_paths = [filepath]
    if lick_cut_plot_path:
        generated_paths.append(lick_cut_plot_path)

    return generated_paths


def visualize_cross_session_test_analysis_phase_aware(test_sessions, animal_id, output_dir):
    """
    Generates a 2-panel summary plot comparing latency and lick sensor cut duration 
    across test phases (Initial, Stress Pre-PL37, PL37, Post-PL37), 
    grouped by the reward size of the *previous* trial.

    Uses grouped bar plots (Mean +/- SEM) for comparison.

    Parameters:
    -----------
    test_sessions : list
        A list of dictionaries, each containing processed data for a test session,
        including 'trial_results', 'session_day', and 'phase'.
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
        print("No test sessions provided for phase-aware cross-session summary.")
        return None

    print(f"\nGenerating Phase-Aware Cross-Session Analysis plot for {animal_id}...")
    start_time = time.time()

    # Define reward size labels (consistent with other plots)
    size_labels_map = {1: 'Very Small', 2: 'Small', 3: 'Medium', 4: 'Large', 5: 'Very Large'}
    reward_sizes = list(size_labels_map.keys())

    # --- Data Aggregation by Phase and Previous Reward Size --- 
    # Structure: phase_data[phase][prev_reward_size][metric] = [values]
    phase_data = {phase: {prev_size: {'latencies': [], 'durations': []} 
                            for prev_size in reward_sizes} 
                    for phase in PHASE_ORDER}
    
    sessions_with_phase = 0
    trials_processed = 0
    for session in test_sessions:
        session_phase = session.get('phase')
        if not session_phase or session_phase not in PHASE_ORDER:
            # print(f"Warning: Session Day {session.get('session_day')} missing valid phase. Skipping.")
            continue
        sessions_with_phase += 1

        trial_results = session.get('trial_results', [])
        if not trial_results or len(trial_results) < 2:
            continue
        
        # Group data by previous reward size for this session
        for i in range(1, len(trial_results)):
            prev_trial = trial_results[i-1]
            current_trial = trial_results[i]

            if prev_trial.get('rewarded', False):
                prev_size = prev_trial.get('reward_size')
                if prev_size in reward_sizes:
                    # Get current trial's latency 
                    latency = current_trial.get('first_lick_latency')
                    if latency is not None and not np.isnan(latency):
                         phase_data[session_phase][prev_size]['latencies'].append(latency)
                         trials_processed += 1 # Count valid trials
                    
                    # Get current trial's lick cut duration
                    cut_duration = None
                    lick_analysis_data = session.get('lick_sensor_cut_analysis', {})
                    if 'trial_metrics' in lick_analysis_data:
                        metrics = lick_analysis_data['trial_metrics']
                        current_trial_num = current_trial.get('trial_num')
                        metric = next((m for m in metrics if m.get('trial_num') == current_trial_num), None)
                        if metric and 'post_cue_cut_duration' in metric:
                            cut_duration = metric['post_cue_cut_duration']
                    elif 'post_cue_cut_duration' in current_trial:
                        cut_duration = current_trial['post_cue_cut_duration']

                    if cut_duration is not None and not np.isnan(cut_duration):
                        phase_data[session_phase][prev_size]['durations'].append(cut_duration)
                        # Assuming duration processing implies latency was also processed if valid

    print(f"  Aggregated data from {sessions_with_phase} sessions with phases, {trials_processed} valid trials.")

    # --- Convert aggregated data to DataFrame for Seaborn plotting --- 
    plot_df_list = []
    for phase in PHASE_ORDER:
        for prev_size in reward_sizes:
            latencies = phase_data[phase][prev_size]['latencies']
            durations = phase_data[phase][prev_size]['durations']
            
            mean_latency = np.mean(latencies) if latencies else np.nan
            sem_latency = sem(latencies) if len(latencies) > 1 else 0
            n_latency = len(latencies)
            
            mean_duration = np.mean(durations) if durations else np.nan
            sem_duration = sem(durations) if len(durations) > 1 else 0
            n_duration = len(durations)
            
            # Only add row if there's data for either metric
            if n_latency > 0 or n_duration > 0:
                plot_df_list.append({
                    'Phase': phase,
                    'Previous Reward Size': prev_size,
                    'Mean Latency': mean_latency,
                    'SEM Latency': sem_latency,
                    'N Latency': n_latency,
                    'Mean Duration': mean_duration,
                    'SEM Duration': sem_duration,
                    'N Duration': n_duration
                })

    if not plot_df_list:
         print("  No valid data points found after aggregation. Cannot generate plot.")
         return None
         
    plot_df = pd.DataFrame(plot_df_list)
    # Convert Phase to categorical for correct ordering and coloring
    plot_df['Phase'] = pd.Categorical(plot_df['Phase'], categories=PHASE_ORDER, ordered=True)

    # --- Plotting --- 
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True) # Two panels: Latency, Duration
    fig.suptitle(f'{animal_id} - Behavior Summary by Phase and Previous Reward Size', fontsize=16, y=0.98)
    
    plot_generated = False

    # Panel 0: Latency
    ax_lat = axes[0]
    if not plot_df['Mean Latency'].dropna().empty:
        plot_generated = True
        sns.barplot(data=plot_df, x='Previous Reward Size', y='Mean Latency', 
                    hue='Phase', palette=PHASE_COLORS, ax=ax_lat, 
                    order=reward_sizes, hue_order=PHASE_ORDER, 
                    errorbar=None) # We'll add SEM manually for clarity
        
        # Add SEM error bars manually
        bar_width = 0.8 / len(PHASE_ORDER) # Approximate width of each bar in a group
        num_rewards = len(reward_sizes)
        for i, phase in enumerate(PHASE_ORDER):
            phase_df = plot_df[plot_df['Phase'] == phase]
            # Calculate offset for this phase within the group
            offset = (i - (len(PHASE_ORDER) - 1) / 2) * bar_width
            x_coords = np.arange(num_rewards) + offset
            ax_lat.errorbar(x=x_coords, y=phase_df['Mean Latency'], 
                            yerr=phase_df['SEM Latency'], fmt='none', c='black', capsize=3)

        ax_lat.set_title("Mean First Lick Latency (+/- SEM)", fontsize=12)
        ax_lat.set_ylabel("Latency (s)")
        ax_lat.grid(True, axis='y', alpha=0.5, linestyle='--')
        ax_lat.legend(title='Phase', loc='upper right', fontsize='small')
    else:
        ax_lat.text(0.5, 0.5, 'No latency data', ha='center', va='center', transform=ax_lat.transAxes, color='gray')
    ax_lat.set_xlabel("") # Remove x-label for top plot

    # Panel 1: Duration
    ax_dur = axes[1]
    if not plot_df['Mean Duration'].dropna().empty:
        plot_generated = True
        sns.barplot(data=plot_df, x='Previous Reward Size', y='Mean Duration', 
                    hue='Phase', palette=PHASE_COLORS, ax=ax_dur,
                    order=reward_sizes, hue_order=PHASE_ORDER,
                    errorbar=None)
        
        # Add SEM error bars manually
        for i, phase in enumerate(PHASE_ORDER):
            phase_df = plot_df[plot_df['Phase'] == phase]
            offset = (i - (len(PHASE_ORDER) - 1) / 2) * bar_width
            x_coords = np.arange(num_rewards) + offset
            ax_dur.errorbar(x=x_coords, y=phase_df['Mean Duration'], 
                            yerr=phase_df['SEM Duration'], fmt='none', c='black', capsize=3)

        ax_dur.set_title("Mean Lick Cut Duration (+/- SEM)", fontsize=12)
        ax_dur.set_ylabel("Duration (s)")
        ax_dur.grid(True, axis='y', alpha=0.5, linestyle='--')
        ax_dur.axhline(y=1.5, color='gray', linestyle=':', alpha=0.7, linewidth=1) # Max duration line
        # Only show legend on top plot for less clutter
        if ax_dur.get_legend() is not None: ax_dur.get_legend().remove()
        ax_dur.set_ylim(bottom=0) # Ensure y-axis starts at 0 for duration
    else:
        ax_dur.text(0.5, 0.5, 'No duration data', ha='center', va='center', transform=ax_dur.transAxes, color='gray')
    
    # Set reward size labels on x-axis for bottom plot
    reward_size_str_labels = [size_labels_map.get(s, str(s)) for s in reward_sizes]
    ax_dur.set_xlabel("Previous Trial Reward Size")
    ax_dur.set_xticklabels(reward_size_str_labels)

    # Final layout adjustment
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # --- Save Figure --- 
    if not plot_generated:
        print("  No data plotted for any phase/reward combination. Skipping save.")
        plt.close(fig)
        return None
        
    output_filename = os.path.join(output_dir, f"{animal_id}_phase_aware_cross_session_analysis.png")
    try:
        plt.savefig(output_filename, dpi=150)
        print(f"  Saved phase-aware cross-session analysis figure: {output_filename}")
    except Exception as e:
        print(f"  ERROR saving phase-aware cross-session analysis figure: {e}")
        output_filename = None
    finally:
        plt.close(fig)
        
    print(f"  Finished phase-aware cross-session analysis plot generation ({time.time() - start_time:.2f} seconds).")
    return output_filename


def visualize_stress_comparison_by_prev_reward_phase_aware(test_sessions, animal_id, output_dir):
    """Generates a 5x2 plot comparing Initial, Stress Pre-PL37, and Post-PL37 phases 
       for latency and duration, grouped by the previous trial's reward size.
       Uses violin plots with individual points and includes pairwise statistical tests.
    """
    if not test_sessions:
        print("No test sessions provided for phase stress comparison.")
        return None

    print(f"\nGenerating Phase Stress Comparison plot (by Prev Reward) for {animal_id}...")
    start_time = time.time()

    # --- Filter sessions to only include the comparison phases --- 
    comparison_sessions = [s for s in test_sessions if s.get('phase') in PHASE_ORDER_COMPARISON]
    if not comparison_sessions:
        print("  No sessions found belonging to Initial Test, Stress Test (Pre-PL37), or Post-PL37 Test phases. Cannot generate comparison plot.")
        return None
        
    num_initial = sum(1 for s in comparison_sessions if s.get('phase') == "Initial Test")
    num_stress_pre = sum(1 for s in comparison_sessions if s.get('phase') == "Stress Test (Pre-PL37)")
    num_post = sum(1 for s in comparison_sessions if s.get('phase') == "Post-PL37 Test")
    print(f"  Using data from {len(comparison_sessions)} sessions: Initial ({num_initial}), Stress Pre-PL37 ({num_stress_pre}), Post-PL37 ({num_post})")

    # --- Analysis Parameters & Setup --- 
    size_labels_map = {1: 'VS', 2: 'S', 3: 'M', 4: 'L', 5: 'VL'}
    ALL_REWARD_SIZES = list(size_labels_map.keys())
    min_data_points_for_test = 3 # Min points per group for stats

    # --- Data Pooling & Conversion to DataFrame --- 
    plot_df_list = []
    all_latencies_overall = []
    all_durations_overall = []

    for session in comparison_sessions:
        session_phase = session.get('phase') # Already filtered, should be valid
        trial_results = session.get('trial_results', [])
        
        if not trial_results or len(trial_results) < 2:
            continue
            
        for i in range(1, len(trial_results)):
            prev_trial = trial_results[i-1]
            current_trial = trial_results[i]

            if prev_trial.get('rewarded', False):
                prev_size = prev_trial.get('reward_size')
                if prev_size in ALL_REWARD_SIZES:
                    latency = current_trial.get('first_lick_latency')
                    cut_duration = None
                    lick_analysis_data = session.get('lick_sensor_cut_analysis', {})
                    if 'trial_metrics' in lick_analysis_data:
                        metrics = lick_analysis_data['trial_metrics']
                        current_trial_num = current_trial.get('trial_num')
                        metric = next((m for m in metrics if m.get('trial_num') == current_trial_num), None)
                        if metric and 'post_cue_cut_duration' in metric: cut_duration = metric['post_cue_cut_duration']
                    elif 'post_cue_cut_duration' in current_trial: cut_duration = current_trial['post_cue_cut_duration']
                    
                    latency_val = latency if (latency is not None and not np.isnan(latency)) else np.nan
                    duration_val = cut_duration if (cut_duration is not None and not np.isnan(cut_duration)) else np.nan
                    
                    if not np.isnan(latency_val): all_latencies_overall.append(latency_val)
                    if not np.isnan(duration_val): all_durations_overall.append(duration_val)
                    
                    plot_df_list.append({
                        'prev_reward_size': prev_size,
                        'Phase': session_phase, # Use Phase column name
                        'latency': latency_val,
                        'duration': duration_val
                    })
                    
    if not plot_df_list:
         print("  No valid data points found after pooling. Cannot generate plot.")
         return None
         
    plot_df = pd.DataFrame(plot_df_list)
    # Set Phase as categorical with the specific comparison order
    plot_df['Phase'] = pd.Categorical(plot_df['Phase'], categories=PHASE_ORDER_COMPARISON, ordered=True)

    # --- Plotting --- 
    fig, axs = plt.subplots(len(ALL_REWARD_SIZES), 2, 
                             figsize=(11, 15), # Adjusted figsize for 3 groups
                             sharex='col', sharey='col') 
    if len(ALL_REWARD_SIZES) == 1: axs = np.array([axs]) # Ensure axs is 2D
        
    fig.suptitle(f"{animal_id} - Phase Comparison by Previous Reward Size", fontsize=14, y=0.99)

    plot_generated = False
    stats_results = {} # Store p-values for annotation

    for i, prev_size in enumerate(ALL_REWARD_SIZES):
        ax_lat = axs[i, 0]
        ax_dur = axs[i, 1]
        prev_label = size_labels_map[prev_size]
        row_title = f"Previous: {prev_label} ({prev_size})"
        ax_lat.set_ylabel(row_title, fontsize=9)
        ax_dur.set_ylabel("", fontsize=9) # Clear duration ylabel
        
        for ax in [ax_lat, ax_dur]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', size=6, width=1.5, labelsize=9)

        df_subset = plot_df[plot_df['prev_reward_size'] == prev_size].dropna(subset=['Phase'])
        
        # --- Latency Plot (Left Column) --- 
        metric = 'latency'
        if not df_subset[metric].dropna().empty:
            plot_generated = True
            sns.violinplot(x='Phase', y=metric, data=df_subset, order=PHASE_ORDER_COMPARISON, 
                           hue='Phase', palette=PHASE_COLORS_COMPARISON, 
                           inner=None, linewidth=1.5, ax=ax_lat, legend=False)
            sns.stripplot(x='Phase', y=metric, data=df_subset, order=PHASE_ORDER_COMPARISON, 
                          color='.25', size=3, alpha=0.5, jitter=0.15, ax=ax_lat, legend=False)
            
            # Perform Stats (Pairwise Mann-Whitney U)
            stats_key = (i, 0)
            stats_results[stats_key] = {}
            groups = {phase: df_subset[df_subset['Phase'] == phase][metric].dropna().tolist() 
                      for phase in PHASE_ORDER_COMPARISON}
            pairs = [(PHASE_ORDER_COMPARISON[0], PHASE_ORDER_COMPARISON[1]),
                     (PHASE_ORDER_COMPARISON[1], PHASE_ORDER_COMPARISON[2]),
                     (PHASE_ORDER_COMPARISON[0], PHASE_ORDER_COMPARISON[2])]
            for p1, p2 in pairs:
                g1_data, g2_data = groups[p1], groups[p2]
                p_val = np.nan
                if len(g1_data) >= min_data_points_for_test and len(g2_data) >= min_data_points_for_test:
                    try: _, p = mannwhitneyu(g1_data, g2_data, alternative='two-sided'); p_val = p
                    except ValueError: p_val = 1.0 # Handle identical data
                stats_results[stats_key][(p1, p2)] = p_val
        else:
            ax_lat.text(0.5, 0.5, 'No latency data', ha='center', va='center', transform=ax_lat.transAxes, color='gray', fontsize=9)
        ax_lat.set_xlabel("")

        # --- Duration Plot (Right Column) --- 
        metric = 'duration'
        if not df_subset[metric].dropna().empty:
            plot_generated = True
            sns.violinplot(x='Phase', y=metric, data=df_subset, order=PHASE_ORDER_COMPARISON, 
                           hue='Phase', palette=PHASE_COLORS_COMPARISON, 
                           inner=None, linewidth=1.5, ax=ax_dur, legend=False)
            sns.stripplot(x='Phase', y=metric, data=df_subset, order=PHASE_ORDER_COMPARISON, 
                          color='.25', size=3, alpha=0.5, jitter=0.15, ax=ax_dur, legend=False)
            ax_dur.axhline(y=1.5, color='gray', linestyle=':', alpha=0.7, linewidth=1) 

            # Perform Stats (Pairwise Mann-Whitney U)
            stats_key = (i, 1)
            stats_results[stats_key] = {}
            groups = {phase: df_subset[df_subset['Phase'] == phase][metric].dropna().tolist() 
                      for phase in PHASE_ORDER_COMPARISON}
            for p1, p2 in pairs:
                g1_data, g2_data = groups[p1], groups[p2]
                p_val = np.nan
                if len(g1_data) >= min_data_points_for_test and len(g2_data) >= min_data_points_for_test:
                    try: _, p = mannwhitneyu(g1_data, g2_data, alternative='two-sided'); p_val = p
                    except ValueError: p_val = 1.0
                stats_results[stats_key][(p1, p2)] = p_val
        else:
            ax_dur.text(0.5, 0.5, 'No duration data', ha='center', va='center', transform=ax_dur.transAxes, color='gray', fontsize=9)
        ax_dur.set_xlabel("")
            
        ax_lat.grid(True, axis='y', alpha=0.5, linestyle='--')
        ax_dur.grid(True, axis='y', alpha=0.5, linestyle='--')

        # Set x-tick labels only for the bottom row
        if i == len(ALL_REWARD_SIZES) - 1:
            phase_labels_short = [p.replace(" Test", "").replace(" (Pre-PL37)","-Pre").replace("Post-PL37","-Post") for p in PHASE_ORDER_COMPARISON]
            ax_lat.set_xticklabels(phase_labels_short, rotation=45, ha='right')
            ax_dur.set_xticklabels(phase_labels_short, rotation=45, ha='right')
        else:
             ax_lat.set_xticklabels([])
             ax_dur.set_xticklabels([])
            
        if i == 0:
            ax_lat.set_title("Latency (s)", fontsize=11)
            ax_dur.set_title("Duration (s)", fontsize=11)
        else:
            ax_lat.set_title(""); ax_dur.set_title("")

    # --- Set shared Y-limits --- 
    if all_latencies_overall:
        lat_min, lat_max = np.percentile(all_latencies_overall, [1, 99])
        lat_pad = (lat_max - lat_min) * 0.1 if lat_max > lat_min else 0.1
        plt.setp(axs[:, 0], ylim=(max(0, lat_min - lat_pad * 2), max(1.0, lat_max + lat_pad * 2))) # Extra padding for stats
    else: plt.setp(axs[:, 0], ylim=(0, 1.5)) 
        
    if all_durations_overall:
        dur_min, dur_max = np.percentile(all_durations_overall, [1, 99])
        dur_pad = (dur_max - dur_min) * 0.1 if dur_max > dur_min else 0.1
        plt.setp(axs[:, 1], ylim=(max(0, dur_min - dur_pad * 2), min(1.55, max(0.1, dur_max + dur_pad * 2))))
    else: plt.setp(axs[:, 1], ylim=(0, 1.55))
        
    # --- Add Significance Annotations --- 
    def format_p(p): # Helper to format p-value
        if np.isnan(p): return "n.s."
        if p < 0.001: return "***" # p < 0.001
        if p < 0.01: return "**"   # p < 0.01
        if p < 0.05: return "*"    # p < 0.05
        return "n.s." # p >= 0.05
        
    for (row, col), p_vals_dict in stats_results.items():
        ax = axs[row, col]
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        y_step = y_range * 0.05 # Step size for multiple comparison lines
        y_start = ylim[1] * 0.85 # Start annotation lines higher up

        # Comparisons: (0,1), (1,2), (0,2) -> (Initial vs StressPre), (StressPre vs PostPL37), (Initial vs PostPL37)
        pairs_indices = [(0, 1), (1, 2), (0, 2)]
        p_vals = [p_vals_dict.get((PHASE_ORDER_COMPARISON[p1_idx], PHASE_ORDER_COMPARISON[p2_idx]), np.nan) 
                  for p1_idx, p2_idx in pairs_indices]
        
        line_y = y_start
        for idx, (p1_idx, p2_idx) in enumerate(pairs_indices):
            p_val = p_vals[idx]
            if not np.isnan(p_val) and p_val < 0.05: # Only draw line if significant
                x1, x2 = p1_idx, p2_idx
                line_y += y_step * idx # Increment y position for each line
                ax.plot([x1, x1, x2, x2], [line_y, line_y + y_step*0.2, line_y + y_step*0.2, line_y], lw=1.2, c='black')
                ax.text((x1 + x2) / 2, line_y + y_step*0.25, format_p(p_val), ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout(rect=[0.04, 0.05, 1, 0.97]) # Adjust layout

    # --- Save Figure --- 
    if not plot_generated:
        print("  No data generated for any comparison panel. Skipping save.")
        plt.close(fig)
        return None
        
    output_filename = os.path.join(output_dir, f"{animal_id}_phase_stress_comparison_by_prev_reward.png")
    try:
        plt.savefig(output_filename, dpi=150)
        print(f"  Saved phase stress comparison figure: {output_filename}")
    except Exception as e:
        print(f"  ERROR saving phase stress comparison figure: {e}")
        output_filename = None
    finally:
        plt.close(fig)
        
    print(f"  Finished phase stress comparison plot generation ({time.time() - start_time:.2f} seconds).")
    return output_filename


def visualize_stress_comparison_pooled_phase_aware(test_sessions, animal_id, output_dir):
    """Generates a 1x2 plot comparing Initial, Stress Pre-PL37, and Post-PL37 phases
       for latency and duration, pooling data across all previous reward sizes.
       Uses violin plots with individual points and includes pairwise statistical tests.
    """
    if not test_sessions:
        print("No test sessions provided for pooled phase stress comparison.")
        return None

    print(f"\nGenerating Pooled Phase Stress Comparison plot for {animal_id}...")
    start_time = time.time()

    # --- Filter sessions to only include the comparison phases --- 
    comparison_sessions = [s for s in test_sessions if s.get('phase') in PHASE_ORDER_COMPARISON]
    if not comparison_sessions:
        print("  No sessions found belonging to Initial Test, Stress Test (Pre-PL37), or Post-PL37 Test phases. Cannot generate pooled comparison plot.")
        return None
        
    num_initial = sum(1 for s in comparison_sessions if s.get('phase') == "Initial Test")
    num_stress_pre = sum(1 for s in comparison_sessions if s.get('phase') == "Stress Test (Pre-PL37)")
    num_post = sum(1 for s in comparison_sessions if s.get('phase') == "Post-PL37 Test")
    print(f"  Using data from {len(comparison_sessions)} sessions: Initial ({num_initial}), Stress Pre-PL37 ({num_stress_pre}), Post-PL37 ({num_post})")

    # --- Analysis Parameters & Setup --- 
    min_data_points_for_test = 3 # Min points per group for stats

    # --- Data Pooling & Conversion to DataFrame (Pooled across prev reward size) --- 
    plot_df_list = []
    all_latencies_overall = []
    all_durations_overall = []

    for session in comparison_sessions:
        session_phase = session.get('phase')
        trial_results = session.get('trial_results', [])
        
        if not trial_results or len(trial_results) < 2:
            continue
            
        for i in range(1, len(trial_results)):
            prev_trial = trial_results[i-1]
            current_trial = trial_results[i]

            # Only consider trials following a rewarded trial
            if prev_trial.get('rewarded', False):
                # Get metrics, ignoring previous reward size
                latency = current_trial.get('first_lick_latency')
                cut_duration = None
                lick_analysis_data = session.get('lick_sensor_cut_analysis', {})
                if 'trial_metrics' in lick_analysis_data:
                    metrics = lick_analysis_data['trial_metrics']
                    current_trial_num = current_trial.get('trial_num')
                    metric = next((m for m in metrics if m.get('trial_num') == current_trial_num), None)
                    if metric and 'post_cue_cut_duration' in metric: cut_duration = metric['post_cue_cut_duration']
                elif 'post_cue_cut_duration' in current_trial: cut_duration = current_trial['post_cue_cut_duration']
                
                latency_val = latency if (latency is not None and not np.isnan(latency)) else np.nan
                duration_val = cut_duration if (cut_duration is not None and not np.isnan(cut_duration)) else np.nan
                
                if not np.isnan(latency_val): all_latencies_overall.append(latency_val)
                if not np.isnan(duration_val): all_durations_overall.append(duration_val)
                
                plot_df_list.append({
                    'Phase': session_phase,
                    'latency': latency_val,
                    'duration': duration_val
                })
                    
    if not plot_df_list:
         print("  No valid data points found after pooling. Cannot generate plot.")
         return None
         
    plot_df = pd.DataFrame(plot_df_list)
    plot_df['Phase'] = pd.Categorical(plot_df['Phase'], categories=PHASE_ORDER_COMPARISON, ordered=True)

    # --- Plotting --- 
    fig, axs = plt.subplots(1, 2, figsize=(9, 6)) # Adjusted figsize for 1x2
    fig.suptitle(f"{animal_id} - Pooled Phase Comparison", fontsize=14, y=0.99)

    plot_generated = False
    stats_results = {} 

    # --- Panel 0: Latency --- 
    ax_lat = axs[0]
    metric = 'latency'
    if not plot_df[metric].dropna().empty:
        plot_generated = True
        sns.violinplot(x='Phase', y=metric, data=plot_df, order=PHASE_ORDER_COMPARISON, 
                       hue='Phase', palette=PHASE_COLORS_COMPARISON, inner=None, 
                       linewidth=1.5, ax=ax_lat, legend=False)
        sns.stripplot(x='Phase', y=metric, data=plot_df, order=PHASE_ORDER_COMPARISON, 
                      color='.25', size=3, alpha=0.5, jitter=0.15, ax=ax_lat, legend=False)
        # Stats
        stats_key = 0
        stats_results[stats_key] = {}
        groups = {phase: plot_df[plot_df['Phase'] == phase][metric].dropna().tolist() 
                  for phase in PHASE_ORDER_COMPARISON}
        pairs = [(PHASE_ORDER_COMPARISON[0], PHASE_ORDER_COMPARISON[1]),
                 (PHASE_ORDER_COMPARISON[1], PHASE_ORDER_COMPARISON[2]),
                 (PHASE_ORDER_COMPARISON[0], PHASE_ORDER_COMPARISON[2])]
        for p1, p2 in pairs:
            g1_data, g2_data = groups[p1], groups[p2]
            p_val = np.nan
            if len(g1_data) >= min_data_points_for_test and len(g2_data) >= min_data_points_for_test:
                try: _, p = mannwhitneyu(g1_data, g2_data, alternative='two-sided'); p_val = p
                except ValueError: p_val = 1.0
            stats_results[stats_key][(p1, p2)] = p_val
    else:
        ax_lat.text(0.5, 0.5, 'No latency data', ha='center', va='center', transform=ax_lat.transAxes, color='gray', fontsize=9)
    ax_lat.set_title("Latency (s)", fontsize=11)
    ax_lat.set_ylabel("Latency (s)", fontsize=10)
    ax_lat.grid(True, axis='y', alpha=0.5, linestyle='--')

    # --- Panel 1: Duration --- 
    ax_dur = axs[1]
    metric = 'duration'
    if not plot_df[metric].dropna().empty:
        plot_generated = True
        sns.violinplot(x='Phase', y=metric, data=plot_df, order=PHASE_ORDER_COMPARISON, 
                       hue='Phase', palette=PHASE_COLORS_COMPARISON, inner=None, 
                       linewidth=1.5, ax=ax_dur, legend=False)
        sns.stripplot(x='Phase', y=metric, data=plot_df, order=PHASE_ORDER_COMPARISON, 
                      color='.25', size=3, alpha=0.5, jitter=0.15, ax=ax_dur, legend=False)
        ax_dur.axhline(y=1.5, color='gray', linestyle=':', alpha=0.7, linewidth=1) 
        # Stats
        stats_key = 1
        stats_results[stats_key] = {}
        groups = {phase: plot_df[plot_df['Phase'] == phase][metric].dropna().tolist() 
                  for phase in PHASE_ORDER_COMPARISON}
        # pairs defined above
        for p1, p2 in pairs:
            g1_data, g2_data = groups[p1], groups[p2]
            p_val = np.nan
            if len(g1_data) >= min_data_points_for_test and len(g2_data) >= min_data_points_for_test:
                try: _, p = mannwhitneyu(g1_data, g2_data, alternative='two-sided'); p_val = p
                except ValueError: p_val = 1.0
            stats_results[stats_key][(p1, p2)] = p_val
    else:
        ax_dur.text(0.5, 0.5, 'No duration data', ha='center', va='center', transform=ax_dur.transAxes, color='gray', fontsize=9)
        ax.set_ylim(-0.05, 1.05) 
    ax_dur.set_title("Duration (s)", fontsize=11)
    ax_dur.set_ylabel("Duration (s)", fontsize=10)
    ax_dur.grid(True, axis='y', alpha=0.5, linestyle='--')

    # --- Set shared Y-limits --- 
    if all_latencies_overall:
        lat_min, lat_max = np.percentile(all_latencies_overall, [1, 99])
        lat_pad = (lat_max - lat_min) * 0.1 if lat_max > lat_min else 0.1
        ax_lat.set_ylim(max(0, lat_min - lat_pad * 2), max(1.0, lat_max + lat_pad * 2))
    else: ax_lat.set_ylim(0, 1.5) 
        
    if all_durations_overall:
        dur_min, dur_max = np.percentile(all_durations_overall, [1, 99])
        dur_pad = (dur_max - dur_min) * 0.1 if dur_max > dur_min else 0.1
        ax_dur.set_ylim(max(0, dur_min - dur_pad * 2), min(1.55, max(0.1, dur_max + dur_pad * 2)))
    else: ax_dur.set_ylim(0, 1.55)

    # --- Add Significance Annotations --- 
    def format_p(p): # Helper defined earlier
        if np.isnan(p): return "n.s."
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "n.s."

    for col, p_vals_dict in stats_results.items():
        ax = axs[col]
        ylim = ax.get_ylim(); y_range = ylim[1] - ylim[0]
        y_step = y_range * 0.05; y_start = ylim[1] * 0.85
        pairs_indices = [(0, 1), (1, 2), (0, 2)]
        p_vals = [p_vals_dict.get((PHASE_ORDER_COMPARISON[p1_idx], PHASE_ORDER_COMPARISON[p2_idx]), np.nan) 
                  for p1_idx, p2_idx in pairs_indices]
        line_y = y_start
        for idx, (p1_idx, p2_idx) in enumerate(pairs_indices):
            p_val = p_vals[idx]
            if not np.isnan(p_val) and p_val < 0.05:
                x1, x2 = p1_idx, p2_idx
                line_y += y_step * idx
                ax.plot([x1, x1, x2, x2], [line_y, line_y + y_step*0.2, line_y + y_step*0.2, line_y], lw=1.2, c='black')
                ax.text((x1 + x2) / 2, line_y + y_step*0.25, format_p(p_val), ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Set shared x-tick labels
    phase_labels_short = [p.replace(" Test", "").replace(" (Pre-PL37)","-Pre").replace("Post-PL37","-Post") for p in PHASE_ORDER_COMPARISON]
    for ax in axs:
        ax.set_xticklabels(phase_labels_short, rotation=45, ha='right')
        ax.set_xlabel("Phase")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', size=6, width=1.5, labelsize=9)
        ax.set_ylim(-0.05, 1.05) 
    plt.tight_layout(rect=[0.05, 0.1, 1, 0.96]) # Adjust layout

    # --- Save Figure --- 
    if not plot_generated:
        print("  No data generated for pooled comparison. Skipping save.")
        plt.close(fig)
        return None
        
    output_filename = os.path.join(output_dir, f"{animal_id}_phase_stress_comparison_pooled.png")
    try:
        plt.savefig(output_filename, dpi=150)
        print(f"  Saved pooled phase stress comparison figure: {output_filename}")
    except Exception as e:
        print(f"  ERROR saving pooled phase stress comparison figure: {e}")
        output_filename = None
    finally:
        plt.close(fig)
        
    print(f"  Finished pooled phase stress comparison plot generation ({time.time() - start_time:.2f} seconds).")
    return output_filename


def visualize_cross_session_time_resolved_licking_phase_aware(filtered_test_sessions, animal_id, output_dir):
    """Generates a 2x1 summary plot comparing time-resolved licking 
       between Initial Test and Stress Test (Pre-PL37) phases.
       Assumes input `filtered_test_sessions` contains only data from these two phases.
       
       Plots data grouped by current or previous reward size.
    """
    if not filtered_test_sessions:
        print("No filtered test sessions provided for time-resolved phase comparison.")
        return None

    print(f"\nGenerating Cross-Session Time-Resolved Licking (Initial vs Stress Pre-PL37) plot for {animal_id}...")
    start_time = time.time()

    # Identify the phases present in the filtered data (should be max 2)
    present_phases = sorted(list(set(s.get('phase') for s in filtered_test_sessions if s.get('phase'))))
    if len(present_phases) == 0:
        print("  No valid phases found in the filtered data. Cannot generate plot.")
        return None
    elif len(present_phases) > 2:
         print(f"  Warning: Expected only Initial and Stress Pre-PL37 phases, but found: {present_phases}. Proceeding with comparison.")
         # Ensure the comparison order is maintained if possible
         phases_to_compare = [p for p in ["Initial Test", "Stress Test (Pre-PL37)"] if p in present_phases]
         if not phases_to_compare: phases_to_compare = present_phases[:2] # Fallback
    else:
        phases_to_compare = present_phases
        
    print(f"  Comparing phases: {phases_to_compare}")
    PHASE_COMPARISON_COLORS = {phases_to_compare[0]: 'royalblue', 
                               phases_to_compare[1]: 'orangered'} if len(phases_to_compare) == 2 else {"Initial Test": 'royalblue'} # Fallback color

    # --- Analysis Parameters (Copied) --- 
    ANALYSIS_START_TIME_REL_REWARD = -2.0
    TOTAL_DURATION_POST_REWARD = 8.0 
    ANALYSIS_END_TIME_REL_REWARD = TOTAL_DURATION_POST_REWARD
    WINDOW_SIZE = 0.1; STEP_SIZE = 0.02; TIME_RESOLUTION = 0.001
    size_labels_map_ts = {1: 'VS', 2: 'S', 3: 'M', 4: 'L', 5: 'VL'}
    ALL_REWARD_SIZES = list(size_labels_map_ts.keys())

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
    win_samples = int(np.round(WINDOW_SIZE / TIME_RESOLUTION)); win_samples = max(1, win_samples)

    # --- Helper function --- (Copied)
    def calculate_single_trial_ts(mtrial, reward_latency):
        lick_downs = mtrial.get('lick_downs_relative', []); lick_ups = mtrial.get('lick_ups_relative', [])
        analysis_start_rel_trial = reward_latency + ANALYSIS_START_TIME_REL_REWARD
        analysis_end_rel_trial = reward_latency + ANALYSIS_END_TIME_REL_REWARD
        max_time_rel_trial_needed = analysis_end_rel_trial + TIME_RESOLUTION
        num_hires_points_trial = int(np.ceil(max_time_rel_trial_needed / TIME_RESOLUTION))
        lick_trace_trial = np.zeros(num_hires_points_trial, dtype=np.int8)
        lick_downs_sorted = sorted([ld for ld in lick_downs if ld is not None])
        lick_ups_sorted = sorted([lu for lu in lick_ups if lu is not None])
        num_bouts = min(len(lick_downs_sorted), len(lick_ups_sorted))
        for i in range(num_bouts):
            start_idx = int(np.floor((lick_downs_sorted[i] + 1e-9) / TIME_RESOLUTION)); end_idx = int(np.ceil(lick_ups_sorted[i] / TIME_RESOLUTION))
            start_idx = max(0, start_idx); end_idx = min(num_hires_points_trial, end_idx)
            if start_idx < end_idx: lick_trace_trial[start_idx:end_idx] = 1
        idx_analysis_start_in_trial_trace = int(np.round(analysis_start_rel_trial / TIME_RESOLUTION))
        idx_analysis_end_in_trial_trace = idx_analysis_start_in_trial_trace + num_hires_points_total_analysis
        actual_start_idx = max(0, idx_analysis_start_in_trial_trace); actual_end_idx = min(len(lick_trace_trial), idx_analysis_end_in_trial_trace)
        extracted_trace = lick_trace_trial[actual_start_idx:actual_end_idx]
        pad_before = max(0, -idx_analysis_start_in_trial_trace); pad_after = max(0, num_hires_points_total_analysis - (len(extracted_trace) + pad_before))
        analysis_trace_hires = np.pad(extracted_trace, (pad_before, pad_after), 'constant')
        if len(analysis_trace_hires) != num_hires_points_total_analysis: return np.full(num_target_time_points, np.nan)
        rolling_proportions = pd.Series(analysis_trace_hires).rolling(window=win_samples, min_periods=1, center=True).mean().to_numpy()
        valid_indices_mask = target_indices_in_hires < len(rolling_proportions)
        valid_target_indices_for_sampling = target_indices_in_hires[valid_indices_mask]
        sampled_proportions = np.full(num_target_time_points, np.nan)
        sampled_proportions[valid_indices_mask] = rolling_proportions[valid_target_indices_for_sampling]
        return sampled_proportions

    # --- Data Pooling Structure --- 
    pooled_ts_data = {
        phase: {
            'current_size': {size: [] for size in ALL_REWARD_SIZES},
            'previous_size': {size: [] for size in ALL_REWARD_SIZES}
        } for phase in phases_to_compare
    }

    # --- Data Processing Loop --- 
    print("  Pooling time series data across filtered sessions...")
    sessions_processed = 0
    for session in filtered_test_sessions:
        session_phase = session.get('phase')
        if session_phase not in phases_to_compare: continue # Should not happen if filtered, but safety check
        
        monitoring_trials = session.get('trial_results', [])
        if not monitoring_trials: continue
        reward_sizes_map_session = {int(t.get('trial_num')): int(t.get('reward_size')) 
                                  for t in monitoring_trials if t.get('trial_num') is not None and t.get('reward_size') is not None}
        if not reward_sizes_map_session: continue
            
        sessions_processed += 1
        # Group by Current Reward Size
        for mtrial in monitoring_trials:
            trial_num = mtrial.get('trial_num'); is_rewarded = mtrial.get('rewarded', False); reward_latency = mtrial.get('reward_latency')
            if trial_num is None or not is_rewarded or reward_latency is None: continue
            reward_size = reward_sizes_map_session.get(int(trial_num))
            if reward_size not in ALL_REWARD_SIZES: continue
            try:
                sampled_proportions = calculate_single_trial_ts(mtrial, reward_latency)
                if sampled_proportions is not None and not np.all(np.isnan(sampled_proportions)):
                     pooled_ts_data[session_phase]['current_size'][reward_size].append(sampled_proportions)
            except Exception: pass
                 
        # Group by Previous Reward Size
        if len(monitoring_trials) >= 2:
            for idx in range(1, len(monitoring_trials)):
                mtrial_curr = monitoring_trials[idx]; mtrial_prev = monitoring_trials[idx-1]
                trial_num_curr = mtrial_curr.get('trial_num'); is_rewarded_curr = mtrial_curr.get('rewarded', False); reward_latency_curr = mtrial_curr.get('reward_latency')
                if trial_num_curr is None or not is_rewarded_curr or reward_latency_curr is None: continue
                is_rewarded_prev = mtrial_prev.get('rewarded', False); trial_num_prev = mtrial_prev.get('trial_num')
                if not is_rewarded_prev or trial_num_prev is None: continue
                reward_size_prev = reward_sizes_map_session.get(int(trial_num_prev))
                if reward_size_prev not in ALL_REWARD_SIZES: continue
                try:
                    sampled_proportions = calculate_single_trial_ts(mtrial_curr, reward_latency_curr)
                    if sampled_proportions is not None and not np.all(np.isnan(sampled_proportions)):
                         pooled_ts_data[session_phase]['previous_size'][reward_size_prev].append(sampled_proportions)
                except Exception: pass
                     
    print(f"  Processed {sessions_processed} sessions for pooling.")

    # --- Aggregation Function --- (Similar to previous)
    def aggregate_pooled_data(pooled_dict):
        aggregated_results = {}
        for phase in pooled_dict:
            aggregated_results[phase] = {}
            for group_type in pooled_dict[phase]:
                aggregated_results[phase][group_type] = {}
                for r_size in pooled_dict[phase][group_type]:
                    trial_list = pooled_dict[phase][group_type][r_size]
                    valid_trials_ts = [ts for ts in trial_list if ts is not None and not np.all(np.isnan(ts)) and len(ts) == num_target_time_points]
                    n_trials = len(valid_trials_ts)
                    mean_ts, sem_ts = None, None
                    if n_trials > 0:
                        trial_ts_stack = np.array(valid_trials_ts)
                        mean_ts = np.nanmean(trial_ts_stack, axis=0)
                        if n_trials > 1:
                            with np.errstate(invalid='ignore'): sem_ts = sem(trial_ts_stack, axis=0, nan_policy='omit')
                            sem_ts = np.nan_to_num(sem_ts, nan=0.0)
                        else: sem_ts = np.zeros_like(mean_ts)
                    aggregated_results[phase][group_type][r_size] = {'mean': mean_ts, 'sem': sem_ts, 'n': n_trials}
        return aggregated_results

    aggregated_data = aggregate_pooled_data(pooled_ts_data)

    # --- Plotting --- 
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True) # 2x1 grid
    fig.suptitle(f"{animal_id} - Time-Resolved Licking (Initial vs Stress Pre-PL37)", fontsize=16, y=0.98)
    
    plot_successful = False
    grouping_types = ['current_size', 'previous_size']
    panel_titles = ['Grouped by Current Reward Size', 'Grouped by Previous Reward Size']
    cmap = plt.cm.viridis 
    norm = mcolors.Normalize(vmin=min(ALL_REWARD_SIZES), vmax=max(ALL_REWARD_SIZES))

    for row_idx, group_type in enumerate(grouping_types):
        ax = axs[row_idx]
        max_y_panel = 0
        panel_has_data = False
        legend_handles = {phase: [] for phase in phases_to_compare}
        legend_labels = {phase: [] for phase in phases_to_compare}
        
        for phase in phases_to_compare:
            data_to_plot = aggregated_data[phase][group_type]
            color = PHASE_COMPARISON_COLORS.get(phase, 'gray') # Get phase color
            
            for r_size in ALL_REWARD_SIZES:
                agg_res = data_to_plot.get(r_size, {})
                mean_ts = agg_res.get('mean'); sem_ts = agg_res.get('sem'); n_val = agg_res.get('n', 0)
                
                if mean_ts is not None and sem_ts is not None and n_val > 0:
                    panel_has_data = True
                    plot_successful = True
                    label = f"Size {r_size} (N={n_val})" # Combine phase and size in label
                    line, = ax.plot(time_points_relative_to_reward, mean_ts, color=color, label=label, linewidth=1.5)
                    ax.fill_between(time_points_relative_to_reward, mean_ts - sem_ts, mean_ts + sem_ts, color=color, alpha=0.15)
                    max_y_panel = max(max_y_panel, np.nanmax(mean_ts + sem_ts))
                    legend_handles[phase].append(line)
                    legend_labels[phase].append(label)
            
        ax.axvline(x=0, color='black', linestyle=':', linewidth=1.5, alpha=0.8, label='_nolegend_')
        ax.grid(True, alpha=0.5, linestyle='--')
        ax.set_title(panel_titles[row_idx], fontsize=12)
        ax.set_ylabel("Mean Lick Proportion", fontsize=10)
        ax.set_ylim(bottom=-0.05, top=max(0.1, max_y_panel * 1.1))
        
        if not panel_has_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, color='gray')
        else:
             # Create combined legend
             combined_handles = []
             combined_labels = []
             for phase in phases_to_compare:
                 # Add a phase title/separator (optional)
                 # combined_handles.append(plt.Line2D([0], [0], color='white')) # Invisible handle
                 # combined_labels.append(f'-- {phase} --') 
                 combined_handles.extend(legend_handles[phase])
                 combined_labels.extend(legend_labels[phase])
             # Add reward line handle manually
             reward_line = plt.Line2D([0], [0], color='black', linestyle=':', linewidth=1.5, alpha=0.8)
             combined_handles.append(reward_line)
             combined_labels.append('Reward Delivery')
             ax.legend(combined_handles, combined_labels, title=f"{phases_to_compare[0]} (Solid) vs {phases_to_compare[1]} (Dashed? - needs update)", 
                       loc='upper right', fontsize='x-small', ncol=2)
             # TODO: Update legend title/logic if using different line styles per phase

    axs[-1].set_xlabel("Time Relative to Reward Delivery (s)", fontsize=10)
    axs[-1].set_xlim(ANALYSIS_START_TIME_REL_REWARD, ANALYSIS_END_TIME_REL_REWARD)
        
    plt.tight_layout(rect=[0.05, 0.03, 1, 0.96]) # Adjust layout

    # --- Save Figure --- 
    if not plot_successful:
        print("  No data plotted for any panel. Skipping save.")
        plt.close(fig)
        return None
        
    output_filename = os.path.join(output_dir, f"{animal_id}_time_resolved_licking_phase_summary.png")
    try:
        plt.savefig(output_filename, dpi=150)
        print(f"  Saved time-resolved licking phase summary: {output_filename}")
    except Exception as e:
        print(f"  ERROR saving time-resolved licking phase summary: {e}")
        output_filename = None
    finally:
        plt.close(fig)
        
    print(f"  Finished time-resolved licking phase summary plot generation ({time.time() - start_time:.2f} seconds).")
    return output_filename


def visualize_cross_session_prev_extremes_stress_summary_phase_aware(filtered_test_sessions, animal_id, output_dir):
    """Generates a 5x2 summary plot comparing time-resolved licking 
       (Initial Test vs Stress Test Pre-PL37 phases), for each current reward size, 
       grouped by previous reward extremes (VS/S vs L/VL).
       Assumes input `filtered_test_sessions` contains only data from these two phases.
    """
    if not filtered_test_sessions:
        print("No filtered test sessions provided for prev extremes phase summary.")
        return None

    print(f"\nGenerating Cross-Session Previous Extremes Phase Summary plot for {animal_id}...")
    start_time = time.time()

    # --- Identify phases present (should be Initial and Stress Pre-PL37) --- 
    present_phases = sorted(list(set(s.get('phase') for s in filtered_test_sessions if s.get('phase'))))
    phases_to_compare = [p for p in ["Initial Test", "Stress Test (Pre-PL37)"] if p in present_phases]
    if len(phases_to_compare) != 2:
        print(f"  Warning: Expected exactly Initial Test and Stress Test (Pre-PL37), found: {present_phases}. Cannot generate comparison plot.")
        return None
    print(f"  Comparing phases: {phases_to_compare}")
    
    # --- Analysis Parameters (Copied) --- 
    ANALYSIS_START_TIME_REL_REWARD = -2.0
    TOTAL_DURATION_POST_REWARD = 8.0 
    ANALYSIS_END_TIME_REL_REWARD = TOTAL_DURATION_POST_REWARD
    WINDOW_SIZE = 0.1; STEP_SIZE = 0.02; TIME_RESOLUTION = 0.001
    ALL_REWARD_SIZES = [1, 2, 3, 4, 5]
    size_labels_map = {1: 'VS', 2: 'S', 3: 'M', 4: 'L', 5: 'VL'}
    PREV_VS_S_SIZES = [1, 2]; PREV_L_VL_SIZES = [4, 5]
    GROUP1_LABEL = "Prev VS/S"; GROUP2_LABEL = "Prev L/VL"
    GROUP_COLORS = {GROUP1_LABEL: 'blue', GROUP2_LABEL: 'orange'}

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
    win_samples = int(np.round(WINDOW_SIZE / TIME_RESOLUTION)); win_samples = max(1, win_samples)

    # --- Helper function --- (Copied)
    def calculate_single_trial_ts(mtrial, reward_latency):
        lick_downs = mtrial.get('lick_downs_relative', []); lick_ups = mtrial.get('lick_ups_relative', [])
        analysis_start_rel_trial = reward_latency + ANALYSIS_START_TIME_REL_REWARD
        analysis_end_rel_trial = reward_latency + ANALYSIS_END_TIME_REL_REWARD
        max_time_rel_trial_needed = analysis_end_rel_trial + TIME_RESOLUTION
        num_hires_points_trial = int(np.ceil(max_time_rel_trial_needed / TIME_RESOLUTION))
        lick_trace_trial = np.zeros(num_hires_points_trial, dtype=np.int8)
        lick_downs_sorted = sorted([ld for ld in lick_downs if ld is not None])
        lick_ups_sorted = sorted([lu for lu in lick_ups if lu is not None])
        num_bouts = min(len(lick_downs_sorted), len(lick_ups_sorted))
        for i in range(num_bouts):
            start_idx = int(np.floor((lick_downs_sorted[i] + 1e-9) / TIME_RESOLUTION)); end_idx = int(np.ceil(lick_ups_sorted[i] / TIME_RESOLUTION))
            start_idx = max(0, start_idx); end_idx = min(num_hires_points_trial, end_idx)
            if start_idx < end_idx: lick_trace_trial[start_idx:end_idx] = 1
        idx_analysis_start_in_trial_trace = int(np.round(analysis_start_rel_trial / TIME_RESOLUTION))
        idx_analysis_end_in_trial_trace = idx_analysis_start_in_trial_trace + num_hires_points_total_analysis
        actual_start_idx = max(0, idx_analysis_start_in_trial_trace); actual_end_idx = min(len(lick_trace_trial), idx_analysis_end_in_trial_trace)
        extracted_trace = lick_trace_trial[actual_start_idx:actual_end_idx]
        pad_before = max(0, -idx_analysis_start_in_trial_trace); pad_after = max(0, num_hires_points_total_analysis - (len(extracted_trace) + pad_before))
        analysis_trace_hires = np.pad(extracted_trace, (pad_before, pad_after), 'constant')
        if len(analysis_trace_hires) != num_hires_points_total_analysis: return np.full(num_target_time_points, np.nan)
        rolling_proportions = pd.Series(analysis_trace_hires).rolling(window=win_samples, min_periods=1, center=True).mean().to_numpy()
        valid_indices_mask = target_indices_in_hires < len(rolling_proportions)
        valid_target_indices_for_sampling = target_indices_in_hires[valid_indices_mask]
        sampled_proportions = np.full(num_target_time_points, np.nan)
        sampled_proportions[valid_indices_mask] = rolling_proportions[valid_target_indices_for_sampling]
        return sampled_proportions

    # --- Data Pooling Structure --- 
    pooled_data = {phase: {curr_size: {GROUP1_LABEL: [], GROUP2_LABEL: []} 
                           for curr_size in ALL_REWARD_SIZES} 
                   for phase in phases_to_compare}

    # --- Data Processing Loop --- 
    print("  Pooling time series data across filtered sessions for prev extremes summary...")
    sessions_processed = 0
    for session in filtered_test_sessions:
        session_phase = session.get('phase')
        if session_phase not in phases_to_compare: continue
        
        monitoring_trials = session.get('trial_results', [])
        if not monitoring_trials or len(monitoring_trials) < 2: continue
        reward_sizes_map_session = {int(t.get('trial_num')): int(t.get('reward_size')) 
                                  for t in monitoring_trials if t.get('trial_num') is not None and t.get('reward_size') is not None}
        if not reward_sizes_map_session: continue
            
        sessions_processed += 1
        for idx in range(1, len(monitoring_trials)):
            mtrial_curr = monitoring_trials[idx]; mtrial_prev = monitoring_trials[idx-1]
            trial_num_curr = mtrial_curr.get('trial_num')
            try:
                reward_size_curr = reward_sizes_map_session.get(int(trial_num_curr))
                if reward_size_curr not in ALL_REWARD_SIZES: continue
                is_rewarded_curr = mtrial_curr.get('rewarded', False); reward_latency_curr = mtrial_curr.get('reward_latency')
                if not is_rewarded_curr or reward_latency_curr is None: continue
                is_rewarded_prev = mtrial_prev.get('rewarded', False); trial_num_prev = mtrial_prev.get('trial_num')
                if not is_rewarded_prev or trial_num_prev is None: continue
                reward_size_prev = reward_sizes_map_session.get(int(trial_num_prev))
                if reward_size_prev is None: continue
                prev_group_key = None
                if reward_size_prev in PREV_VS_S_SIZES: prev_group_key = GROUP1_LABEL
                elif reward_size_prev in PREV_L_VL_SIZES: prev_group_key = GROUP2_LABEL
                else: continue
                sampled_proportions = calculate_single_trial_ts(mtrial_curr, reward_latency_curr)
                if sampled_proportions is not None and not np.all(np.isnan(sampled_proportions)):
                     pooled_data[session_phase][reward_size_curr][prev_group_key].append(sampled_proportions)
            except Exception: pass
                     
    print(f"  Processed {sessions_processed} sessions for prev extremes pooling.")

    # --- Aggregation Function --- (Similar)
    def aggregate_pooled_extremes_data(data_dict):
        aggregated_results = {phase: {curr_size: {} for curr_size in ALL_REWARD_SIZES} for phase in phases_to_compare}
        for phase in data_dict:
            for curr_size in data_dict[phase]:
                for group_label in data_dict[phase][curr_size]:
                    trial_list = data_dict[phase][curr_size][group_label]
                    valid_trials_ts = [ts for ts in trial_list if ts is not None and not np.all(np.isnan(ts)) and len(ts) == num_target_time_points]
                    n_trials = len(valid_trials_ts)
                    mean_ts, sem_ts = None, None
                    if n_trials > 0:
                        trial_ts_stack = np.array(valid_trials_ts)
                        mean_ts = np.nanmean(trial_ts_stack, axis=0)
                        if n_trials > 1:
                            with np.errstate(invalid='ignore'): sem_ts = sem(trial_ts_stack, axis=0, nan_policy='omit')
                            sem_ts = np.nan_to_num(sem_ts, nan=0.0)
                        else: sem_ts = np.zeros_like(mean_ts)
                    aggregated_results[phase][curr_size][group_label] = {'mean': mean_ts, 'sem': sem_ts, 'n': n_trials}
        return aggregated_results

    aggregated_data = aggregate_pooled_extremes_data(pooled_data)

    # --- Plotting --- 
    fig, axs = plt.subplots(len(ALL_REWARD_SIZES), 2, figsize=(10, 12), sharex=True, sharey=True)
    if len(ALL_REWARD_SIZES) == 1: axs = np.array([axs]) # Ensure 2D
    fig.suptitle(f"{animal_id} - Licking by Previous Extremes (Initial vs Stress Pre-PL37)", fontsize=14, y=0.99)

    plot_successful = False; max_y_overall = 0

    for row_idx, curr_size in enumerate(ALL_REWARD_SIZES):
        for col_idx, phase in enumerate(phases_to_compare):
            ax = axs[row_idx, col_idx]
            data_to_plot = aggregated_data[phase][curr_size]
            panel_has_data = False
            legend_handles = []; legend_labels = []
            
            for group_label in [GROUP1_LABEL, GROUP2_LABEL]:
                agg_res = data_to_plot.get(group_label, {})
                mean_ts = agg_res.get('mean'); sem_ts = agg_res.get('sem'); n_val = agg_res.get('n', 0)
                
                if mean_ts is not None and sem_ts is not None and n_val > 0:
                    panel_has_data = True; plot_successful = True
                    color = GROUP_COLORS[group_label]
                    label = f"{group_label} (N={n_val})" 
                    line, = ax.plot(time_points_relative_to_reward, mean_ts, color=color, label=label, linewidth=1.5)
                    ax.fill_between(time_points_relative_to_reward, mean_ts - sem_ts, mean_ts + sem_ts, color=color, alpha=0.2)
                    max_y_overall = max(max_y_overall, np.nanmax(mean_ts + sem_ts))
                    legend_handles.append(line); legend_labels.append(label)
            
            ax.axvline(x=0, color='black', linestyle=':', linewidth=1, alpha=0.7, label='_nolegend_')
            ax.grid(True, alpha=0.4, linestyle='--')

            if not panel_has_data:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, color='gray')
            else:
                ax.legend(handles=legend_handles, labels=legend_labels, loc='upper right', fontsize='x-small')

            # Titles/Labels
            if row_idx == 0: # Top row: Column titles (Phases)
                 ax.set_title(phase.replace(" Test", "").replace(" (Pre-PL37)","-Pre"), fontsize=11)
            if col_idx == 0: # Left column: Row titles (Current Reward Size)
                 row_title = f"Current: {size_labels_map[curr_size]} ({curr_size})"
                 ax.set_ylabel(row_title, fontsize=9)
            if row_idx == len(ALL_REWARD_SIZES) - 1: # Bottom row: X-axis label
                 ax.set_xlabel("Time Relative to Reward (s)", fontsize=10)
            if col_idx > 0: plt.setp(ax.get_yticklabels(), visible=False) # Hide inner Y ticks
            if row_idx < len(ALL_REWARD_SIZES) - 1: plt.setp(ax.get_xticklabels(), visible=False) # Hide inner X ticks

    # Set shared axes limits
    # plt.setp(axs, ylim=(-0.05, max(1.05, max_y_overall * 1.1))) 
    plt.setp(axs, xlim=(ANALYSIS_START_TIME_REL_REWARD, ANALYSIS_END_TIME_REL_REWARD))
    fig.text(0.01, 0.5, "Mean Lick Proportion", va='center', rotation='vertical', fontsize=12)
    plt.tight_layout(rect=[0.05, 0.03, 1, 0.96])

    # --- Save Figure --- 
    if not plot_successful:
        print("  No data plotted for any panel. Skipping save.")
        plt.close(fig)
        return None
        
    output_filename = os.path.join(output_dir, f"{animal_id}_prev_extremes_phase_summary.png")
    try:
        plt.savefig(output_filename, dpi=150)
        print(f"  Saved prev extremes phase summary: {output_filename}")
    except Exception as e:
        print(f"  ERROR saving prev extremes phase summary: {e}")
        output_filename = None
    finally:
        plt.close(fig)
        
    print(f"  Finished prev extremes phase summary plot generation ({time.time() - start_time:.2f} seconds).")
    return output_filename


def visualize_cross_session_test_analysis_with_phase_lines(test_sessions, animal_id, output_dir,
                                                            pl37_vline_day=None, post_pl37_vline_day=None):
    """
    Generates a 5x2 summary plot comparing latency and lick sensor cut duration 
    across test sessions, grouped by the reward size of the *previous* trial.
    Includes vertical lines indicating the start of specific phases (PL37, Post-PL37).
    Shows individual trial data points and mean+/-SEM per day.

    Adapted from the original visualize_cross_session_test_analysis.

    Parameters:
    -----------
    test_sessions : list
        A list of dictionaries, where each dictionary contains processed data 
        for a single test session, including 'trial_results', 'session_day', 
        and optionally 'session_info' with 'stress_status' or 'phase'.
    animal_id : str
        Identifier for the animal.
    output_dir : str
        Directory to save the output figure.
    pl37_vline_day : int or None
        The overall session day number when PL37 phase starts.
    post_pl37_vline_day : int or None
        The overall session day number when Post-PL37 phase starts.
        
    Returns:
    --------
    str or None
        The absolute path to the saved figure file, or None if saving failed or no data.
    """
    if not test_sessions:
        print("No test sessions provided for cross-session summary with phase lines.")
        return None

    # Sort sessions by day to ensure chronological plotting
    test_sessions.sort(key=lambda x: x['session_day'])
    session_days = [s['session_day'] for s in test_sessions]

    # Define reward size labels and colormap (consistent with other plots)
    size_labels_map = {1: 'Very Small', 2: 'Small', 3: 'Medium', 4: 'Large', 5: 'Very Large'}
    reward_sizes = list(size_labels_map.keys())
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=min(reward_sizes), vmax=max(reward_sizes))

    # --- Find original stress boundary (if info exists) --- 
    first_stress_day = None
    stress_boundary_pos = None
    for session in test_sessions:
        if session.get('session_info', {}).get('stress_status', False):
            first_stress_day = session['session_day']
            break 
    if first_stress_day is not None:
        stress_boundary_pos = first_stress_day - 0.5
        
    # --- Calculate NEW phase boundary positions --- 
    pl37_line_pos = pl37_vline_day if pl37_vline_day is not None else None # Draw exactly on day
    post_pl37_boundary_pos = post_pl37_vline_day - 0.5 if post_pl37_vline_day is not None else None
    # -----------------------------------------------

    # --- Data Aggregation (same as original cross-session plot) --- 
    plot_data = {size: {day: {'latencies': [], 'durations': []} for day in session_days} 
                 for size in reward_sizes}
    all_latencies_overall = [] 
    all_durations_overall = []

    for session in test_sessions:
        session_day = session['session_day']
        trial_results = session.get('trial_results', [])
        if not trial_results or len(trial_results) < 2: continue
        
        for i in range(1, len(trial_results)):
            prev_trial = trial_results[i-1]
            current_trial = trial_results[i]

            if prev_trial.get('rewarded', False) and 'reward_size' in prev_trial:
                prev_size = prev_trial['reward_size']
                if prev_size in reward_sizes:
                    latency = current_trial.get('first_lick_latency')
                    if latency is not None and not np.isnan(latency):
                         plot_data[prev_size][session_day]['latencies'].append(latency)
                         all_latencies_overall.append(latency)
                    
                    cut_duration = None
                    lick_analysis_data = session.get('lick_sensor_cut_analysis', {})
                    if 'trial_metrics' in lick_analysis_data:
                        metrics = lick_analysis_data['trial_metrics']
                        current_trial_num = current_trial.get('trial_num')
                        metric = next((m for m in metrics if m.get('trial_num') == current_trial_num), None)
                        if metric and 'post_cue_cut_duration' in metric: cut_duration = metric['post_cue_cut_duration']
                    elif 'post_cue_cut_duration' in current_trial: cut_duration = current_trial['post_cue_cut_duration']

                    if cut_duration is not None and not np.isnan(cut_duration):
                        plot_data[prev_size][session_day]['durations'].append(cut_duration)
                        all_durations_overall.append(cut_duration)

    # --- Plotting (5x2 grid) --- 
    fig, axes = plt.subplots(len(reward_sizes), 2, figsize=(18, 22), sharex=True)
    fig.suptitle(f'{animal_id} - Behavior Summary by Previous Reward Size (Phase Lines Added)', fontsize=16, y=0.98)

    # Calculate shared Y-limits
    lat_ylim = (0, 3) # Keep fixed latency ylim from original for consistency? Or recalculate?
    # Recalculate duration ylim based on data
    dur_min, dur_max = (np.min(all_durations_overall), np.max(all_durations_overall)) if all_durations_overall else (0, 1.5)
    dur_padding = (dur_max - dur_min) * 0.1 if (dur_max - dur_min) > 0 else 0.1
    dur_ylim = (max(0, dur_min - dur_padding), min(1.5, dur_max + dur_padding)) # Cap at 1.5s

    # Track if labels for lines have been added
    stress_label_added = False
    pl37_label_added = False
    post_pl37_label_added = False

    for i, size in enumerate(reward_sizes):
        row_index = i
        color = cmap(norm(size))
        label = size_labels_map[size]

        ax_lat = axes[row_index, 0]
        ax_dur = axes[row_index, 1]

        # Plot individual points and mean/sem per day (original logic)
        for day_idx, day in enumerate(session_days):
            day_latencies = plot_data[size][day]['latencies']
            day_durations = plot_data[size][day]['durations']

            if day_latencies:
                jitter_lat = np.random.normal(0, 0.08, size=len(day_latencies))
                ax_lat.scatter(np.array([day] * len(day_latencies)) + jitter_lat, day_latencies, 
                           alpha=0.4, s=45, color=color, edgecolors='none') 
                mean_lat = np.mean(day_latencies)
                sem_lat = sem(day_latencies) if len(day_latencies) > 1 else 0
                ax_lat.errorbar(day, mean_lat, yerr=sem_lat, fmt='o', color='black',
                                markersize=6, capsize=4, elinewidth=1.5, markeredgecolor='white')
            # Optionally plot a marker indicating no data for this day
            # else: ax_lat.plot(day, lat_ylim[0] - 0.1, 'x', color='lightgray', markersize=4) 

            if day_durations:
                jitter_dur = np.random.normal(0, 0.08, size=len(day_durations))
                ax_dur.scatter(np.array([day] * len(day_durations)) + jitter_dur, day_durations, 
                           alpha=0.4, s=45, color=color, edgecolors='none')
                mean_dur = np.mean(day_durations)
                sem_dur = sem(day_durations) if len(day_durations) > 1 else 0
                ax_dur.errorbar(day, mean_dur, yerr=sem_dur, fmt='o', color='black', 
                                markersize=6, capsize=4, elinewidth=1.5, markeredgecolor='white')
            # Optionally plot a marker indicating no data for this day
            # else: ax_dur.plot(day, dur_ylim[0] - 0.1, 'x', color='lightgray', markersize=4) 

        ax_lat.set_title(f"Latency after '{label}' Reward")
        ax_lat.set_ylabel("First Lick Latency (s)")
        ax_lat.grid(True, alpha=0.5)
        ax_lat.set_ylim(lat_ylim) # Apply fixed latency ylim

        ax_dur.set_title(f"Cut Duration after '{label}' Reward")
        ax_dur.set_ylabel("Lick Cut Duration (s)")
        ax_dur.grid(True, alpha=0.5)
        ax_dur.set_ylim(dur_ylim) # Apply calculated duration ylim
        ax_dur.axhline(y=1.5, color='red', linestyle=':', alpha=0.5, linewidth=1) # Add max duration line
        
        # --- Add boundary lines to BOTH axes in the row --- 
        for ax in [ax_lat, ax_dur]:
            if stress_boundary_pos is not None:
                ax.axvline(x=stress_boundary_pos, color='blue', linestyle=':', alpha=0.6, # Match style from fig 1 
                           linewidth=1.5, label='Stress Phase Start' if not stress_label_added else '_nolegend_')
            if pl37_line_pos is not None:
                ax.axvline(x=pl37_line_pos, color='green', linestyle='-.', alpha=0.7, # New Position
                           linewidth=1.5, label='PL37' if not pl37_label_added else '_nolegend_') # New Label
            if post_pl37_boundary_pos is not None:
                ax.axvline(x=post_pl37_boundary_pos, color='purple', linestyle=':', alpha=0.7, # New Line Style
                           linewidth=1.5, label='Stress Phase End' if not post_pl37_label_added else '_nolegend_') # New Label
        # -----------------------------------------------------
        
        # Ensure labels are only added once (in the first row)
        if i == 0:
            if stress_boundary_pos is not None: stress_label_added = True
            if pl37_line_pos is not None: pl37_label_added = True
            if post_pl37_boundary_pos is not None: post_pl37_label_added = True

        # Set x-axis label only for the bottom row
        if row_index == len(reward_sizes) - 1:
            ax_lat.set_xlabel("Session Day")
            ax_dur.set_xlabel("Session Day")
        
        # Set x-ticks for all subplots 
        ax_lat.set_xticks(session_days)
        ax_dur.set_xticks(session_days)
        ax_lat.tick_params(axis='x', labelsize=8) # Reduce x-tick label size
        ax_dur.tick_params(axis='x', labelsize=8) # Reduce x-tick label size

    # Create a single legend for the boundary lines
    handles, labels = axes[0, 0].get_legend_handles_labels() # Get labels from one subplot
    # Filter out non-boundary labels if any were accidentally added (e.g., from errorbar)
    filtered_handles = []
    filtered_labels = []
    boundary_labels_ordered = ['Stress Phase Start', 'PL37', 'Stress Phase End'] # Updated labels
    for label_name in boundary_labels_ordered:
         try:
             idx = labels.index(label_name)
             filtered_handles.append(handles[idx])
             filtered_labels.append(labels[idx])
         except ValueError:
             pass # Label not found
             
    if filtered_handles: # Only add legend if lines were actually drawn
         fig.legend(filtered_handles, filtered_labels, loc='upper right', 
                    bbox_to_anchor=(0.98, 0.97), fontsize='small') # Add smaller font size

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect for suptitle and legend

    # Save the figure
    try:
        filename = f"{animal_id}_TEST_prev_reward_summary_with_phase_lines.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved cross-session summary figure with phase lines: {filepath}")
        plt.close(fig)
        return filepath
    except Exception as e:
        print(f"Error saving cross-session summary figure with phase lines: {e}")
        plt.close(fig)
        return None


def visualize_pooled_comparison_four_phases(test_sessions, animal_id, output_dir):
    """Generates a 1x2 plot comparing Latency and Duration across four phases:
       Pre Stress, Stress, PL37, Post Stress.
       Data is pooled across all previous reward sizes.
       Includes violin plots, individual points, means, and comparisons vs Stress phase.
    """
    if not test_sessions:
        print("No test sessions provided for 4-phase pooled comparison.")
        return None

    print(f"\nGenerating Pooled 4-Phase Comparison plot for {animal_id}...")
    start_time = time.time()

    # --- Define Phase Labels and Order --- 
    internal_phases = PHASE_ORDER 
    # NEW Display names for the plot
    display_phases_map = {
        "Initial Test": "Pre Stress", # Changed from Control
        "Stress Test (Pre-PL37)": "Stress",   # Changed from Pre Stress
        "PL37 Test": "PL37",
        "Post-PL37 Test": "Post Stress"
    }
    display_phases_order = [display_phases_map[p] for p in internal_phases]
    display_phase_colors = {display_phases_map[p]: PHASE_COLORS[p] for p in internal_phases}
    # -------------------------------------

    # --- Filter sessions (using internal phase names) --- 
    comparison_sessions = [s for s in test_sessions if s.get('phase') in internal_phases]
    if not comparison_sessions:
        print(f"  No sessions found belonging to the target phases: {internal_phases}. Cannot generate plot.")
        return None
        
    # --- Count unique days per phase --- 
    days_per_phase = {phase: set() for phase in internal_phases}
    for s in comparison_sessions:
        phase = s.get('phase'); day = s.get('session_day')
        if phase and day is not None: days_per_phase[phase].add(day)
    print("  Number of unique session days per phase:")
    for phase_internal in internal_phases:
        display_name = display_phases_map[phase_internal]
        print(f"    - {display_name} ({phase_internal}): {len(days_per_phase[phase_internal])}")
    if len(comparison_sessions) != sum(len(d) for d in days_per_phase.values()):
         print(f"  (Note: Total sessions processed = {len(comparison_sessions)})")

    # --- Analysis Parameters & Setup --- 
    min_data_points_for_test = 3 

    # --- Data Pooling & Conversion to DataFrame --- 
    plot_df_list = []
    all_latencies_overall = []
    all_durations_overall = []
    for session in comparison_sessions:
        session_phase_internal = session.get('phase')
        session_phase_display = display_phases_map.get(session_phase_internal) 
        if not session_phase_display: continue
        trial_results = session.get('trial_results', [])
        if not trial_results or len(trial_results) < 2: continue
        for i in range(1, len(trial_results)):
            prev_trial = trial_results[i-1]; current_trial = trial_results[i]
            if prev_trial.get('rewarded', False):
                latency = current_trial.get('first_lick_latency')
                cut_duration = None
                lick_analysis_data = session.get('lick_sensor_cut_analysis', {})
                if 'trial_metrics' in lick_analysis_data:
                    metrics = lick_analysis_data['trial_metrics']
                    current_trial_num = current_trial.get('trial_num')
                    metric_data = next((m for m in metrics if m.get('trial_num') == current_trial_num), None)
                    if metric_data and 'post_cue_cut_duration' in metric_data: cut_duration = metric_data['post_cue_cut_duration']
                elif 'post_cue_cut_duration' in current_trial: cut_duration = current_trial['post_cue_cut_duration']
                latency_val = latency if (latency is not None and not np.isnan(latency)) else np.nan
                duration_val = cut_duration if (cut_duration is not None and not np.isnan(cut_duration)) else np.nan
                if not np.isnan(latency_val): all_latencies_overall.append(latency_val)
                if not np.isnan(duration_val): all_durations_overall.append(duration_val)
                plot_df_list.append({'Phase': session_phase_display, 'latency': latency_val, 'duration': duration_val})
    if not plot_df_list: print("  No valid data points found after pooling. Cannot generate plot."); return None
    plot_df = pd.DataFrame(plot_df_list)
    plot_df['Phase'] = pd.Categorical(plot_df['Phase'], categories=display_phases_order, ordered=True)
    plot_df.dropna(subset=['Phase'], inplace=True)

    # --- Plotting --- 
    fig, axs = plt.subplots(1, 2, figsize=(12, 7)) 
    fig.suptitle(f"{animal_id} - Pooled Comparison Across Phases", fontsize=14, y=0.99)
    plot_generated = False
    stats_results = {} 

    # --- NEW: Helper to format p-value consistently --- 
    def format_p_consistent(p):
        if np.isnan(p): return "p=nan"
        if p < 0.001: 
            return f"p={p:.1e}" # Scientific format for p < 0.001
        else: # p >= 0.001
            # Format with 3 decimal places, handle p=1.000 case if desired
            p_str = f"{p:.3f}"
            # Optional: simplify if p is very close to 1
            # if abs(p - 1.0) < 1e-6: p_str = "1.000"
            return f"p={p_str}" 
        
    metrics_to_plot = ['latency', 'duration']
    panel_titles = ["First Lick Delay (s)", "Pre Reward Lick Duration (s)"] # Updated titles
    overall_data_store = {'latency': all_latencies_overall, 'duration': all_durations_overall}

    for col_idx, metric in enumerate(metrics_to_plot):
        ax = axs[col_idx]
        all_data_metric = overall_data_store[metric]
        stats_results[col_idx] = {'means': {}, 'pairwise_p': {}} 
        
        if not plot_df[metric].dropna().empty:
            plot_generated = True
            sns.violinplot(x='Phase', y=metric, data=plot_df, order=display_phases_order, 
                           hue='Phase', palette=display_phase_colors, inner=None, 
                           linewidth=1.5, ax=ax, legend=False)
            sns.stripplot(x='Phase', y=metric, data=plot_df, order=display_phases_order, 
                          color='.25', size=3, alpha=0.5, jitter=0.15, ax=ax, legend=False)
            groups_data = {phase_display: plot_df[plot_df['Phase'] == phase_display][metric].dropna().tolist() 
                           for phase_display in display_phases_order}
            
            # Calculate and store means
            for i, phase_display in enumerate(display_phases_order):
                 mean_val = np.mean(groups_data[phase_display]) if groups_data[phase_display] else np.nan
                 stats_results[col_idx]['means'][i] = mean_val
                 
            # --- Perform comparisons vs Stress (column 1) --- 
            stress_phase_name = display_phases_order[1] # "Stress"
            g_stress_data = groups_data[stress_phase_name]
            # Comparisons: (0 vs 1), (1 vs 2), (1 vs 3)
            comparison_indices = [0, 2, 3] 
            for i in comparison_indices:
                phase_to_compare = display_phases_order[i]
                g_compare_data = groups_data[phase_to_compare]
                p_val = np.nan
                if len(g_stress_data) >= min_data_points_for_test and len(g_compare_data) >= min_data_points_for_test:
                    try: _, p = mannwhitneyu(g_stress_data, g_compare_data, alternative='two-sided'); p_val = p
                    except ValueError: p_val = 1.0
                stats_results[col_idx]['pairwise_p'][i] = p_val # Store p-value vs Stress (index 1)
        else:
            ax.text(0.5, 0.5, f'No {metric} data', ha='center', va='center', transform=ax.transAxes, color='gray', fontsize=9)
        
        ax.set_title(panel_titles[col_idx], fontsize=11)
        ax.set_ylabel(panel_titles[col_idx], fontsize=10)
        ax.grid(True, axis='y', alpha=0.5, linestyle='--')
        if metric == 'duration': ax.axhline(y=1.5, color='gray', linestyle=':', alpha=0.7, linewidth=1)
        
        # Set Y-limits 
        if all_data_metric:
            min_val, max_val = np.percentile(all_data_metric, [1, 99])
            padding = (max_val - min_val) * 0.1 if max_val > min_val else 0.1
            upper_limit_stat = max_val + padding * 3 
            lower_limit_stat = max(0, min_val - padding * 2)
            if metric == 'duration': upper_limit_stat = min(1.55, upper_limit_stat)
            ax.set_ylim(lower_limit_stat, upper_limit_stat)
        else: ax.set_ylim(0, 1.5 if metric == 'duration' else 3.0)

        # --- Add Mean and Pairwise Annotations --- 
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        y_step = y_range * 0.05 # Define y_step based on 5% of y-axis range
        y_mean_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.75
        y_pval_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.85
        mean_x_offset = 0.1 
        
        # Add means
        means_dict = stats_results[col_idx].get('means', {})
        for i, mean_val in means_dict.items():
             if not np.isnan(mean_val):
                 ax.text(i + mean_x_offset, y_mean_pos, f'{mean_val:.2f}', ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')
                         
        # Add pairwise p-values vs Stress (column 1)
        pairwise_p_dict = stats_results[col_idx].get('pairwise_p', {})
        # Annotate 0 vs 1
        p_val_01 = pairwise_p_dict.get(0) # Comparison Pre Stress (0) vs Stress (1)
        if p_val_01 is not None: # Check if comparison was performed
            ax.text(0.5, y_pval_pos, format_p_consistent(p_val_01), ha='center', va='bottom', fontsize=9, fontweight='bold')
        # Annotate 1 vs 2
        p_val_12 = pairwise_p_dict.get(2) # Comparison Stress (1) vs PL37 (2)
        if p_val_12 is not None:
            ax.text(1.5, y_pval_pos, format_p_consistent(p_val_12), ha='center', va='bottom', fontsize=9, fontweight='bold')
        # Annotate 1 vs 3
        p_val_13 = pairwise_p_dict.get(3) # Comparison Stress (1) vs Post Stress (3)
        if p_val_13 is not None:
            # Place between column index 1 and 3 -> at x=2.5
            ax.text(2.5, y_pval_pos, format_p_consistent(p_val_13), ha='center', va='bottom', fontsize=9, fontweight='bold')
                         
        # Clean up axes appearance
        ax.set_xlabel("") # Remove x-axis title 
        ax.set_xticklabels(display_phases_order, rotation=45, ha='right')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', size=6, width=1.5, labelsize=9)

    plt.tight_layout(rect=[0.05, 0.15, 1, 0.95]) # Adjust layout 

    # --- Save Figure --- 
    if not plot_generated: print("  No data generated. Skipping save."); plt.close(fig); return None
    output_filename = os.path.join(output_dir, f"{animal_id}_pooled_4phase_comparison.png")
    try:
        plt.savefig(output_filename, dpi=150)
        print(f"  Saved pooled 4-phase comparison figure: {output_filename}")
    except Exception as e: print(f"  ERROR saving figure: {e}"); output_filename = None
    finally: plt.close(fig)
    print(f"  Finished pooled 4-phase comparison plot generation ({time.time() - start_time:.2f} seconds).")
    return output_filename


def visualize_pooled_stress_epoch_comparison(test_sessions, animal_id, output_dir):
    """Generates a 1x2 plot comparing specific stress epochs:
       Last 3 Pre-Stress, First 3 Stress, Last 3 Stress, PL37, Post-Stress.
       Data is pooled across all previous reward sizes within these epochs.
       Includes violin plots, individual points, means, and comparisons vs 'Stress first 3 days'.
    """
    if not test_sessions:
        print("No test sessions provided for stress epoch comparison.")
        return None

    print(f"\nGenerating Pooled Stress Epoch Comparison plot for {animal_id}...")
    start_time = time.time()

    # --- Define Epoch Labels and Order --- 
    epoch_labels = [
        "3-days Pre Stress", 
        "Stress first 3 days", 
        "Stress last 3 days", 
        "PL37", 
        "Post Stress"
    ]
    # --- Define Colors for Epochs (Consistent with 4-Phase Plot) ---
    base_stress_color = PHASE_COLORS.get("Stress Test (Pre-PL37)", "#CCCCCC") # Default gray if not found
    stress_palette = sns.light_palette(base_stress_color, n_colors=5) # Create a palette from base

    epoch_colors = {
        epoch_labels[0]: PHASE_COLORS.get("Initial Test", "#AAAAAA"),
        epoch_labels[1]: stress_palette[2], # A lighter shade of base_stress_color
        epoch_labels[2]: stress_palette[3], # A darker shade of base_stress_color (or use dark_palette)
        # If using dark_palette: sns.dark_palette(base_stress_color, n_colors=3)[1]
        epoch_labels[3]: PHASE_COLORS.get("PL37 Test", "#BBBBBB"),
        epoch_labels[4]: PHASE_COLORS.get("Post-PL37 Test", "#DDDDDD")
    }
    # Ensure all epoch_labels have a color, fallback if any key was missed in PHASE_COLORS
    default_epoch_color_fallback = sns.color_palette('pastel', len(epoch_labels))
    for idx, label in enumerate(epoch_labels):
        if label not in epoch_colors or epoch_colors[label] is None:
            print(f"Warning: Color for epoch '{label}' not explicitly defined, using fallback.")
            epoch_colors[label] = default_epoch_color_fallback[idx]
    # --------------------------------------

    # --- Select Sessions for Each Epoch --- 
    # First, separate sessions by internal phase name and sort by day
    sessions_by_phase = {phase: [] for phase in PHASE_ORDER}
    for s in test_sessions:
        phase = s.get('phase')
        if phase in sessions_by_phase:
            sessions_by_phase[phase].append(s)
    for phase in sessions_by_phase:
        sessions_by_phase[phase].sort(key=lambda x: x['session_day'])
        
    # Now, select sessions based on epoch definitions
    selected_sessions_by_epoch = {label: [] for label in epoch_labels}
    days_per_epoch = {label: set() for label in epoch_labels}

    # Group 0: Last 3 days of Initial Test ("Pre Stress" internally)
    initial_sessions = sessions_by_phase["Initial Test"]
    selected_sessions_by_epoch[epoch_labels[0]] = initial_sessions[-3:] # Takes last 3 or fewer if less than 3 exist
    days_per_epoch[epoch_labels[0]] = {s['session_day'] for s in selected_sessions_by_epoch[epoch_labels[0]] if s.get('session_day') is not None}

    # Group 1: First 3 days of Stress Test (Pre-PL37)
    stress_pre_sessions = sessions_by_phase["Stress Test (Pre-PL37)"]
    selected_sessions_by_epoch[epoch_labels[1]] = stress_pre_sessions[:3] # Takes first 3 or fewer
    days_per_epoch[epoch_labels[1]] = {s['session_day'] for s in selected_sessions_by_epoch[epoch_labels[1]] if s.get('session_day') is not None}

    # Group 2: Last 3 days of Stress Test (Pre-PL37)
    # Avoid overlap with first 3 if total <= 6
    if len(stress_pre_sessions) > 3:
        selected_sessions_by_epoch[epoch_labels[2]] = stress_pre_sessions[-3:]
    else:
        # If 3 or fewer stress sessions, this group will be empty or duplicate Group 1.
        # Let's make it empty to avoid confusion. Or should it take all available?
        # Current decision: Empty if <= 3 total stress sessions. Modify if needed.
        print("  Warning: Not enough Stress (Pre-PL37) sessions (>3) to define distinct 'Last 3 days'. Group will be empty.")
        selected_sessions_by_epoch[epoch_labels[2]] = [] 
    days_per_epoch[epoch_labels[2]] = {s['session_day'] for s in selected_sessions_by_epoch[epoch_labels[2]] if s.get('session_day') is not None}

    # Group 3: PL37 Test
    pl37_sessions = sessions_by_phase["PL37 Test"]
    selected_sessions_by_epoch[epoch_labels[3]] = pl37_sessions
    days_per_epoch[epoch_labels[3]] = {s['session_day'] for s in selected_sessions_by_epoch[epoch_labels[3]] if s.get('session_day') is not None}

    # Group 4: Post-PL37 Test
    post_pl37_sessions = sessions_by_phase["Post-PL37 Test"]
    selected_sessions_by_epoch[epoch_labels[4]] = post_pl37_sessions
    days_per_epoch[epoch_labels[4]] = {s['session_day'] for s in selected_sessions_by_epoch[epoch_labels[4]] if s.get('session_day') is not None}

    print("\n  Number of unique session days per epoch group:")
    for label in epoch_labels:
        print(f"    - {label}: {len(days_per_epoch[label])}")

    # --- Data Pooling & Conversion to DataFrame --- 
    plot_df_list = []
    all_latencies_overall = []
    all_durations_overall = []

    for epoch_label, sessions_in_epoch in selected_sessions_by_epoch.items():
        if not sessions_in_epoch: continue # Skip empty epochs
        for session in sessions_in_epoch:
            trial_results = session.get('trial_results', [])
            if not trial_results or len(trial_results) < 2: continue
            for i in range(1, len(trial_results)):
                prev_trial = trial_results[i-1]; current_trial = trial_results[i]
                if prev_trial.get('rewarded', False):
                    latency = current_trial.get('first_lick_latency')
                    cut_duration = None
                    lick_analysis_data = session.get('lick_sensor_cut_analysis', {})
                    if 'trial_metrics' in lick_analysis_data:
                        metrics = lick_analysis_data['trial_metrics']
                        current_trial_num = current_trial.get('trial_num')
                        metric_data = next((m for m in metrics if m.get('trial_num') == current_trial_num), None)
                        if metric_data and 'post_cue_cut_duration' in metric_data: cut_duration = metric_data['post_cue_cut_duration']
                    elif 'post_cue_cut_duration' in current_trial: cut_duration = current_trial['post_cue_cut_duration']
                    latency_val = latency if (latency is not None and not np.isnan(latency)) else np.nan
                    duration_val = cut_duration if (cut_duration is not None and not np.isnan(cut_duration)) else np.nan
                    if not np.isnan(latency_val): all_latencies_overall.append(latency_val)
                    if not np.isnan(duration_val): all_durations_overall.append(duration_val)
                    plot_df_list.append({'Epoch': epoch_label, 'latency': latency_val, 'duration': duration_val})
                        
    if not plot_df_list:
         print("  No valid data points found after pooling for epochs. Cannot generate plot.")
         return None
         
    plot_df = pd.DataFrame(plot_df_list)
    # Ensure Epoch is categorical with the correct 5-epoch order
    plot_df['Epoch'] = pd.Categorical(plot_df['Epoch'], categories=epoch_labels, ordered=True)
    plot_df.dropna(subset=['Epoch'], inplace=True)

    # --- Plotting --- 
    fig, axs = plt.subplots(1, 2, figsize=(14, 7)) # Adjusted figsize for 5 groups
    fig.suptitle(f"{animal_id} - Pooled Comparison Across Stress Epochs", fontsize=14, y=0.99)
    plot_generated = False
    stats_results = {} 
    min_data_points_for_test = 3 # Define again for local scope

    # --- P-value Formatting Helper --- 
    def format_p_consistent(p):
        if np.isnan(p): return "p=nan"
        if p < 0.001: return f"p={p:.1e}"
        else: p_str = f"{p:.3f}"; return f"p={p_str}"
        
    metrics_to_plot = ['latency', 'duration']
    panel_titles = ["First Lick Delay (s)", "Pre Reward Lick Duration (s)"] # Keep updated titles
    overall_data_store = {'latency': all_latencies_overall, 'duration': all_durations_overall}

    for col_idx, metric in enumerate(metrics_to_plot):
        ax = axs[col_idx]
        all_data_metric = overall_data_store[metric]
        stats_results[col_idx] = {'means': {}, 'pairwise_p': {}} 
        
        if not plot_df[metric].dropna().empty:
            plot_generated = True
            sns.violinplot(x='Epoch', y=metric, data=plot_df, order=epoch_labels, 
                           hue='Epoch', palette=epoch_colors, inner=None, 
                           linewidth=1.5, ax=ax, legend=False)
            sns.stripplot(x='Epoch', y=metric, data=plot_df, order=epoch_labels, 
                          color='.25', size=3, alpha=0.5, jitter=0.15, ax=ax, legend=False)
            groups_data = {epoch_label: plot_df[plot_df['Epoch'] == epoch_label][metric].dropna().tolist() 
                           for epoch_label in epoch_labels}
            
            # Calculate and store means
            for i, epoch_label in enumerate(epoch_labels):
                 mean_val = np.mean(groups_data[epoch_label]) if groups_data[epoch_label] else np.nan
                 stats_results[col_idx]['means'][i] = mean_val
                 
            # --- Perform comparisons vs Stress first 3 days (column 1) --- 
            stress_first_3_name = epoch_labels[1] 
            g_stress_first_3_data = groups_data.get(stress_first_3_name, []) # Use .get for safety
            # Comparisons: (0 vs 1), (1 vs 2), (1 vs 3), (1 vs 4)
            comparison_indices = [0, 2, 3, 4] 
            for i in comparison_indices:
                epoch_to_compare = epoch_labels[i]
                g_compare_data = groups_data.get(epoch_to_compare, []) # Use .get
                p_val = np.nan
                # Check if both groups exist and have enough data
                if g_stress_first_3_data and g_compare_data and \
                   len(g_stress_first_3_data) >= min_data_points_for_test and len(g_compare_data) >= min_data_points_for_test:
                    try: _, p = mannwhitneyu(g_stress_first_3_data, g_compare_data, alternative='two-sided'); p_val = p
                    except ValueError: p_val = 1.0
                stats_results[col_idx]['pairwise_p'][i] = p_val # Store p-value vs Stress first 3 (index 1)
        else:
            ax.text(0.5, 0.5, f'No {metric} data', ha='center', va='center', transform=ax.transAxes, color='gray', fontsize=9)
        
        # --- MODIFICATION: Add Detailed Print Output for Stats --- 
        print(f"\n  --- Statistics for {panel_titles[col_idx]} ({metric}) ---")
        current_means = stats_results[col_idx]['means']
        print("    Mean values per epoch:")
        for i, epoch_label in enumerate(epoch_labels):
            mean_val = current_means.get(i, np.nan)
            print(f"      - {epoch_label}: {mean_val:.3f}")
        
        print("\n    Pairwise comparisons vs 'Stress first 3 days':")
        stress_first_3_mean = current_means.get(1, np.nan) # Mean of epoch_labels[1]
        current_pairwise_p = stats_results[col_idx]['pairwise_p']
        comparison_indices_for_print = [0, 2, 3, 4] # Indices of epochs compared against epoch 1
        
        for comp_idx in comparison_indices_for_print:
            epoch_being_compared = epoch_labels[comp_idx]
            mean_of_epoch_being_compared = current_means.get(comp_idx, np.nan)
            p_value_for_comparison = current_pairwise_p.get(comp_idx, np.nan)
            
            print(f"      Comparison: {epoch_being_compared} (mean: {mean_of_epoch_being_compared:.3f}) vs Stress first 3 days (mean: {stress_first_3_mean:.3f})")
            print(f"        P-value: {format_p_consistent(p_value_for_comparison)}\n")
        # --- END MODIFICATION ---
            
        ax.set_title(panel_titles[col_idx], fontsize=11)
        ax.set_ylabel(panel_titles[col_idx], fontsize=10)
        ax.grid(True, axis='y', alpha=0.5, linestyle='--')
        if metric == 'duration': ax.axhline(y=1.5, color='gray', linestyle=':', alpha=0.7, linewidth=1)
        
        # Set Y-limits 
        if all_data_metric:
            min_val, max_val = np.percentile(all_data_metric, [1, 99])
            padding = (max_val - min_val) * 0.1 if max_val > min_val else 0.1
            upper_limit_stat = max_val + padding * 3 
            lower_limit_stat = max(0, min_val - padding * 2)
            if metric == 'duration': upper_limit_stat = min(1.55, upper_limit_stat)
            ax.set_ylim(lower_limit_stat, upper_limit_stat)
        else: ax.set_ylim(0, 1.5 if metric == 'duration' else 3.0)

        # --- Add Mean and Pairwise Annotations --- 
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        y_step = y_range * 0.04 # Smaller step for potentially more comparisons
        y_mean_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.75 # Keep means lower
        y_pval_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.85 # Keep pvals higher
        mean_x_offset = 0.1 
        
        # Add means
        means_dict = stats_results[col_idx].get('means', {})
        for i, mean_val in means_dict.items():
             if not np.isnan(mean_val):
                 ax.text(i + mean_x_offset, y_mean_pos, f'{mean_val:.2f}', ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')
                         
        # Add pairwise p-values vs Stress first 3 (column 1)
        pairwise_p_dict = stats_results[col_idx].get('pairwise_p', {})
        # Annotate 0 vs 1
        p_val_01 = pairwise_p_dict.get(0) 
        if p_val_01 is not None: 
            ax.text(0.5, y_pval_pos, format_p_consistent(p_val_01), ha='center', va='bottom', fontsize=8, fontweight='bold') # Smaller font for pvals
        # Annotate 1 vs 2
        p_val_12 = pairwise_p_dict.get(2) 
        if p_val_12 is not None:
            ax.text(1.5, y_pval_pos, format_p_consistent(p_val_12), ha='center', va='bottom', fontsize=8, fontweight='bold')
        # Annotate 1 vs 3
        p_val_13 = pairwise_p_dict.get(3) 
        if p_val_13 is not None:
            ax.text(2.5, y_pval_pos, format_p_consistent(p_val_13), ha='center', va='bottom', fontsize=8, fontweight='bold')
        # Annotate 1 vs 4
        p_val_14 = pairwise_p_dict.get(4) 
        if p_val_14 is not None:
             ax.text(3.5, y_pval_pos, format_p_consistent(p_val_14), ha='center', va='bottom', fontsize=8, fontweight='bold')
                        
        # Clean up axes appearance
        ax.set_xlabel("") # Remove x-axis title 
        ax.set_xticklabels(epoch_labels, rotation=45, ha='right')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', size=6, width=1.5, labelsize=9)

    plt.tight_layout(rect=[0.05, 0.15, 1, 0.95]) # Adjust layout 

    # --- Save Figure --- 
    if not plot_generated: print("  No data generated for stress epochs. Skipping save."); plt.close(fig); return None
    output_filename = os.path.join(output_dir, f"{animal_id}_pooled_stress_epoch_comparison.png")
    try:
        plt.savefig(output_filename, dpi=150)
        print(f"  Saved pooled stress epoch comparison figure: {output_filename}")
    except Exception as e: print(f"  ERROR saving stress epoch figure: {e}"); output_filename = None
    finally: plt.close(fig)
    print(f"  Finished pooled stress epoch comparison plot generation ({time.time() - start_time:.2f} seconds).")
    return output_filename


def visualize_phase_comparison_by_prev_reward(test_sessions, animal_id, output_dir):
    """Generates a 5x2 plot comparing the 4 key phases (Pre-Stress, Stress, PL37, Post-Stress)
       for latency and duration, broken down by the previous trial's reward size.
       Uses violin plots and includes pairwise statistical tests vs the 'Stress' phase.
    """
    if not test_sessions:
        print("No test sessions provided for phase comparison by previous reward.")
        return None

    print(f"\nGenerating 4-Phase Comparison by Previous Reward plot for {animal_id}...")
    start_time = time.time()

    # --- Define Phases, Display Names, and Order (Consistent with Fig 3) --- 
    internal_phases = PHASE_ORDER 
    display_phases_map = {
        "Initial Test": "Pre Stress",
        "Stress Test (Pre-PL37)": "Stress",
        "PL37 Test": "PL37",
        "Post-PL37 Test": "Post Stress"
    }
    display_phases_order = [display_phases_map[p] for p in internal_phases if p in display_phases_map] # Ensure order
    display_phase_colors = {display_phases_map[p]: PHASE_COLORS[p] for p in internal_phases if p in display_phases_map}
    # --------------------------------------------------------------------
    
    # --- Define Reward Sizes and Labels ---
    size_labels_map = {1: 'VS', 2: 'S', 3: 'M', 4: 'L', 5: 'VL'}
    ALL_REWARD_SIZES = list(size_labels_map.keys())
    min_data_points_for_test = 3

    # --- Filter sessions (using internal phase names) --- 
    comparison_sessions = [s for s in test_sessions if s.get('phase') in internal_phases]
    if not comparison_sessions:
        print(f"  No sessions found belonging to the target phases: {internal_phases}. Cannot generate plot.")
        return None

    # --- Data Pooling & Conversion to DataFrame --- 
    plot_df_list = []
    all_latencies_overall = []
    all_durations_overall = []

    for session in comparison_sessions:
        session_phase_internal = session.get('phase')
        session_phase_display = display_phases_map.get(session_phase_internal) 
        if not session_phase_display: continue
        
        trial_results = session.get('trial_results', [])
        if not trial_results or len(trial_results) < 2: continue
            
        for i in range(1, len(trial_results)):
            prev_trial = trial_results[i-1]
            current_trial = trial_results[i]

            if prev_trial.get('rewarded', False):
                prev_size = prev_trial.get('reward_size')
                if prev_size in ALL_REWARD_SIZES:
                    # Get metrics
                    latency = current_trial.get('first_lick_latency')
                    cut_duration = None
                    lick_analysis_data = session.get('lick_sensor_cut_analysis', {})
                    if 'trial_metrics' in lick_analysis_data:
                        metrics = lick_analysis_data['trial_metrics']
                        current_trial_num = current_trial.get('trial_num')
                        metric_data = next((m for m in metrics if m.get('trial_num') == current_trial_num), None)
                        if metric_data and 'post_cue_cut_duration' in metric_data: cut_duration = metric_data['post_cue_cut_duration']
                    elif 'post_cue_cut_duration' in current_trial: cut_duration = current_trial['post_cue_cut_duration']
                    
                    latency_val = latency if (latency is not None and not np.isnan(latency)) else np.nan
                    duration_val = cut_duration if (cut_duration is not None and not np.isnan(cut_duration)) else np.nan
                    
                    # Add to overall lists for limit calculation
                    if not np.isnan(latency_val): all_latencies_overall.append(latency_val)
                    if not np.isnan(duration_val): all_durations_overall.append(duration_val)
                    
                    # Add data point to list for DataFrame
                    plot_df_list.append({
                        'prev_reward_size': prev_size,
                        'Phase': session_phase_display, # Use display name for plotting
                        'latency': latency_val,
                        'duration': duration_val
                    })
                    
    if not plot_df_list:
         print("  No valid data points found after pooling. Cannot generate plot.")
         return None
         
    plot_df = pd.DataFrame(plot_df_list)
    # Set Phase as categorical with the specific comparison order
    plot_df['Phase'] = pd.Categorical(plot_df['Phase'], categories=display_phases_order, ordered=True)
    # Ensure prev_reward_size is treated consistently (e.g., as category or int)
    plot_df['prev_reward_size'] = pd.Categorical(plot_df['prev_reward_size'], categories=ALL_REWARD_SIZES, ordered=True)
    plot_df.dropna(subset=['Phase', 'prev_reward_size'], inplace=True)

    # --- Plotting (5x2 Grid) --- 
    fig, axs = plt.subplots(len(ALL_REWARD_SIZES), 2, 
                             figsize=(11, 15), # Similar size to prev 5x2 plot
                             sharex='col', sharey='col') 
    if len(ALL_REWARD_SIZES) == 1: axs = np.array([axs]) # Ensure axs is 2D
        
    fig.suptitle(f"{animal_id} - Phase Comparison by Previous Reward Size", fontsize=14, y=0.99)

    plot_generated = False
    stats_results = {} # Store p-values: stats_results[(row, col)][(phase1, phase2)] = p_val

    # --- P-value Formatting Helper --- 
    def format_p_consistent(p):
        if np.isnan(p): return "p=nan"
        if p < 0.001: return f"p={p:.1e}" # Scientific format for p < 0.001
        else: # p >= 0.001
            # Format with 3 decimal places
            p_str = f"{p:.3f}"
            return f"p={p_str}"
    # ---------------------------------

    metrics_to_plot = ['latency', 'duration']
    # --- Match Figure 3 Titles --- 
    metric_titles = ["First Lick Delay (s)", "Pre Reward Lick Duration (s)"] 

    for i, prev_size in enumerate(ALL_REWARD_SIZES):
        df_subset_row = plot_df[plot_df['prev_reward_size'] == prev_size]
        prev_label = size_labels_map[prev_size]
        # --- Row title removed --- 
        
        # --- Clear default y-label for right column --- 
        axs[i, 1].set_ylabel("", fontsize=9)
        
        for col_idx, metric in enumerate(metrics_to_plot):
            ax = axs[i, col_idx]
            stats_key = (i, col_idx) # Use tuple (row, col) as key
            stats_results[stats_key] = {'means': {}, 'pairwise_p': {}} # Initialize storage

            # Apply common styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', size=6, width=1.5, labelsize=8)
            ax.grid(True, axis='y', alpha=0.5, linestyle='--')
            if metric == 'duration': ax.axhline(y=1.5, color='gray', linestyle=':', alpha=0.7, linewidth=1)

            if not df_subset_row[metric].dropna().empty:
                plot_generated = True
                # Violin + Stripplot
                sns.violinplot(x='Phase', y=metric, data=df_subset_row, order=display_phases_order, 
                               hue='Phase', palette=display_phase_colors, inner=None, 
                               linewidth=1.5, ax=ax, legend=False)
                sns.stripplot(x='Phase', y=metric, data=df_subset_row, order=display_phases_order, 
                              color='.25', size=3, alpha=0.5, jitter=0.15, ax=ax, legend=False)
                
                # --- Calculate Means and Perform Stats: Comparisons vs "Stress" --- 
                groups_data = {phase_display: df_subset_row[df_subset_row['Phase'] == phase_display][metric].dropna().tolist() 
                               for phase_display in display_phases_order}
                
                # Calculate and store means 
                for phase_idx, phase_name in enumerate(display_phases_order):
                    mean_val = np.mean(groups_data[phase_name]) if groups_data[phase_name] else np.nan
                    stats_results[stats_key]['means'][phase_idx] = mean_val # Store mean by phase index (0, 1, 2, 3)
                   
                stress_phase_name = display_phases_order[1] # "Stress"
                g_stress_data = groups_data.get(stress_phase_name, [])
                comparison_indices = [0, 2, 3] # Indices to compare against index 1
                
                for comp_idx in comparison_indices:
                    phase_to_compare = display_phases_order[comp_idx]
                    g_compare_data = groups_data.get(phase_to_compare, [])
                    p_val = np.nan
                    if g_stress_data and g_compare_data and \
                       len(g_stress_data) >= min_data_points_for_test and len(g_compare_data) >= min_data_points_for_test:
                        try: _, p = mannwhitneyu(g_stress_data, g_compare_data, alternative='two-sided'); p_val = p
                        except ValueError: p_val = 1.0 
                    # Store p-value using tuple key (stress_idx, compared_idx)
                    stats_results[stats_key]['pairwise_p'][(1, comp_idx)] = p_val 
            else:
                ax.text(0.5, 0.5, f'No {metric} data', ha='center', va='center', transform=ax.transAxes, color='gray', fontsize=9)

            # --- REMOVE Inset Legend --- 
            # legend_text = f"Prev Reward: {prev_label} ({prev_size})"
            # ax.text(0.03, 0.97, legend_text, transform=ax.transAxes, ...)
            # ---------------------------
            
            # Set titles: Column titles (top row), Row titles (left column)
            if i == 0: 
                # Set column titles only on the top row
                ax.set_title(metric_titles[col_idx], fontsize=11)
            elif col_idx == 0: # Left column, not top row
                # Set Row Title (Previous Reward Size) on the left plot
                row_title_text = f"Previous Reward Size: {prev_label}"
                ax.set_title(row_title_text, fontsize=10, loc='left') # Title on left plot
            else:
                 # Clear title for right plot if not top row
                 ax.set_title("")
                  
            # --- Set Y-axis Label (like Figure 3) --- 
            # Set label on the appropriate column axis for the middle row
            if i == len(ALL_REWARD_SIZES) // 2:
                ax.set_ylabel(metric_titles[col_idx], fontsize=10)
            else: # Clear ylabel for other rows
                ax.set_ylabel("")
            # -----------------------------------------
                 
            # Set x-axis labels only on the bottom row, REMOVE x-axis title
            if i == len(ALL_REWARD_SIZES) - 1: 
                ax.set_xticklabels(display_phases_order, rotation=45, ha='right')
                ax.set_xlabel("") # Remove X-axis label "Phase"
            else:
                ax.set_xticklabels([])
                ax.set_xlabel("")

    # --- Set shared Y-limits after all plots are drawn --- 
    if all_latencies_overall:
        lat_min, lat_max = np.percentile(all_latencies_overall, [1, 99])
        lat_pad = (lat_max - lat_min) * 0.1 if lat_max > lat_min else 0.1
        # Increase top padding slightly more for annotations
        plt.setp(axs[:, 0], ylim=(max(0, lat_min - lat_pad * 2), max(1.0, lat_max + lat_pad * 3.5))) 
    else: plt.setp(axs[:, 0], ylim=(0, 1.5)) 
        
    if all_durations_overall:
        dur_min, dur_max = np.percentile(all_durations_overall, [1, 99])
        dur_pad = (dur_max - dur_min) * 0.1 if dur_max > dur_min else 0.1
        # Increase top padding slightly more for annotations
        plt.setp(axs[:, 1], ylim=(max(0, dur_min - dur_pad * 2), min(1.55, max(0.1, dur_max + dur_pad * 3.5)))) 
    else: plt.setp(axs[:, 1], ylim=(0, 1.55))
        
    # --- Add Annotations (Means and P-values like Figure 3) --- 
    for (row, col), stats_dict in stats_results.items():
        ax = axs[row, col]
        ylim = ax.get_ylim() 
        y_range = ylim[1] - ylim[0]
        if y_range == 0: continue # Skip annotations if y-range is zero
        
        # Define positions relative to current axis limits
        y_mean_pos = ylim[0] + y_range * 0.80 # Adjusted position
        y_pval_pos = ylim[0] + y_range * 0.90 # Adjusted position
        mean_x_offset = 0.1 
        
        # --- Annotate Means --- 
        means_for_plot = stats_dict.get('means', {})
        for phase_idx, mean_val in means_for_plot.items():
             if not np.isnan(mean_val):
                 ax.text(phase_idx + mean_x_offset, y_mean_pos, f'{mean_val:.2f}', 
                         ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')

        # --- Annotate P-values --- 
        p_vals_for_plot = stats_dict.get('pairwise_p', {})
        # Compare 0 vs 1 -> place at x=0.5
        p_01 = p_vals_for_plot.get((1, 0), np.nan)
        ax.text(0.5, y_pval_pos, format_p_consistent(p_01), ha='center', va='bottom', fontsize=8, fontweight='bold')
        # Compare 1 vs 2 -> place at x=1.5
        p_12 = p_vals_for_plot.get((1, 2), np.nan)
        ax.text(1.5, y_pval_pos, format_p_consistent(p_12), ha='center', va='bottom', fontsize=8, fontweight='bold')
        # Compare 1 vs 3 -> place at x=2.5
        p_13 = p_vals_for_plot.get((1, 3), np.nan)
        ax.text(2.5, y_pval_pos, format_p_consistent(p_13), ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # --- Previous Star Annotations Removed ---
    
    plt.tight_layout(rect=[0.04, 0.05, 1, 0.97]) # Adjust layout

    # --- Save Figure --- 
    if not plot_generated:
        print("  No data generated for any comparison panel. Skipping save.")
        plt.close(fig)
        return None
        
    output_filename = os.path.join(output_dir, f"{animal_id}_phase_comparison_by_prev_reward.png")
    try:
        plt.savefig(output_filename, dpi=150)
        print(f"  Saved phase comparison by previous reward figure: {output_filename}")
    except Exception as e:
        print(f"  ERROR saving phase comparison by previous reward figure: {e}")
        output_filename = None
    finally:
        plt.close(fig)
        
    print(f"  Finished phase comparison by previous reward plot generation ({time.time() - start_time:.2f} seconds).")
    return output_filename


def visualize_time_resolved_licking_by_phase(test_sessions, animal_id, output_dir):
    """Generates a 5x1 plot showing time-resolved licking around reward delivery,
       comparing the 4 key phases (Pre-Stress, Stress, PL37, Post-Stress).
       Each panel corresponds to a different CURRENT reward size.
    """
    if not test_sessions:
        print("No test sessions provided for time-resolved phase comparison.")
        return None

    print(f"\nGenerating Time-Resolved Licking by Phase (Current Reward) plot for {animal_id}...")
    start_time = time.time()

    # --- Define Phases, Display Names, and Order (Consistent) --- 
    internal_phases = PHASE_ORDER 
    display_phases_map = {
        "Initial Test": "Pre Stress",
        "Stress Test (Pre-PL37)": "Stress",
        "PL37 Test": "PL37",
        "Post-PL37 Test": "Post Stress"
    }
    display_phases_order = [display_phases_map[p] for p in internal_phases if p in display_phases_map] 
    display_phase_colors = {display_phases_map[p]: PHASE_COLORS[p] for p in internal_phases if p in display_phases_map}
    # --------------------------------------------------------------------
    
    # --- Define Reward Sizes and Labels ---
    size_labels_map = {1: 'VS', 2: 'S', 3: 'M', 4: 'L', 5: 'VL'}
    ALL_REWARD_SIZES = list(size_labels_map.keys())

    # --- Analysis Parameters (Reuse from previous time-resolved functions) --- 
    ANALYSIS_START_TIME_REL_REWARD = -2.0
    TOTAL_DURATION_POST_REWARD = 8.0 
    ANALYSIS_END_TIME_REL_REWARD = TOTAL_DURATION_POST_REWARD
    WINDOW_SIZE = 0.1; STEP_SIZE = 0.02; TIME_RESOLUTION = 0.001 

    # --- Shared Calculation Setup --- (Reuse)
    time_points_relative_to_reward = np.arange(ANALYSIS_START_TIME_REL_REWARD + STEP_SIZE,
                                               ANALYSIS_END_TIME_REL_REWARD + STEP_SIZE,
                                               STEP_SIZE)
    time_points_relative_to_reward = time_points_relative_to_reward[time_points_relative_to_reward <= ANALYSIS_END_TIME_REL_REWARD + 1e-9]
    num_target_time_points = len(time_points_relative_to_reward)
    hires_trace_duration_rel_reward = ANALYSIS_END_TIME_REL_REWARD - ANALYSIS_START_TIME_REL_REWARD
    num_hires_points_total_analysis = int(np.round(hires_trace_duration_rel_reward / TIME_RESOLUTION))
    target_indices_in_hires = np.round((time_points_relative_to_reward - ANALYSIS_START_TIME_REL_REWARD) / TIME_RESOLUTION).astype(int) - 1
    target_indices_in_hires = np.clip(target_indices_in_hires, 0, num_hires_points_total_analysis - 1)
    win_samples = int(np.round(WINDOW_SIZE / TIME_RESOLUTION)); win_samples = max(1, win_samples)

    # --- Helper function to calculate single trial time series (Reuse/Redefine) --- 
    # Assuming this is available or copied correctly within the module scope
    # If not defined elsewhere, copy the full definition here.
    # Simplified version for brevity in this plan:
    def calculate_single_trial_ts(mtrial, reward_latency):
        lick_downs = mtrial.get('lick_downs_relative', []); lick_ups = mtrial.get('lick_ups_relative', [])
        analysis_start_rel_trial = reward_latency + ANALYSIS_START_TIME_REL_REWARD
        analysis_end_rel_trial = reward_latency + ANALYSIS_END_TIME_REL_REWARD
        max_time_rel_trial_needed = analysis_end_rel_trial + TIME_RESOLUTION
        num_hires_points_trial = int(np.ceil(max_time_rel_trial_needed / TIME_RESOLUTION))
        lick_trace_trial = np.zeros(num_hires_points_trial, dtype=np.int8)
        lick_downs_sorted = sorted([ld for ld in lick_downs if ld is not None])
        lick_ups_sorted = sorted([lu for lu in lick_ups if lu is not None])
        num_bouts = min(len(lick_downs_sorted), len(lick_ups_sorted))
        for i in range(num_bouts):
            start_idx = int(np.floor((lick_downs_sorted[i] + 1e-9) / TIME_RESOLUTION)); end_idx = int(np.ceil(lick_ups_sorted[i] / TIME_RESOLUTION))
            start_idx = max(0, start_idx); end_idx = min(num_hires_points_trial, end_idx)
            if start_idx < end_idx: lick_trace_trial[start_idx:end_idx] = 1
        idx_analysis_start_in_trial_trace = int(np.round(analysis_start_rel_trial / TIME_RESOLUTION))
        idx_analysis_end_in_trial_trace = idx_analysis_start_in_trial_trace + num_hires_points_total_analysis
        actual_start_idx = max(0, idx_analysis_start_in_trial_trace); actual_end_idx = min(len(lick_trace_trial), idx_analysis_end_in_trial_trace)
        extracted_trace = lick_trace_trial[actual_start_idx:actual_end_idx]
        pad_before = max(0, -idx_analysis_start_in_trial_trace); pad_after = max(0, num_hires_points_total_analysis - (len(extracted_trace) + pad_before))
        analysis_trace_hires = np.pad(extracted_trace, (pad_before, pad_after), 'constant')
        if len(analysis_trace_hires) != num_hires_points_total_analysis: return np.full(num_target_time_points, np.nan)
        rolling_proportions = pd.Series(analysis_trace_hires).rolling(window=win_samples, min_periods=1, center=True).mean().to_numpy()
        valid_indices_mask = target_indices_in_hires < len(rolling_proportions)
        valid_target_indices_for_sampling = target_indices_in_hires[valid_indices_mask]
        sampled_proportions = np.full(num_target_time_points, np.nan)
        sampled_proportions[valid_indices_mask] = rolling_proportions[valid_target_indices_for_sampling]
        return sampled_proportions
    # ----------------------------------------------------------------------

    # --- Data Pooling Structure --- 
    # pooled_data[display_phase_name][current_reward_size] = [trace1, trace2, ...]
    pooled_data = {phase_display: {curr_size: [] for curr_size in ALL_REWARD_SIZES} 
                   for phase_display in display_phases_order}

    # --- Data Processing Loop --- 
    print("  Pooling time series data...")
    sessions_processed = 0
    trials_pooled = 0
    for session in test_sessions:
        session_phase_internal = session.get('phase')
        session_phase_display = display_phases_map.get(session_phase_internal)
        if not session_phase_display: continue # Skip if phase not one of the 4 key ones
        
        monitoring_trials = session.get('trial_results', [])
        if not monitoring_trials: continue
        
        sessions_processed += 1
        # Group by Current Reward Size
        for mtrial in monitoring_trials:
            trial_num = mtrial.get('trial_num')
            is_rewarded = mtrial.get('rewarded', False)
            reward_latency = mtrial.get('reward_latency')
            reward_size = mtrial.get('reward_size') # CURRENT reward size
            
            if trial_num is None or not is_rewarded or reward_latency is None or reward_size not in ALL_REWARD_SIZES:
                continue
                
            try:
                sampled_proportions = calculate_single_trial_ts(mtrial, reward_latency)
                if sampled_proportions is not None and not np.all(np.isnan(sampled_proportions)):
                     pooled_data[session_phase_display][reward_size].append(sampled_proportions)
                     trials_pooled += 1
            except Exception as e:
                 # print(f"Warning: Error calculating TS for trial {trial_num}: {e}")
                 pass
                     
    print(f"  Processed {sessions_processed} sessions, pooled {trials_pooled} valid trial traces.")

    # --- Aggregation Function --- 
    def aggregate_ts_data(data_dict):
        aggregated_results = {phase: {} for phase in display_phases_order}
        for phase in data_dict:
            for r_size in data_dict[phase]:
                trial_list = data_dict[phase][r_size]
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
                aggregated_results[phase][r_size] = {'mean': mean_ts, 'sem': sem_ts, 'n': n_trials}
        return aggregated_results

    aggregated_data = aggregate_ts_data(pooled_data)

    # --- Plotting (5x1 Grid) --- 
    fig, axs = plt.subplots(len(ALL_REWARD_SIZES), 1, 
                             figsize=(10, 12), # Adjusted figsize for 5x1
                             sharex=True, sharey=False) # Changed sharey to False
    if len(ALL_REWARD_SIZES) == 1: axs = [axs] # Make iterable
        
    fig.suptitle(f"{animal_id} - Time-Resolved Licking by Phase (Grouped by Current Reward)", fontsize=14, y=0.98)

    plot_successful = False
    max_y_overall = 0

    for row_idx, curr_size in enumerate(ALL_REWARD_SIZES):
        ax = axs[row_idx]
        curr_label = size_labels_map[curr_size]
        panel_title = f"Current Reward Size: {curr_label} ({curr_size})"
        ax.set_title(panel_title, fontsize=13, loc='center') # Increased panel title size
        panel_has_data = False
        legend_handles = []
        legend_labels = []
        
        for phase in display_phases_order:
            agg_res = aggregated_data[phase].get(curr_size, {})
            mean_ts = agg_res.get('mean')
            sem_ts = agg_res.get('sem')
            n_val = agg_res.get('n', 0)
            
            if mean_ts is not None and sem_ts is not None and n_val > 0:
                panel_has_data = True
                plot_successful = True
                color = display_phase_colors.get(phase, 'gray')
                label = f"{phase} (N={n_val})" 
                line, = ax.plot(time_points_relative_to_reward, mean_ts, color=color, label=label, linewidth=1.5)
                ax.fill_between(time_points_relative_to_reward, mean_ts - sem_ts, mean_ts + sem_ts, color=color, alpha=0.2)
                max_y_overall = max(max_y_overall, np.nanmax(mean_ts + sem_ts) if mean_ts is not None else 0)
                legend_handles.append(line)
                legend_labels.append(label)
        
        ax.axvline(x=0, color='black', linestyle=':', linewidth=1, alpha=0.7, label='_nolegend_')
        ax.grid(True, alpha=0.5, linestyle='--')

        # Set tick label size
        ax.tick_params(axis='both', which='major', labelsize=8)

        if not panel_has_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, color='gray')
        else:
            # Add legend to this specific subplot
            ax.legend(handles=legend_handles, labels=legend_labels, loc='upper right', fontsize='x-small', title="")

        # X-axis Label (Bottom Row Only)
        if row_idx == len(ALL_REWARD_SIZES) - 1:
            ax.set_xlabel("Time Relative to Reward (s)", fontsize=12) # Increased X-axis label size
        
        # Shared Y-axis label (Add outside loop)

    # Set shared axes limits & Label
    # plt.setp(axs, ylim=(-0.05, max(1.05, max_y_overall * 1.1))) 
    plt.setp(axs, xlim=(ANALYSIS_START_TIME_REL_REWARD, ANALYSIS_END_TIME_REL_REWARD))
    fig.text(0.01, 0.5, "Mean Lick Proportion", va='center', rotation='vertical', fontsize=14) # Increased shared Y-axis label size
        
    # Final layout adjustment
    plt.tight_layout(rect=[0.05, 0.03, 1, 0.96]) # Adjust margins

    # --- Save Figure --- 
    if not plot_successful:
        print("  No data plotted for any panel. Skipping save.")
        plt.close(fig)
        return None
        
    output_filename = os.path.join(output_dir, f"{animal_id}_time_resolved_licking_by_phase.png")
    try:
        plt.savefig(output_filename, dpi=150)
        print(f"  Saved time-resolved licking by phase figure: {output_filename}")
    except Exception as e:
        print(f"  ERROR saving time-resolved licking by phase figure: {e}")
        output_filename = None
    finally:
        plt.close(fig)
        
    print(f"  Finished time-resolved licking by phase plot generation ({time.time() - start_time:.2f} seconds).")
    return output_filename


def visualize_time_resolved_licking_by_reward_size(test_sessions, animal_id, output_dir):
    """Generates a 1x4 plot showing time-resolved licking around reward delivery,
       comparing the 5 different CURRENT reward sizes within each phase.
       Each panel corresponds to a different phase.
    """
    if not test_sessions:
        print("No test sessions provided for time-resolved reward size comparison.")
        return None

    print(f"\nGenerating Time-Resolved Licking by Reward Size (per Phase) plot for {animal_id}...")
    start_time = time.time()

    # --- Define Phases, Display Names, and Order (Consistent) --- 
    internal_phases = PHASE_ORDER 
    display_phases_map = {
        "Initial Test": "Pre Stress",
        "Stress Test (Pre-PL37)": "Stress",
        "PL37 Test": "PL37",
        "Post-PL37 Test": "Post Stress"
    }
    display_phases_order = [display_phases_map[p] for p in internal_phases if p in display_phases_map] 
    # display_phase_colors not needed directly for plotting here, colors based on reward size
    # --------------------------------------------------------------------
    
    # --- Define Reward Sizes, Labels, and Colormap ---
    size_labels_map = {1: 'VS', 2: 'S', 3: 'M', 4: 'L', 5: 'VL'}
    ALL_REWARD_SIZES = list(size_labels_map.keys())
    reward_size_cmap = plt.cm.viridis # Use Viridis for reward sizes
    reward_size_norm = mcolors.Normalize(vmin=min(ALL_REWARD_SIZES), vmax=max(ALL_REWARD_SIZES))
    # --------------------------------------------------

    # --- Analysis Parameters (Reuse) --- 
    ANALYSIS_START_TIME_REL_REWARD = -2.0
    ANALYSIS_END_TIME_REL_REWARD = 8.0 
    WINDOW_SIZE = 0.1; STEP_SIZE = 0.02; TIME_RESOLUTION = 0.001 

    # --- Shared Calculation Setup --- (Reuse)
    time_points_relative_to_reward = np.arange(ANALYSIS_START_TIME_REL_REWARD + STEP_SIZE,
                                               ANALYSIS_END_TIME_REL_REWARD + STEP_SIZE,
                                               STEP_SIZE)
    time_points_relative_to_reward = time_points_relative_to_reward[time_points_relative_to_reward <= ANALYSIS_END_TIME_REL_REWARD + 1e-9]
    num_target_time_points = len(time_points_relative_to_reward)
    hires_trace_duration_rel_reward = ANALYSIS_END_TIME_REL_REWARD - ANALYSIS_START_TIME_REL_REWARD
    num_hires_points_total_analysis = int(np.round(hires_trace_duration_rel_reward / TIME_RESOLUTION))
    target_indices_in_hires = np.round((time_points_relative_to_reward - ANALYSIS_START_TIME_REL_REWARD) / TIME_RESOLUTION).astype(int) - 1
    target_indices_in_hires = np.clip(target_indices_in_hires, 0, num_hires_points_total_analysis - 1)
    win_samples = int(np.round(WINDOW_SIZE / TIME_RESOLUTION)); win_samples = max(1, win_samples)

    # --- Helper function (Assume available or copied) --- 
    def calculate_single_trial_ts(mtrial, reward_latency):
        # ... (Full implementation needed if not defined elsewhere in module) ...
        lick_downs = mtrial.get('lick_downs_relative', []); lick_ups = mtrial.get('lick_ups_relative', [])
        analysis_start_rel_trial = reward_latency + ANALYSIS_START_TIME_REL_REWARD
        analysis_end_rel_trial = reward_latency + ANALYSIS_END_TIME_REL_REWARD
        max_time_rel_trial_needed = analysis_end_rel_trial + TIME_RESOLUTION
        num_hires_points_trial = int(np.ceil(max_time_rel_trial_needed / TIME_RESOLUTION))
        lick_trace_trial = np.zeros(num_hires_points_trial, dtype=np.int8)
        lick_downs_sorted = sorted([ld for ld in lick_downs if ld is not None])
        lick_ups_sorted = sorted([lu for lu in lick_ups if lu is not None])
        num_bouts = min(len(lick_downs_sorted), len(lick_ups_sorted))
        for i in range(num_bouts):
            start_idx = int(np.floor((lick_downs_sorted[i] + 1e-9) / TIME_RESOLUTION)); end_idx = int(np.ceil(lick_ups_sorted[i] / TIME_RESOLUTION))
            start_idx = max(0, start_idx); end_idx = min(num_hires_points_trial, end_idx)
            if start_idx < end_idx: lick_trace_trial[start_idx:end_idx] = 1
        idx_analysis_start_in_trial_trace = int(np.round(analysis_start_rel_trial / TIME_RESOLUTION))
        idx_analysis_end_in_trial_trace = idx_analysis_start_in_trial_trace + num_hires_points_total_analysis
        actual_start_idx = max(0, idx_analysis_start_in_trial_trace); actual_end_idx = min(len(lick_trace_trial), idx_analysis_end_in_trial_trace)
        extracted_trace = lick_trace_trial[actual_start_idx:actual_end_idx]
        pad_before = max(0, -idx_analysis_start_in_trial_trace); pad_after = max(0, num_hires_points_total_analysis - (len(extracted_trace) + pad_before))
        analysis_trace_hires = np.pad(extracted_trace, (pad_before, pad_after), 'constant')
        if len(analysis_trace_hires) != num_hires_points_total_analysis: return np.full(num_target_time_points, np.nan)
        rolling_proportions = pd.Series(analysis_trace_hires).rolling(window=win_samples, min_periods=1, center=True).mean().to_numpy()
        valid_indices_mask = target_indices_in_hires < len(rolling_proportions)
        valid_target_indices_for_sampling = target_indices_in_hires[valid_indices_mask]
        sampled_proportions = np.full(num_target_time_points, np.nan)
        sampled_proportions[valid_indices_mask] = rolling_proportions[valid_target_indices_for_sampling]
        return sampled_proportions
    # ------------------------------------------------------

    # --- Data Pooling Structure & Loop (Same as previous function) --- 
    pooled_data = {phase_display: {curr_size: [] for curr_size in ALL_REWARD_SIZES} 
                   for phase_display in display_phases_order}
    print("  Pooling time series data...")
    sessions_processed = 0; trials_pooled = 0
    for session in test_sessions:
        session_phase_internal = session.get('phase')
        session_phase_display = display_phases_map.get(session_phase_internal)
        if not session_phase_display: continue
        monitoring_trials = session.get('trial_results', [])
        if not monitoring_trials: continue
        sessions_processed += 1
        for mtrial in monitoring_trials:
            trial_num = mtrial.get('trial_num'); is_rewarded = mtrial.get('rewarded', False)
            reward_latency = mtrial.get('reward_latency'); reward_size = mtrial.get('reward_size')
            if trial_num is None or not is_rewarded or reward_latency is None or reward_size not in ALL_REWARD_SIZES: continue
            try:
                sampled_proportions = calculate_single_trial_ts(mtrial, reward_latency)
                if sampled_proportions is not None and not np.all(np.isnan(sampled_proportions)):
                     pooled_data[session_phase_display][reward_size].append(sampled_proportions); trials_pooled += 1
            except Exception: pass
    print(f"  Processed {sessions_processed} sessions, pooled {trials_pooled} valid trial traces.")

    # --- Aggregation Function (Same as previous function) --- 
    def aggregate_ts_data(data_dict):
        aggregated_results = {phase: {} for phase in display_phases_order}
        for phase in data_dict:
            for r_size in data_dict[phase]:
                trial_list = data_dict[phase][r_size]
                valid_trials_ts = [ts for ts in trial_list if ts is not None and not np.all(np.isnan(ts)) and len(ts) == num_target_time_points]
                n_trials = len(valid_trials_ts); mean_ts, sem_ts = None, None
                if n_trials > 0:
                    trial_ts_stack = np.array(valid_trials_ts); mean_ts = np.nanmean(trial_ts_stack, axis=0)
                    if n_trials > 1:
                        with np.errstate(invalid='ignore'): sem_ts = sem(trial_ts_stack, axis=0, nan_policy='omit'); sem_ts = np.nan_to_num(sem_ts, nan=0.0)
                    else: sem_ts = np.zeros_like(mean_ts)
                aggregated_results[phase][r_size] = {'mean': mean_ts, 'sem': sem_ts, 'n': n_trials}
        return aggregated_results
    aggregated_data = aggregate_ts_data(pooled_data)

    # --- Plotting (4x1 Grid) --- 
    n_phases = len(display_phases_order)
    fig, axs = plt.subplots(n_phases, 1, 
                             figsize=(8, 10), # Adjusted figsize for 4x1
                             sharex=True, sharey=False) # Changed sharey to False
    if n_phases == 1: axs = [axs] # Make iterable if only one phase
        
    fig.suptitle(f"{animal_id} - Time-Resolved Licking by Reward Size (per Phase)", fontsize=14, y=0.98) # Adjust y slightly for taller fig

    plot_successful = False
    max_y_overall = 0

    for row_idx, phase in enumerate(display_phases_order):
        ax = axs[row_idx]
        ax.set_title(phase, fontsize=11) # Panel title is the Phase name
        panel_has_data = False
        legend_handles = []
        legend_labels = []
        
        for r_size in ALL_REWARD_SIZES:
            agg_res = aggregated_data[phase].get(r_size, {})
            mean_ts = agg_res.get('mean')
            sem_ts = agg_res.get('sem')
            n_val = agg_res.get('n', 0)
            
            if mean_ts is not None and sem_ts is not None and n_val > 0:
                panel_has_data = True
                plot_successful = True
                color = reward_size_cmap(reward_size_norm(r_size)) # Color by reward size
                size_label_short = size_labels_map.get(r_size, str(r_size))
                label = f"{size_label_short} (N={n_val})" 
                line, = ax.plot(time_points_relative_to_reward, mean_ts, color=color, label=label, linewidth=1.5)
                ax.fill_between(time_points_relative_to_reward, mean_ts - sem_ts, mean_ts + sem_ts, color=color, alpha=0.15)
                max_y_overall = max(max_y_overall, np.nanmax(mean_ts + sem_ts) if mean_ts is not None else 0)
                legend_handles.append(line)
                legend_labels.append(label)
        
        ax.axvline(x=0, color='black', linestyle=':', linewidth=1, alpha=0.7, label='_nolegend_')
        ax.grid(True, alpha=0.5, linestyle='--')

        if not panel_has_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, color='gray')
        else:
            ax.legend(handles=legend_handles, labels=legend_labels, loc='upper right', fontsize=8, title="") # Explicit font size

        # --- Apply Y-ticks and Tick Label Size INSIDE the loop for each ax ---
        ax.yaxis.set_major_locator(FixedLocator([0, 0.5, 1.0]))
        ax.yaxis.set_major_formatter(ScalarFormatter()) # Ensures standard number formatting
        ax.tick_params(axis='both', which='major', labelsize=8) 
        # ---------------------------------------------------------------------

        # Set X-axis label only on the bottom plot
        if row_idx == n_phases - 1:
            ax.set_xlabel("Time Relative to Reward (s)", fontsize=10)
        else:
            ax.set_xlabel("")

        # Set Y-axis label (handled by fig.text)
        ax.set_ylabel("") # Clear individual y-labels

    # Set shared axes limits
    plt.setp(axs, ylim=(-0.05, max(1.05, max_y_overall * 1.1))) 
    plt.setp(axs, xlim=(ANALYSIS_START_TIME_REL_REWARD, ANALYSIS_END_TIME_REL_REWARD))
    
    # Add shared Y-axis label to the figure
    fig.text(0.02, 0.5, "Mean Lick Proportion", va='center', ha='center', rotation='vertical', fontsize=14) # Match fontsize from other plot
         
    # Final layout adjustment
    plt.tight_layout(rect=[0.05, 0.03, 1, 0.96]) # Adjust margins for shared label and title

    # --- Save Figure --- 
    if not plot_successful:
        print("  No data plotted for any panel. Skipping save.")
        plt.close(fig)
        return None
        
    output_filename = os.path.join(output_dir, f"{animal_id}_time_resolved_licking_by_reward_size.png")
    try:
        plt.savefig(output_filename, dpi=150)
        print(f"  Saved time-resolved licking by reward size figure: {output_filename}")
    except Exception as e:
        print(f"  ERROR saving time-resolved licking by reward size figure: {e}")
        output_filename = None
    finally:
        plt.close(fig)
        
    print(f"  Finished time-resolved licking by reward size plot generation ({time.time() - start_time:.2f} seconds).")
    return output_filename


def visualize_pre_vs_stress_prev_extremes(test_sessions, animal_id, output_dir):
    """Generates a 5x2 plot comparing Pre-Stress vs. Stress time-resolved licking,
       for each current reward size, grouped by previous reward extremes (VS/S vs L/VL).
    """
    if not test_sessions:
        print("No test sessions provided for Pre vs Stress prev extremes summary.")
        return None

    print(f"\nGenerating Pre-Stress vs Stress Previous Extremes Summary plot for {animal_id}...")
    start_time = time.time()

    # --- Define Conditions and Internal Phase Names --- 
    condition_map = {
        "Pre Stress": "Initial Test",
        "Stress": "Stress Test (Pre-PL37)"
    }
    conditions_to_plot = list(condition_map.keys()) # ["Pre Stress", "Stress"]
    # --------------------------------------------------
    
    # --- Define Reward Sizes, Labels, and Groupings --- 
    size_labels_map = {1: 'VS', 2: 'S', 3: 'M', 4: 'L', 5: 'VL'} # For current reward size labels
    ALL_REWARD_SIZES = list(size_labels_map.keys())
    PREV_VS_S_SIZES = [1, 2]
    PREV_L_VL_SIZES = [4, 5]
    PREV_EXTREME_LABELS = {
        "Prev VS/S": PREV_VS_S_SIZES,
        "Prev L/VL": PREV_L_VL_SIZES
    }
    # Consistent colors for the two extreme groups
    prev_extreme_colors = {"Prev VS/S": 'royalblue', "Prev L/VL": 'darkorange'}
    # ----------------------------------------------------

    # --- Analysis Parameters (Reuse) --- 
    ANALYSIS_START_TIME_REL_REWARD = -2.0
    ANALYSIS_END_TIME_REL_REWARD = 8.0 
    WINDOW_SIZE = 0.1; STEP_SIZE = 0.02; TIME_RESOLUTION = 0.001 

    # --- Shared Calculation Setup --- (Reuse)
    time_points_relative_to_reward = np.arange(ANALYSIS_START_TIME_REL_REWARD + STEP_SIZE,
                                               ANALYSIS_END_TIME_REL_REWARD + STEP_SIZE,
                                               STEP_SIZE)
    time_points_relative_to_reward = time_points_relative_to_reward[time_points_relative_to_reward <= ANALYSIS_END_TIME_REL_REWARD + 1e-9]
    num_target_time_points = len(time_points_relative_to_reward)
    hires_trace_duration_rel_reward = ANALYSIS_END_TIME_REL_REWARD - ANALYSIS_START_TIME_REL_REWARD
    num_hires_points_total_analysis = int(np.round(hires_trace_duration_rel_reward / TIME_RESOLUTION))
    target_indices_in_hires = np.round((time_points_relative_to_reward - ANALYSIS_START_TIME_REL_REWARD) / TIME_RESOLUTION).astype(int) - 1
    target_indices_in_hires = np.clip(target_indices_in_hires, 0, num_hires_points_total_analysis - 1)
    win_samples = int(np.round(WINDOW_SIZE / TIME_RESOLUTION)); win_samples = max(1, win_samples)

    # --- Helper function (Assume available or copied) --- 
    def calculate_single_trial_ts(mtrial, reward_latency):
        lick_downs = mtrial.get('lick_downs_relative', []); lick_ups = mtrial.get('lick_ups_relative', [])
        analysis_start_rel_trial = reward_latency + ANALYSIS_START_TIME_REL_REWARD
        analysis_end_rel_trial = reward_latency + ANALYSIS_END_TIME_REL_REWARD
        max_time_rel_trial_needed = analysis_end_rel_trial + TIME_RESOLUTION
        num_hires_points_trial = int(np.ceil(max_time_rel_trial_needed / TIME_RESOLUTION))
        lick_trace_trial = np.zeros(num_hires_points_trial, dtype=np.int8)
        lick_downs_sorted = sorted([ld for ld in lick_downs if ld is not None])
        lick_ups_sorted = sorted([lu for lu in lick_ups if lu is not None])
        num_bouts = min(len(lick_downs_sorted), len(lick_ups_sorted))
        for i in range(num_bouts):
            start_idx = int(np.floor((lick_downs_sorted[i] + 1e-9) / TIME_RESOLUTION)); end_idx = int(np.ceil(lick_ups_sorted[i] / TIME_RESOLUTION))
            start_idx = max(0, start_idx); end_idx = min(num_hires_points_trial, end_idx)
            if start_idx < end_idx: lick_trace_trial[start_idx:end_idx] = 1
        idx_analysis_start_in_trial_trace = int(np.round(analysis_start_rel_trial / TIME_RESOLUTION))
        idx_analysis_end_in_trial_trace = idx_analysis_start_in_trial_trace + num_hires_points_total_analysis
        actual_start_idx = max(0, idx_analysis_start_in_trial_trace); actual_end_idx = min(len(lick_trace_trial), idx_analysis_end_in_trial_trace)
        extracted_trace = lick_trace_trial[actual_start_idx:actual_end_idx]
        pad_before = max(0, -idx_analysis_start_in_trial_trace); pad_after = max(0, num_hires_points_total_analysis - (len(extracted_trace) + pad_before))
        analysis_trace_hires = np.pad(extracted_trace, (pad_before, pad_after), 'constant')
        if len(analysis_trace_hires) != num_hires_points_total_analysis: return np.full(num_target_time_points, np.nan)
        rolling_proportions = pd.Series(analysis_trace_hires).rolling(window=win_samples, min_periods=1, center=True).mean().to_numpy()
        valid_indices_mask = target_indices_in_hires < len(rolling_proportions)
        valid_target_indices_for_sampling = target_indices_in_hires[valid_indices_mask]
        sampled_proportions = np.full(num_target_time_points, np.nan)
        sampled_proportions[valid_indices_mask] = rolling_proportions[valid_target_indices_for_sampling]
        return sampled_proportions
    # ------------------------------------------------------

    # --- Data Pooling Structure & Loop --- 
    # pooled_data[condition_label][current_reward_size][prev_extreme_label] = [trace1, ...]
    pooled_data = {cond: {curr_size: {prev_label: [] for prev_label in PREV_EXTREME_LABELS} 
                           for curr_size in ALL_REWARD_SIZES} 
                   for cond in conditions_to_plot}
    
    session_days_by_condition = {cond: set() for cond in conditions_to_plot}
    trials_pooled_count = 0

    print("  Pooling time series data for Pre-Stress vs Stress by previous extremes...")
    for session in test_sessions:
        session_phase_internal = session.get('phase')
        current_condition = None
        for cond_label, internal_phase_name in condition_map.items():
            if session_phase_internal == internal_phase_name:
                current_condition = cond_label
                break
        if not current_condition: continue # Skip if not Pre-Stress or Stress

        session_days_by_condition[current_condition].add(session.get('session_day'))
        monitoring_trials = session.get('trial_results', [])
        if not monitoring_trials or len(monitoring_trials) < 2: continue
            
        for idx in range(1, len(monitoring_trials)):
            mtrial_curr = monitoring_trials[idx]
            mtrial_prev = monitoring_trials[idx-1]
            
            # Current trial info
            trial_num_curr = mtrial_curr.get('trial_num')
            is_rewarded_curr = mtrial_curr.get('rewarded', False)
            reward_latency_curr = mtrial_curr.get('reward_latency')
            reward_size_curr = mtrial_curr.get('reward_size')

            if not (trial_num_curr is not None and is_rewarded_curr and 
                    reward_latency_curr is not None and reward_size_curr in ALL_REWARD_SIZES):
                continue

            # Previous trial info
            is_rewarded_prev = mtrial_prev.get('rewarded', False)
            reward_size_prev = mtrial_prev.get('reward_size')

            if not (is_rewarded_prev and reward_size_prev is not None):
                continue

            prev_extreme_key = None
            if reward_size_prev in PREV_VS_S_SIZES: prev_extreme_key = "Prev VS/S"
            elif reward_size_prev in PREV_L_VL_SIZES: prev_extreme_key = "Prev L/VL"
            else: continue # Previous reward not in extreme groups
            
            try:
                sampled_proportions = calculate_single_trial_ts(mtrial_curr, reward_latency_curr)
                if sampled_proportions is not None and not np.all(np.isnan(sampled_proportions)):
                     pooled_data[current_condition][reward_size_curr][prev_extreme_key].append(sampled_proportions)
                     trials_pooled_count +=1
            except Exception: pass # Ignore errors for single trial processing
    
    print("  Session days included:")
    for cond, days_set in session_days_by_condition.items():
        print(f"    - {cond}: {len(days_set)} days")
    print(f"  Total valid trial traces pooled: {trials_pooled_count}")

    # --- Aggregation Function --- 
    def aggregate_extremes_data(data_dict):
        aggregated_results = {cond: {curr_size: {} for curr_size in ALL_REWARD_SIZES} for cond in conditions_to_plot}
        for cond in data_dict:
            for curr_size in data_dict[cond]:
                for prev_label in data_dict[cond][curr_size]:
                    trial_list = data_dict[cond][curr_size][prev_label]
                    valid_trials_ts = [ts for ts in trial_list if ts is not None and not np.all(np.isnan(ts)) and len(ts) == num_target_time_points]
                    n_trials = len(valid_trials_ts); mean_ts, sem_ts = None, None
                    if n_trials > 0:
                        trial_ts_stack = np.array(valid_trials_ts); mean_ts = np.nanmean(trial_ts_stack, axis=0)
                        if n_trials > 1:
                            with np.errstate(invalid='ignore'): sem_ts = sem(trial_ts_stack, axis=0, nan_policy='omit'); sem_ts = np.nan_to_num(sem_ts, nan=0.0)
                        else: sem_ts = np.zeros_like(mean_ts) if mean_ts is not None else None
                    aggregated_results[cond][curr_size][prev_label] = {'mean': mean_ts, 'sem': sem_ts, 'n': n_trials}
        return aggregated_results
    aggregated_data = aggregate_extremes_data(pooled_data)

    # --- Plotting (5 Rows for Current Reward Size x 2 Columns for Condition) --- 
    fig, axs = plt.subplots(len(ALL_REWARD_SIZES), len(conditions_to_plot), 
                             figsize=(10, 12), # Adjust as needed
                             sharex=True, sharey=True)
    if len(ALL_REWARD_SIZES) == 1 and len(conditions_to_plot) == 1: axs = np.array([[axs]]) # Ensure 2D for single case
    elif len(ALL_REWARD_SIZES) == 1: axs = axs.reshape(1, -1) # Ensure 2D if only one row
    elif len(conditions_to_plot) == 1: axs = axs.reshape(-1, 1) # Ensure 2D if only one col
        
    fig.suptitle(f"{animal_id} - Pre-Stress vs Stress: Licking by Previous Reward Extremes", fontsize=14, y=0.98)

    plot_successful = False; max_y_overall = 0

    for row_idx, curr_size in enumerate(ALL_REWARD_SIZES):
        curr_reward_label = size_labels_map[curr_size]
        # Set row title on the Y-axis of the first plot in the row
        axs[row_idx, 0].set_ylabel(f"Curr: {curr_reward_label} ({curr_size})", fontsize=9)

        for col_idx, condition in enumerate(conditions_to_plot):
            ax = axs[row_idx, col_idx]
            panel_has_data = False
            legend_handles = []; legend_labels = []
            
            for prev_label, prev_color in prev_extreme_colors.items():
                agg_res = aggregated_data[condition][curr_size].get(prev_label, {})
                mean_ts = agg_res.get('mean'); sem_ts = agg_res.get('sem'); n_val = agg_res.get('n', 0)
                
                if mean_ts is not None and sem_ts is not None and n_val > 0:
                    panel_has_data = True; plot_successful = True
                    label_text = f"{prev_label} (N={n_val})"
                    line, = ax.plot(time_points_relative_to_reward, mean_ts, color=prev_color, label=label_text, linewidth=1.5)
                    ax.fill_between(time_points_relative_to_reward, mean_ts - sem_ts, mean_ts + sem_ts, color=prev_color, alpha=0.15)
                    max_y_overall = max(max_y_overall, np.nanmax(mean_ts + sem_ts) if mean_ts is not None else 0)
                    legend_handles.append(line); legend_labels.append(label_text)
            
            ax.axvline(x=0, color='black', linestyle=':', linewidth=1, alpha=0.7)
            ax.grid(True, alpha=0.5, linestyle='--')
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(['0.0', '0.5', '1.0'])
            ax.set_ylim(-0.05, 1.05) # Fixed Y-limit for consistency

            if not panel_has_data:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, color='gray')
            else:
                ax.legend(handles=legend_handles, labels=legend_labels, loc='upper right', fontsize=8, title="Prev. Reward")

            if row_idx == 0: # Top row: Set column titles (Conditions)
                ax.set_title(condition, fontsize=11)
            if row_idx == len(ALL_REWARD_SIZES) - 1: # Bottom row: X-axis label
                ax.set_xlabel("Time Relative to Reward (s)", fontsize=10)
            else:
                ax.set_xlabel("")
            if col_idx > 0 : ax.set_ylabel("") # Clear y-label for right column

    # Shared Y-axis label (if not using row titles as y-labels, adjust position)
    # fig.text(0.01, 0.5, "Mean Lick Proportion", va='center', rotation='vertical', fontsize=12)
        
    plt.tight_layout(rect=[0.05, 0.03, 1, 0.96])

    # --- Save Figure --- 
    if not plot_successful: print("  No data plotted. Skipping save."); plt.close(fig); return None
    output_filename = os.path.join(output_dir, f"{animal_id}_pre_vs_stress_prev_extremes.png")
    try:
        plt.savefig(output_filename, dpi=150); print(f"  Saved: {output_filename}")
    except Exception as e: print(f"  ERROR saving figure: {e}"); output_filename = None
    finally: plt.close(fig)
    print(f"  Finished Pre-Stress vs Stress Previous Extremes plot ({time.time() - start_time:.2f}s).")
    return output_filename


def visualize_time_resolved_licking_single_phase(test_sessions, animal_id, output_dir, target_phase, plot_title_prefix=""):
    """
    Generates a single plot of time-resolved licking probabilities for a specific experimental phase,
    with individual lines for different reward sizes.

    This is a variant of `visualize_time_resolved_licking_by_reward_size` that focuses on one phase.

    Parameters:
    -----------
    test_sessions : list
        A list of dictionaries, each containing processed data for a test session.
    animal_id : str
        Identifier for the animal.
    output_dir : str
        Directory to save the output figure.
    target_phase : str
        The specific phase to plot (e.g., "Initial Test").
    plot_title_prefix : str, optional
        A prefix for the plot title, e.g., "Pre-Stress".
    
    Returns:
    --------
    str or None
        The absolute path to the saved figure file, or None if saving failed or no data.
    """
    if not test_sessions:
        print("No test sessions provided for single-phase time-resolved licking plot.")
        return None
    
    print(f"\nGenerating Single-Phase Time-Resolved Licking plot for phase '{target_phase}' for {animal_id}...")
    start_time = time.time()

    # --- Analysis Parameters (Copied from visualize_time_resolved_licking_by_reward_size) ---
    ANALYSIS_START_TIME_REL_REWARD = -2.0
    TOTAL_DURATION_POST_REWARD = 8.0 
    ANALYSIS_END_TIME_REL_REWARD = TOTAL_DURATION_POST_REWARD
    WINDOW_SIZE = 0.1
    STEP_SIZE = 0.02
    TIME_RESOLUTION = 0.001
    
    # Reward category definitions
    REWARD_CATEGORY_STRINGS = ["Very Small", "Small", "Medium", "Large", "Very Large"] 
    NUMERICAL_SIZE_TO_CATEGORY_STRING_MAP = {1: "Very Small", 2: "Small", 3: "Medium", 4: "Large", 5: "Very Large"}
    CATEGORY_STRING_TO_NUMERICAL_SIZE_MAP = {v: k for k, v in NUMERICAL_SIZE_TO_CATEGORY_STRING_MAP.items()}
    EXPECTED_NUMERICAL_SIZES = list(NUMERICAL_SIZE_TO_CATEGORY_STRING_MAP.keys())

    # --- Shared Calculation Setup (Copied) ---
    time_points_relative_to_reward = np.arange(ANALYSIS_START_TIME_REL_REWARD + STEP_SIZE,
                                               ANALYSIS_END_TIME_REL_REWARD + STEP_SIZE,
                                               STEP_SIZE)
    time_points_relative_to_reward = time_points_relative_to_reward[time_points_relative_to_reward <= ANALYSIS_END_TIME_REL_REWARD + 1e-9] 
    num_target_time_points = len(time_points_relative_to_reward)
    hires_trace_duration_rel_reward = ANALYSIS_END_TIME_REL_REWARD - ANALYSIS_START_TIME_REL_REWARD
    num_hires_points_total_analysis = int(np.round(hires_trace_duration_rel_reward / TIME_RESOLUTION))
    target_indices_in_hires = np.round((time_points_relative_to_reward - ANALYSIS_START_TIME_REL_REWARD) / TIME_RESOLUTION).astype(int) - 1
    target_indices_in_hires = np.clip(target_indices_in_hires, 0, num_hires_points_total_analysis - 1)
    win_samples = int(np.round(WINDOW_SIZE / TIME_RESOLUTION)); win_samples = max(1, win_samples)

    # --- Helper function for calculation (Copied) ---
    def calculate_single_trial_ts(mtrial, reward_latency):
        # This function is identical to the one in `visualize_time_resolved_licking_by_reward_size`
        lick_downs = mtrial.get('lick_downs_relative', []); lick_ups = mtrial.get('lick_ups_relative', [])
        lick_downs = lick_downs if lick_downs is not None else [] ; lick_ups = lick_ups if lick_ups is not None else []
        analysis_start_rel_trial = reward_latency + ANALYSIS_START_TIME_REL_REWARD
        analysis_end_rel_trial = reward_latency + ANALYSIS_END_TIME_REL_REWARD
        max_time_rel_trial_needed = analysis_end_rel_trial + TIME_RESOLUTION 
        num_hires_points_trial = int(np.ceil(max_time_rel_trial_needed / TIME_RESOLUTION))
        lick_trace_trial = np.zeros(num_hires_points_trial, dtype=np.int8)
        lick_downs_sorted = sorted([ld for ld in lick_downs if isinstance(ld, (int, float, np.number))])
        lick_ups_sorted = sorted([lu for lu in lick_ups if isinstance(lu, (int, float, np.number))])
        num_bouts = min(len(lick_downs_sorted), len(lick_ups_sorted))
        for i in range(num_bouts):
            start_time = lick_downs_sorted[i]; end_time = lick_ups_sorted[i]
            if end_time > start_time:
                start_idx = int(np.floor((start_time + 1e-9) / TIME_RESOLUTION))
                end_idx = int(np.ceil(end_time / TIME_RESOLUTION))
                start_idx = max(0, start_idx); end_idx = min(num_hires_points_trial, end_idx)
                if start_idx < end_idx: lick_trace_trial[start_idx:end_idx] = 1
        idx_analysis_start_in_trial_trace = int(np.round(analysis_start_rel_trial / TIME_RESOLUTION))
        idx_analysis_end_in_trial_trace = idx_analysis_start_in_trial_trace + num_hires_points_total_analysis
        actual_start_idx = max(0, idx_analysis_start_in_trial_trace)
        actual_end_idx = min(len(lick_trace_trial), idx_analysis_end_in_trial_trace)
        extracted_trace = lick_trace_trial[actual_start_idx:actual_end_idx]
        pad_before = max(0, -idx_analysis_start_in_trial_trace)
        pad_after = max(0, num_hires_points_total_analysis - (len(extracted_trace) + pad_before))
        analysis_trace_hires = np.pad(extracted_trace, (pad_before, pad_after), 'constant', constant_values=0)
        if len(analysis_trace_hires) != num_hires_points_total_analysis: return np.full(num_target_time_points, np.nan)
        rolling_proportions_series = pd.Series(analysis_trace_hires)
        rolling_proportions = rolling_proportions_series.rolling(window=win_samples, min_periods=1, center=True).mean().to_numpy()
        valid_indices_mask = (target_indices_in_hires >= 0) & (target_indices_in_hires < len(rolling_proportions))
        valid_target_indices_for_sampling = target_indices_in_hires[valid_indices_mask]
        sampled_proportions = np.full(num_target_time_points, np.nan)
        if np.any(valid_indices_mask): sampled_proportions[valid_indices_mask] = rolling_proportions[valid_target_indices_for_sampling]
        return sampled_proportions

    # --- Data Aggregation (Simplified for one phase) ---
    def aggregate_ts_data(data_dict):
        # This function is also identical
        aggregated_results = {}
        for phase, phase_data in data_dict.items():
            aggregated_results[phase] = {}
            for reward_cat, trial_list in phase_data.items():
                valid_trials_ts = [ts for ts in trial_list if ts is not None and not np.all(np.isnan(ts)) and len(ts) == num_target_time_points]
                n_trials = len(valid_trials_ts)
                mean_ts, sem_ts = None, None
                if n_trials > 0:
                    trial_ts_stack = np.array(valid_trials_ts)
                    mean_ts = np.nanmean(trial_ts_stack, axis=0)
                    if n_trials > 1:
                        with np.errstate(invalid='ignore'): sem_ts = sem(trial_ts_stack, axis=0, nan_policy='omit')
                        sem_ts = np.nan_to_num(sem_ts, nan=0.0)
                    else: sem_ts = np.zeros_like(mean_ts)
                aggregated_results[phase][reward_cat] = {'mean': mean_ts, 'sem': sem_ts, 'n': n_trials}
        return aggregated_results
    
    print(f"  Pooling and aggregating time-resolved data for phase '{target_phase}'...")
    start_agg_time = time.time()
    
    # Structure: pooled_data[phase][reward_category] = [trace1, trace2, ...]
    pooled_data = {target_phase: {r_cat: [] for r_cat in REWARD_CATEGORY_STRINGS}}

    # Loop through sessions and trials to pool data
    for session in test_sessions:
        session_phase = session.get('phase')
        if session_phase != target_phase:
            continue
        
        trial_results = session.get('trial_results', [])
        if not trial_results: continue
        
        for trial_dict in trial_results:
            if not isinstance(trial_dict, dict) or not trial_dict.get('rewarded', False): continue
            reward_latency = trial_dict.get('reward_latency')
            numerical_reward = trial_dict.get('reward_size')
            reward_category_str = NUMERICAL_SIZE_TO_CATEGORY_STRING_MAP.get(numerical_reward)

            if not reward_category_str or reward_latency is None: continue
                
            try:
                sampled_proportions = calculate_single_trial_ts(trial_dict, reward_latency)
                if sampled_proportions is not None and not np.all(np.isnan(sampled_proportions)):
                    pooled_data[session_phase][reward_category_str].append(sampled_proportions)
            except Exception: pass
    
    # Aggregate the pooled data
    averaged_lick_traces = aggregate_ts_data(pooled_data)
    print(f"  Aggregation finished in {time.time() - start_agg_time:.2f} seconds.")

    # --- Plotting (Single Panel) ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Use prefix for a more descriptive title
    full_title_prefix = f"{animal_id} - {plot_title_prefix}" if plot_title_prefix else f"{animal_id}"
    fig.suptitle(f"{full_title_prefix}\nTime-Resolved Licking by Reward Size", fontsize=16, y=0.98)

    # Colormap and normalization for reward sizes
    reward_size_cmap = plt.cm.viridis
    num_sizes_for_norm = [k for k,v in NUMERICAL_SIZE_TO_CATEGORY_STRING_MAP.items() if v in REWARD_CATEGORY_STRINGS]
    if num_sizes_for_norm:
        reward_size_norm = mcolors.Normalize(vmin=min(num_sizes_for_norm), vmax=max(num_sizes_for_norm))
    else:
        reward_size_norm = mcolors.Normalize(vmin=1, vmax=5)

    plot_generated = False
    
    ax.set_title(f"Phase: {target_phase}")
    ax.set_xlabel("Time Relative to Reward (s)")
    ax.set_ylabel("Mean Lick Probability")
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.axvline(0, color='k', linestyle='--', lw=1.5, alpha=0.8, label="Reward (t=0)")
    ax.set_ylim(bottom=-0.05)
    has_data_in_subplot = False

    phase_data = averaged_lick_traces.get(target_phase, {})
    if phase_data:
        for reward_cat in REWARD_CATEGORY_STRINGS:
            data = phase_data.get(reward_cat)
            if data and data.get('n', 0) > 0 and data.get('mean') is not None and not np.all(np.isnan(data['mean'])):
                plot_generated = True
                has_data_in_subplot = True
                
                mean_ts = data['mean']
                sem_ts = data['sem']
                n_trials = data['n']
                
                numerical_size = CATEGORY_STRING_TO_NUMERICAL_SIZE_MAP.get(reward_cat)
                plot_color = reward_size_cmap(reward_size_norm(numerical_size)) if numerical_size is not None else 'grey'
                
                ax.plot(time_points_relative_to_reward, mean_ts, label=f"{reward_cat} (n={n_trials})", color=plot_color, linewidth=2)
                
                if n_trials > 1 and sem_ts is not None and not np.all(np.isnan(sem_ts)):
                    ax.fill_between(time_points_relative_to_reward, mean_ts - sem_ts, mean_ts + sem_ts,
                                    color=plot_color, alpha=0.2)
    
    if has_data_in_subplot:
        ax.legend(title="Reward Size", loc='upper right')
    else:
        ax.text(0.5, 0.5, 'No data available for this phase', ha='center', va='center', transform=ax.transAxes, color='gray')

    ax.set_xlim(time_points_relative_to_reward[0], time_points_relative_to_reward[-1])
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # --- Save Figure ---
    if not plot_generated:
        print("  No data plotted. Skipping save.")
        plt.close(fig)
        return None

    filename_suffix = plot_title_prefix.replace(" ", "_") if plot_title_prefix else target_phase.replace(" ", "_")
    output_filename = os.path.join(output_dir, f"{animal_id}_time_resolved_licking_{filename_suffix}.png")
    
    try:
        plt.savefig(output_filename, dpi=150)
        print(f"  Saved single-phase time-resolved licking figure: {output_filename}")
    except Exception as e:
        print(f"  ERROR saving single-phase time-resolved licking figure: {e}")
        output_filename = None
    finally:
        plt.close(fig)
        
    print(f"  Finished single-phase plot generation ({time.time() - start_time:.2f} seconds).")
    return output_filename

# --- End of File ---