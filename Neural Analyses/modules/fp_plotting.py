import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

def plot_raw_signals(gcamp_raw_tsd, isosbestic_raw_tsd, output_dir):
    """
    Plots the raw (demultiplexed) GCaMP and Isosbestic signals.
    """
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    axs[0].plot(gcamp_raw_tsd.as_series(), label='Raw GCaMP (Demuxed)', color='green')
    axs[0].set_ylabel('Fluorescence (Raw GCaMP)')
    axs[0].legend(loc='upper right')
    axs[0].set_title('Raw Demultiplexed Fiber Photometry Signals')

    axs[1].plot(isosbestic_raw_tsd.as_series(), label='Raw Isosbestic (Demuxed)', color='purple')
    axs[1].set_xlabel('Time (s) since FP recording start')
    axs[1].set_ylabel('Fluorescence (Raw Isosbestic)')
    axs[1].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'raw_signals.png'))
    plt.close(fig)

def plot_photobleaching_correction(raw_tsd, bleach_fit_tsd, detrended_tsd, signal_name, 
                                 raw_color, fit_color, detrended_color, output_dir):
    """
    Plots the photobleaching correction process for a single signal.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(raw_tsd.as_series(), label=f'Raw {signal_name}', color=raw_color, alpha=0.7)
    ax.plot(bleach_fit_tsd.as_series(), label=f'{signal_name} Bleach Fit', color=fit_color, linestyle='--')
    ax.plot(detrended_tsd.as_series(), label=f'Detrended {signal_name}', color=detrended_color, linewidth=1.5)
    ax.set_ylabel(f'Fluorescence ({signal_name})')
    ax.legend(loc='upper right')
    ax.set_title(f'Photobleaching Correction ({signal_name})')
    ax.set_xlabel('Time (s) since FP recording start')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'photobleaching_correction_{signal_name.lower()}.png'))
    plt.close(fig)

def plot_motion_correction(gcamp_detrended_tsd, isosbestic_detrended_tsd, gcamp_motion_corrected_tsd, output_dir):
    """
    Plots the result of the motion correction step.
    """
    fig = plt.figure(figsize=(12, 6))
    plt.plot(gcamp_detrended_tsd.as_series(), label='Detrended GCaMP (Input)', color='green', alpha=0.5)
    plt.plot(isosbestic_detrended_tsd.as_series(), label='Detrended Isosbestic (Control)', color='purple', alpha=0.5)
    plt.plot(gcamp_motion_corrected_tsd.as_series(), label='Motion Corrected GCaMP', color='blue', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Fluorescence (A.U.)')
    plt.title('Motion Correction Result')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'motion_correction.png'))
    plt.close(fig)

def plot_dff_and_events(dff_tsd, cue_onsets_ts, reward_onsets_ts, output_dir):
    """
    Plots the full dF/F trace with vertical lines for cue and reward events.
    """
    fig = plt.figure(figsize=(15, 7))
    plt.plot(dff_tsd.as_series(), label='dF/F (Processed)', color='blue', zorder=1)
    
    first_cue_plot = True
    for t_cue in cue_onsets_ts.t:
        plt.axvline(x=t_cue, color='cyan', linestyle='--', linewidth=1.2, alpha=0.9, 
                    label='Cue Onset' if first_cue_plot else "", zorder=2)
        first_cue_plot = False
        
    first_reward_plot = True
    for t_reward in reward_onsets_ts.t:
        plt.axvline(x=t_reward, color='magenta', linestyle=':', linewidth=1.2, alpha=0.9, 
                    label='Reward Onset' if first_reward_plot else "", zorder=2)
        first_reward_plot = False
            
    plt.xlabel('Time (s) since FP recording start')
    plt.ylabel('dF/F')
    plt.title('Processed dF/F Signal with Behavioral Events')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.xlim(dff_tsd.t[0], dff_tsd.t[-1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dff_with_events.png'))
    plt.close(fig)

def plot_event_triggered_average(epochs_array, avg_dff, sem_dff, window_before_sec, window_after_sec, event_name, output_dir):
    """
    Plots the event-triggered average with SEM shading.
    """
    fig = plt.figure(figsize=(10, 6))
    time_vector = np.linspace(-window_before_sec, window_after_sec, num=len(avg_dff))
    
    plt.plot(time_vector, avg_dff, label=f'Mean dF/F to {event_name}', color='dodgerblue')
    plt.fill_between(time_vector, avg_dff - sem_dff, avg_dff + sem_dff, 
                     color='dodgerblue', alpha=0.2, label='SEM')
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label=f'{event_name} Onset')
    plt.xlabel(f'Time Relative to {event_name} Onset (s)')
    plt.ylabel('Average dF/F')
    plt.title(f'Avg dF/F to {event_name} ({epochs_array.shape[0]} trials)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'eta_{event_name.lower()}.png'))
    plt.close(fig)

def plot_cue_triggered_epochs_overlay(cue_epochs_array, is_rewarded_cue_np, cue_avg_dff, 
                                      window_before_cue_sec, window_after_cue_sec, output_dir):
    """
    Plots all individual cue-triggered epochs on a single overlay plot, colored by reward status.
    """
    num_epochs, num_samples = cue_epochs_array.shape
    time_vector = np.linspace(-window_before_cue_sec, window_after_cue_sec, num=num_samples)
    
    rewarded_indices = np.where(is_rewarded_cue_np == True)[0]
    unrewarded_indices = np.where(is_rewarded_cue_np == False)[0]
    
    fig = plt.figure(figsize=(12, 7))
    
    for i in unrewarded_indices:
        plt.plot(time_vector, cue_epochs_array[i, :], color='gray', alpha=0.3, linewidth=0.7)
    for i in rewarded_indices:
        plt.plot(time_vector, cue_epochs_array[i, :], color='lightcoral', alpha=0.5, linewidth=0.7)

    if len(unrewarded_indices) > 0:
        plt.plot([], [], color='gray', alpha=0.7, label=f'Unrewarded Cues ({len(unrewarded_indices)})')
    if len(rewarded_indices) > 0:
        plt.plot([], [], color='lightcoral', alpha=0.7, label=f'Rewarded Cues ({len(rewarded_indices)})')

    if cue_avg_dff is not None:
        plt.plot(time_vector, cue_avg_dff, color='dodgerblue', linewidth=2, label='Mean dF/F (All Cues)')
    
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Cue Onset')
    plt.xlabel('Time Relative to Cue Onset (s)')
    plt.ylabel('dF/F')
    plt.title(f'Individual Cue-Triggered dF/F Epochs ({num_epochs} trials)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cue_epochs_overlay.png'))
    plt.close(fig)

def plot_cue_triggered_heatmap(cue_epochs_array, is_rewarded_cue_np, 
                               window_before_cue_sec, window_after_cue_sec, output_dir):
    """
    Plots a heatmap of cue-triggered dF/F epochs with markers for rewarded trials.
    """
    num_epochs, num_samples = cue_epochs_array.shape
    time_vector = np.linspace(-window_before_cue_sec, window_after_cue_sec, num=num_samples)

    fig = plt.figure(figsize=(10, 8))
    
    vmin = np.percentile(cue_epochs_array, 1)
    vmax = np.percentile(cue_epochs_array, 99)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    im = plt.imshow(cue_epochs_array, aspect='auto', cmap='viridis', 
                    extent=[time_vector[0], time_vector[-1], num_epochs - 0.5, -0.5],
                    norm=norm, interpolation='nearest')
    
    plt.colorbar(im, label='dF/F')
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Cue Onset')
    
    rewarded_trial_indices = np.where(is_rewarded_cue_np == True)[0]
    if len(rewarded_trial_indices) > 0:
        star_x_position = time_vector[0]
        plt.scatter([star_x_position] * len(rewarded_trial_indices), rewarded_trial_indices,
                    marker='*', color='yellow', s=100, edgecolor='yellow',
                    label='Rewarded Trial', zorder=5)

    plt.xlabel('Time Relative to Cue Onset (s)')
    plt.ylabel('Trial Number (Original Order)')
    plt.title(f'Heatmap of Cue-Triggered dF/F ({num_epochs} trials)')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = {l: h for h, l in zip(handles, labels)}
    if unique_labels:
        plt.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', labelcolor='white')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cue_epochs_heatmap.png'))
    plt.close(fig)

def plot_reward_triggered_heatmap(reward_epochs_array, dff_tsd, reward_onsets_ts, cue_onsets_ts,
                                  window_before_reward_sec, window_after_reward_sec, output_dir):
    """
    Plots a heatmap of reward-triggered epochs with markers for preceding sound onsets.
    """
    num_epochs, num_samples = reward_epochs_array.shape
    time_vector = np.linspace(-window_before_reward_sec, window_after_reward_sec, num=num_samples)
    
    fs = dff_tsd.rate
    samples_before_reward = int(window_before_reward_sec * fs)
    samples_after_reward = int(window_after_reward_sec * fs)
    total_epoch_samples_reward = samples_before_reward + samples_after_reward
    
    actual_reward_times = []
    for r_time in reward_onsets_ts.t:
        event_idx = np.argmin(np.abs(dff_tsd.t - r_time))
        start_idx, end_idx = event_idx - samples_before_reward, event_idx + samples_after_reward
        if start_idx >= 0 and end_idx <= len(dff_tsd.d) and (end_idx - start_idx) == total_epoch_samples_reward:
            actual_reward_times.append(r_time)

    sound_onset_dots = []
    for epoch_idx, abs_reward_time in enumerate(actual_reward_times):
        preceding_cues = cue_onsets_ts.t[cue_onsets_ts.t <= abs_reward_time]
        if preceding_cues.size > 0:
            abs_cue_time = preceding_cues[-1]
            if abs_reward_time - abs_cue_time <= 15.0:
                relative_cue_time = abs_cue_time - abs_reward_time
                if time_vector[0] <= relative_cue_time <= time_vector[-1]:
                    sound_onset_dots.append({'x': relative_cue_time, 'y': epoch_idx})
    
    fig = plt.figure(figsize=(10, 8))
    vmin = np.percentile(reward_epochs_array, 1)
    vmax = np.percentile(reward_epochs_array, 99)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    im = plt.imshow(reward_epochs_array, aspect='auto', cmap='cividis',
                    extent=[time_vector[0], time_vector[-1], num_epochs - 0.5, -0.5],
                    norm=norm, interpolation='nearest')
    
    plt.colorbar(im, label='dF/F')
    plt.axvline(0, color='magenta', linestyle='--', linewidth=1.5, label='Reward Onset')
    
    if sound_onset_dots:
        x_coords = [d['x'] for d in sound_onset_dots]
        y_coords = [d['y'] for d in sound_onset_dots]
        plt.scatter(x_coords, y_coords, marker='o', color='red', s=30, edgecolor='white', 
                    linewidth=0.5, label='Sound Onset', zorder=5)

    plt.xlabel('Time Relative to Reward Onset (s)')
    plt.ylabel('Trial Number (Original Order)')
    plt.title(f'Heatmap of Reward-Triggered dF/F ({num_epochs} trials)')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = {l: h for h, l in zip(handles, labels)}
    if unique_labels:
        plt.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', labelcolor='white', frameon=False)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_epochs_heatmap.png'))
    plt.close(fig) 