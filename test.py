import matplotlib.pyplot as plt
import numpy as np


def analyze_and_plot_blocks(raw_data, num_workers, num_days, num_shifts, plt_show = False):
    """
    Analyzes a flat list of shift assignments to calculate lengths of consistent shift blocks
    and plots a histogram.

    Args:
    raw_data (list): Flattened list of binary shift assignments.
    num_workers (int): Number of workers.
    num_days (int): Number of days.
    num_shifts (int): Number of shift types per day.

    Returns:
    list: A list containing the lengths of all identified consistent shift blocks.
    """

    # 1. Reshape and Decode Data
    worker_schedules = []
    current_index = 0

    for w in range(num_workers):
        daily_shifts = []
        for d in range(num_days):
            # Extract the triplet (or tuple of size num_shifts) for this day
            # Data is expected to be: [w1_d1_s1, w1_d1_s2, ..., w1_d2_s1, ...]
            if current_index + num_shifts > len(raw_data):
                break  # Safety break

            day_data = raw_data[current_index: current_index + num_shifts]
            current_index += num_shifts

            # Determine shift ID (0 = Free, 1..N = Shift Type)
            shift_id = 0
            for i, val in enumerate(day_data):
                if val == 1.0:
                    shift_id = i + 1
                    break
            daily_shifts.append(shift_id)
        worker_schedules.append(daily_shifts)

    # 2. Calculate Block Lengths
    all_block_lengths = []

    for schedule in worker_schedules:
        current_len = 0
        last_shift = -1

        for shift in schedule:
            if shift == 0:  # Free/Off day breaks the consistency block
                if current_len > 0:
                    all_block_lengths.append(current_len)
                current_len = 0
                last_shift = -1
                continue

            if shift == last_shift:
                # Continue existing block
                current_len += 1
            else:
                # Shift type changed (but not to free): New block starts
                if current_len > 0:
                    all_block_lengths.append(current_len)
                current_len = 1
                last_shift = shift

        # Capture block if it ends at the very end of the schedule
        if current_len > 0:
            all_block_lengths.append(current_len)

    # 3. Plot Histogram
    if all_block_lengths:
        plt.figure(figsize=(8, 5))
        # Create bins centered on integers: 0.5, 1.5, 2.5, ...
        max_val = max(all_block_lengths)
        bins = np.arange(0.5, max_val + 1.5, 1)

        plt.hist(all_block_lengths, bins=bins, color='#4c72b0', edgecolor='white', rwidth=0.8)
        plt.title("Distribution of Block Lengths (Consistency Analysis)")
        plt.xlabel("Length of Consistent Block (Days)")
        plt.ylabel("Frequency")
        plt.xticks(range(1, max_val + 1))
        plt.grid(axis='y', alpha=0.5, linestyle='--')
        plt.savefig('block_histogram.png')
        if plt_show == True:
            plt.show()
    else:
        print("No consistent work blocks found.")

    return all_block_lengths


# --- Test Run with User Data ---
raw_data = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]

num_workers = 5
num_days = 4
num_shifts = 3

print("Blocklängen:", analyze_and_plot_blocks(raw_data, num_workers, num_days, num_shifts))