import matplotlib.pyplot as plt
import numpy as np
import re

# Helper function to extract and parse calibration data
def extract_calibration_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    
    # Split the data into CCW and CW sections
    sections = re.split('Measuring CW...|Measuring CCW...', data)
    ccw_data = sections[1] if len(sections) > 1 else ""
    cw_data = sections[2] if len(sections) > 2 else ""

    # Function to parse individual sections
    def parse_section(section):
        duty_cycles, right_speeds, left_speeds = [], [], []
        for line in section.split('\n'):
            match = re.match(r'duty: ([\-\d\.]+), right: ([\-\d\.]+), left: ([\-\d\.]+)', line)
            if match:
                duty_cycles.append(float(match.group(1)))
                right_speeds.append(0.08375*float(match.group(2)))
                left_speeds.append(0.08375*float(match.group(3)))
        return np.array(duty_cycles), np.array(right_speeds), np.array(left_speeds)

    # Parse CCW and CW sections
    duty_cycles_ccw, right_speeds_ccw, left_speeds_ccw = parse_section(ccw_data)
    duty_cycles_cw, right_speeds_cw, left_speeds_cw = parse_section(cw_data)

    return (duty_cycles_ccw, right_speeds_ccw, left_speeds_ccw), (duty_cycles_cw, right_speeds_cw, left_speeds_cw)

# Extract calibration data
file_path = 'calibration1.txt'
(ccw_duty_cycles, ccw_right_speeds, ccw_left_speeds), (cw_duty_cycles, cw_right_speeds, cw_left_speeds) = extract_calibration_data(file_path)

# Function to plot calibration data
def plot_calibration_data(duty_cycles, right_speeds, left_speeds, movement_type):
    plt.figure(figsize=(12, 10))
    for i, (speeds, title, color) in enumerate(zip([right_speeds, left_speeds], ['Right Speed', 'Left Speed'], ['teal', 'purple'])):
        for j, (dc, speeds, label) in enumerate(zip([duty_cycles, speeds], [speeds, duty_cycles], [title + ' vs. Duty Cycle', 'Duty Cycle vs. ' + title])):
            plt.subplot(2, 2, i*2 + j + 1)
            plt.plot(dc, speeds, 'o', color=color)
            fit = np.polyfit(dc, speeds, 1)
            fit_fn = np.poly1d(fit)
            plt.plot(dc, fit_fn(dc), '--k', label=f'Fit: y={fit[0]:.2f}x+{fit[1]:.2f}')
            plt.title(f'{label} ({movement_type})')
            plt.xlabel('Duty Cycle' if j == 0 else title)
            plt.ylabel(title if j == 0 else 'Duty Cycle')
            plt.legend()
    plt.tight_layout()
    plt.show()

# Plot CCW and CW calibration data
plot_calibration_data(*extract_calibration_data(file_path)[0], 'CCW')
plot_calibration_data(*extract_calibration_data(file_path)[1], 'CW')
