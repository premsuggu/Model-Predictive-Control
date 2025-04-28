import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import csv

# Load data from CSV
def load_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return np.array([[float(val) for val in row] for row in reader])

states = load_csv("state_trajectory.csv")
num_steps, nx = states.shape

# Reference trajectory (constant for now)
x_ref = np.tile(np.array([1.5, 0.0, 0.0, 0.0]), (num_steps, 1))

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Custom colors for each state
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# Plot objects for states and references
lines = [ax.plot([], [], label=f"x{i+1}", color=colors[i], linewidth=2)[0] for i in range(nx)]
ref_lines = [ax.plot([], [], linestyle='--', label=f"x{i+1}_ref", color=colors[i], alpha=0.5)[0] for i in range(nx)]

# Axis limits and labels
ax.set_xlim(0, num_steps)
ax.set_ylim(-2, 3)
ax.set_xlabel("Time step", fontsize=14)
ax.set_ylabel("State value", fontsize=14)
ax.set_title("NMPC State Tracking with Reference", fontsize=16, fontweight='bold')

# Legend and grid
ax.legend(loc='upper right', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)

# Background color for fun
fig.patch.set_facecolor('#f0f0f0')
ax.set_facecolor('#ffffff')

# Annotation for current time step
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

def init():
    for line, ref in zip(lines, ref_lines):
        line.set_data([], [])
        ref.set_data([], [])
    time_text.set_text('')
    return lines + ref_lines + [time_text]

def update(frame):
    x_vals = np.arange(frame + 1)
    for i in range(nx):
        lines[i].set_data(x_vals, states[:frame+1, i])
        ref_lines[i].set_data(x_vals, x_ref[:frame+1, i])
    time_text.set_text(f'Time step: {frame}')
    return lines + ref_lines + [time_text]

ani = animation.FuncAnimation(
    fig, update, frames=num_steps, init_func=init,
    blit=True, interval=100, repeat=False
)

plt.show()
