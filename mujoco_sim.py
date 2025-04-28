import mujoco
import mujoco.viewer
import numpy as np
import time

# Load NMPC trajectories
states = np.loadtxt("state_trajectory.csv", delimiter=",")
controls = np.loadtxt("control_trajectory.csv", delimiter=",") 

# Load MuJoCo model
model = mujoco.MjModel.from_xml_path("mujoco_models/pendulum_cart.xml")
data = mujoco.MjData(model)

# Set initial state (completely upright)
data.qpos[0] = states[0, 0]    # Initial cart position
data.qpos[1] = states[0, 2]    # Initial pendulum angle (theta)
data.qvel[0] = states[0, 1]    # Initial cart velocity
data.qvel[1] = states[0, 3]    # Initial pendulum angular velocity

# Viewer launch
with mujoco.viewer.launch_passive(model, data) as viewer:
    timestep = model.opt.timestep

    for t in range(len(states)):
        x = states[t]

        # Update simulation state from trajectory
        data.qpos[0] = x[0]
        data.qpos[1] = x[2]
        data.qvel[0] = x[1]
        data.qvel[1] = x[3]

        mujoco.mj_forward(model, data) # Recompute derived quantities

        viewer.sync()
        time.sleep(timestep)   # Real-time playback at NMPC timestep

print("Simulation completed successfully.")
