
import time
import traceback
import numpy as np
import gymnasium as gym
import mujoco.viewer

# ============================================================
# STEP 1: LOAD GYMNASIUM ANT-v5 AND INITIALISE THE SIMULATION
# ============================================================

env = gym.make("Ant-v5")

observation, info = env.reset(seed=42)

# Access the underlying MuJoCo model and data
model = env.unwrapped.model
data = env.unwrapped.data

print("Ant-v5 initialised successfully")

# ============================================================
# STEP 2: CONFIGURE THE ROBOT AND MOTOR PARAMETERS
# ============================================================

n_motors = env.action_space.shape[0]

# Eight phase offsets, separated by 45 degrees
phase = np.linspace(
    0,
    2 * np.pi,
    n_motors,
    endpoint=False
)

# Actuator control limits
ctrl_low = env.action_space.low
ctrl_high = env.action_space.high

# Initial controller parameters
frequency = 0.5
amplitude = 0.6

# Simulation timing
physics_dt = model.opt.timestep
frame_skip = env.unwrapped.frame_skip
control_dt = physics_dt * frame_skip

cycle = 0.0

print("Number of actuators:", n_motors)
print("Physics timestep:", physics_dt)
print("Control interval:", control_dt)

# ============================================================
# STEP 3: GENERATE SINUSOIDAL MOTOR COMMANDS
# ============================================================

# Each actuator receives a sinusoidal signal
# with a different phase offset.

# ============================================================
# STEP 4: EXECUTE THE MUJOCO PHYSICS SIMULATION
# ============================================================

try:

    # Launch the passive MuJoCo viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:

        print("Viewer launched. Simulation running...")

        # ====================================================
        # STEP 5: CONTINUOUSLY RUN AND DISPLAY THE ROBOT
        # ====================================================

        while viewer.is_running():

            step_start = time.perf_counter()

            # Update the oscillator cycle
            cycle += frequency * control_dt

            # Generate motor commands
            action = amplitude * np.sin(
                2 * np.pi * cycle + phase
            )

            # Apply actuator limits
            action = np.clip(
                action,
                ctrl_low,
                ctrl_high
            )

            # Lock MuJoCo state while Gymnasium advances
            # the simulation to prevent viewer conflicts.
            with viewer.lock():

                observation, reward, terminated, truncated, info = env.step(action)

                # Reset after an episode ends
                if terminated or truncated:

                    observation, info = env.reset()
                    cycle = 0.0

            # Synchronise the graphical viewer
            viewer.sync()

            # Pace simulation to approximately real time
            elapsed = time.perf_counter() - step_start
            sleep_time = control_dt - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

except KeyboardInterrupt:

    print("Simulation stopped by user.")

except Exception:

    print("Simulation error:")
    traceback.print_exc()

finally:

    env.close()
    print("Environment closed.")

