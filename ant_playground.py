import os
import time
import numpy as np
import gymnasium as gym
import mujoco
import mujoco.viewer

# Load the Ant model that ships with gymnasium
xml_path = os.path.join(os.path.dirname(gym.__file__), "envs", "mujoco", "assets", "ant.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

n_joints = model.nu  # 8 motors
phase = np.linspace(0, 2 * np.pi, n_joints, endpoint=False)
ctrl_low = model.actuator_ctrlrange[:, 0]
ctrl_high = model.actuator_ctrlrange[:, 1]
torso_id = model.body("torso").id

# Shared settings that the keyboard can change
state = {"freq": 1.5, "amp": 0.6, "random": False, "reset": False, "cycle": 0.0}

# GLFW key codes
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265
KEY_M, KEY_R = 77, 82  # M = toggle random, R = reset robot

def key_callback(keycode):
    # The viewer runs this on its own thread, so only change flags here.
    if keycode == KEY_UP:
        state["amp"] = min(state["amp"] + 0.1, 1.0)
    elif keycode == KEY_DOWN:
        state["amp"] = max(state["amp"] - 0.1, 0.0)
    elif keycode == KEY_RIGHT:
        state["freq"] = min(state["freq"] + 0.1, 4.0)
    elif keycode == KEY_LEFT:
        state["freq"] = max(state["freq"] - 0.1, 0.0)
    elif keycode == KEY_M:
        state["random"] = not state["random"]
    elif keycode == KEY_R:
        state["reset"] = True
    print(f"freq={state['freq']:.1f}  amp={state['amp']:.1f}  random={state['random']}")

# Match Ant-v5: one control update every 5 physics steps (0.05 s)
DECIMATION = 5
control_dt = model.opt.timestep * DECIMATION

def reset_robot():
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)  # recompute positions after reset
    state["cycle"] = 0.0

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # Manual reset (R key)
        if state["reset"]:
            with viewer.lock():
                reset_robot()
            state["reset"] = False

        # Auto-reset if the robot falls or flips upside down
        upright = data.xmat[torso_id].reshape(3, 3)[2, 2]
        if data.qpos[2] < 0.2 or upright < 0.0:
            print("Robot fell over, resetting")
            with viewer.lock():
                reset_robot()

        # Advance the gait phase so changing speed doesn't make the legs jump
        state["cycle"] += state["freq"] * control_dt
        if state["random"]:
            ctrl = np.random.uniform(ctrl_low, ctrl_high)
        else:
            ctrl = state["amp"] * np.sin(2 * np.pi * state["cycle"] + phase)

        with viewer.lock():
            data.ctrl[:] = np.clip(ctrl, ctrl_low, ctrl_high)

        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)
        viewer.sync()

        # Pace the simulation to real time
        sleep_time = control_dt - (time.time() - step_start)
        if sleep_time > 0:
            time.sleep(sleep_time)
