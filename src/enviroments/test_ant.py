#step 1: load the env
import gymnasium as gym
import mujoco
import mujoco.viewer
#step 2: simulate the robot
env= gym.make("Ant-v5")
observation, info= env.reset(seed=42)
model= env.unwrapped.model
data= env.unwrapped.data
with mujoco.viewer.launch_passive(model, data) as viewer: 
    for step in range(500):
        action= env.action_space.sample()
        observation, reward, terminated, truncated, info= env.step(action)
        viewer.sync()
        if terminated or truncated:
            observation, info= env.reset()
            #finally close the environment
            env.close()
            print("Ant GUI simulation completed")