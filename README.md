```
# Bio-Inspired Sensorimotor Coordination in Robotics

Integrating Neuroscience, Robotics, and Intelligent Systems

Overview
This repository contains the development and research for my Master's project, exploring bio-inspired sensorimotor coordination in robotics. The project combines principles from neuroscience, robotics, and machine learning to investigate adaptive and intelligent robotic control.

Technologies & Tools

- **Python** – Primary programming language for implementation, simulation, and data analysis.
- **ROS / ROS 2** – Robot middleware for communication, sensor integration, and control.
- **Robotics Toolbox for Python** – Robot kinematics, dynamics, trajectory planning, and modelling.
- **NumPy & SciPy** – Numerical computation and scientific analysis.
- **Matplotlib** – Visualisation of sensorimotor behaviour and experimental results.
- **Machine Learning** – Learning-based approaches for adaptive sensorimotor coordination.

## Machine Learning Algorithms

Potential algorithms to be explored include:

- **Reinforcement Learning (RL)** – Learning motor control policies through interaction with the environment.
- **Artificial Neural Networks (ANNs)** – Modelling bio-inspired processing and sensorimotor mappings.
- **Deep Reinforcement Learning (DRL)** – Learning complex control behaviours where appropriate.
- **Adaptive Control** – Adjusting control behaviour in response to changing conditions.
- **Bio-Inspired Neural Models** – Exploring biologically motivated mechanisms for sensorimotor coordination.

*The final algorithms will be selected according to the research objectives and experimental requirements.*

## Project Objectives

- Investigate biological principles underlying sensorimotor coordination.
- Develop a bio-inspired robotic control architecture.
- Integrate sensory feedback with motor control.
- Explore machine learning for adaptive robotic behaviour.
- Evaluate performance through simulation and/or physical robotic experiments.

## Repository Structure

- `src/` – Core implementation and control algorithms.
- `simulations/` – Robotic models and simulation environments.
- `experiments/` – Experimental scripts and configurations.
- `notebooks/` – Data analysis and prototyping.
- `docs/` – Research documentation and methodology.

## Project Status
🚧 **In Development** — Literature review, architecture design, and implementation.

## Academic Context
Master's research project investigating the intersection of computational neuroscience, robotics, and intelligent systems.

## Simulation Video Recording

The controller runner can optionally save an evaluation rollout as an MP4.
MuJoCo renders each robot as an RGB image at 1280 x 720 where supported, and
Gymnasium's `RecordVideo` wrapper captures the initial frame and the frames
produced as the controller's actions advance the environment. Closing the
wrapper finalises the video, including rollouts that end at the configured
step limit. These videos provide qualitative evidence of gait and stability
alongside the quantitative reward and motion metrics in the CSV files.

Install the video and MuJoCo dependencies in the active Python environment:

```powershell
pip install "gymnasium[mujoco]" moviepy imageio-ffmpeg
```

Run one recorded episode (from the repository root):

```powershell
python -m src.simulations.run_controller --env Ant-v5 --controller cpg --steps 500 --seed 42 --record-video
python -m src.simulations.run_controller --env Ant-v5 --controller reservoir --steps 500 --seed 42 --record-video
python -m src.simulations.run_controller --env Swimmer-v5 --controller cpg --steps 500 --seed 42 --record-video
python -m src.simulations.run_controller --env Swimmer-v5 --controller reservoir --steps 500 --seed 42 --record-video
```

By default, recordings are written under `results/videos/ant/` and
`results/videos/swimmer/`, with names such as `ant_cpg_seed42.mp4` and
`swimmer_reservoir_seed42.mp4`. Existing videos are preserved; rerunning a
condition gives the new file a `_run02`, `_run03`, and similar suffix.

The experiment runner stays headless unless `--record-video` is supplied. It
records at most the first selected evaluation episode for each environment and
controller by default; adjust `--video-episodes` to record more:

```powershell
python -m src.simulations.run_experiments --episodes 3 --steps 500 --seeds 42 43 44 --record-video --video-episodes 1 --video-dir results/videos
```

With seed 42 and one selected episode, this produces four paired recordings:
Ant with CPG, Ant with the untrained/random ESN baseline, Swimmer with CPG, and
Swimmer with the untrained/random ESN baseline. Use the same seed when comparing
controllers in an environment. The experiment runner continues appending
quantitative episode metrics to `results/csv/`; video recording is opt-in and
does not change the controller implementations.
