

#  Member 1 (DC Motor Model + RL Training & Evaluation)

# Member 1 – Contribution Documentation

This file documents all contributions made by **Member-1** in the AI-Based PID Controller with Reinforcement Learning project.

#  Research Papers Studied

As part of the theoretical foundation, Member-1 studied:

## 1. “The Design of the PID Controller” – Robert A. Paz (2001)

Klipsch School of Electrical and Computer Engineering

-Helped understand classical PID control
-Transfer functions, stability, tuning, error response
-Essential for comparing classical PID with RL-based control

## 2. “Introduction to Reinforcement Learning” – Majid Ghasemi & Dariush Ebrahimi

Department of Computer Science, Wilfrid Laurier University, Canada

- Used to understand RL agent behavior, reward shaping, and policy learning
- Provided theoretical grounding for SAC-based controller design


# Responsibilities of Member-1

Member-1 completed **two major components**:

# 1. Full DC Motor Model in MATLAB (with noise + load + RK4 integration)

File: "dc_motor_model.m"

## Implemented:

- Complete armature + mechanical system dynamics
- Parameters: R, L, Ke, Kt, J, b
- Step voltage input
- Load disturbance at 0.8s
- Sensor noise added to speed
- RK4 numerical integration
- Data logging to CSV
- Automatic plot generation

## Output produced:

-motor_model.csv
-motor_model_plot.png

# 2. RL Training & Evaluation Pipeline (Python + PyTorch + Stable-Baselines3 SAC)

File: "train_and_eval_rl.py"

Member-1 wrote the complete **training → evaluation → plotting** pipeline.

## Implemented:

## Environment creation

python
- "env = DCMotorEnv(dt=0.001, episode_length=1.5, randomize=True)"

This uses Member-2’s custom environment.

## SAC agent setup

* Learning rate
* Tau
* Gamma
* Entropy coefficient
* Automatic device selection (CPU/GPU)

## Callbacks

* Checkpoint saving
* Evaluation every 5000 steps
* Auto-save best model

## Training script

Trains for 50,000 steps and saves:
- "models/sac_dc_final.zip
   models/best/ "


## Evaluation function

* Runs trained agent for 3 episodes
* Collects omega (speed) data
* Calls Member-2’s `simulate_pid_baseline()`
* Generates comparison plot:
  
"pid_vs_rl_comparison.png"

## Output produced:

* Trained SAC model
* Log files
* PID vs RL performance plot

# Files Fully Developed by Member-1

-dc_motor_model.m
-train_and_eval_rl.py
-motor_model.csv
-motor_model_plot.png
-pid_vs_rl_comparison.png

# Skills Demonstrated

* MATLAB system modeling
* RK4 ODE numerical simulation
* Python RL programming
* Stable-Baselines3 (SAC)
* Device-aware training (CPU/GPU)
* Experiment plotting & evaluation
* Integration with Member-2’s environment

# Summary

Member-1 successfully developed both the **motor simulation model** (MATLAB) and the **RL training pipeline** (Python).
This contribution forms the core of how the RL agent interacts, learns, and is evaluated against classical PID control—ensuring the system is scientifically valid and fully testable.



