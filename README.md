<h1 align="center"> DC Motor PID Optimization using Reinforcement Learning</h1>
<p align="center">
  <b>Classical PID meets Deep Reinforcement Learning</b>
  <br>
  <i>Stable Control, Faster Response, Smarter Tuning</i>
   <b>By Kaushiki Singh and Kaukab Erum </b>
</p>


#  Member 1 (DC Motor Model + RL Training & Evaluation)

# Member 1 – Contribution Documentation
This file documents all contributions made by **Member-1** in the AI-Based PID Controller with Reinforcement Learning project.

#  Research Papers Studied
As part of the theoretical foundation, Member-1 studied:

## 1. “The Design of the PID Controller” – Robert A. Paz (2001)
Klipsch School of Electrical and Computer Engineering
-Helped understand classical PID control
## 2. “Introduction to Reinforcement Learning” – Majid Ghasemi & Dariush Ebrahimi
Department of Computer Science, Wilfrid Laurier University, Canada
- Used to understand RL agent behavior, reward shaping, and policy learning

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

# Files Developed by Member-1
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
-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# **Member 2 Contribution – RL Implementation & Training**
This project focuses on optimizing PID controller parameters (Kp, Ki, Kd) for a DC motor using Reinforcement Learning instead of manual trial-and-error tuning.


# Objective
To train an RL agent that automatically learns optimal PID gains by interacting with a DC motor simulation environment and minimizing control error.

# RL Method Used
We use **Deep Deterministic Policy Gradient (DDPG)** — a continuous action-space RL algorithm suitable for PID gain tuning.

The trained agent:
- Observes motor behavior
- Adjusts PID parameters dynamically
- Minimizes steady-state error & overshoot
- Improves system response over episodes

# Project Structure (Member 2)
```
member2/
│
├── dc_motor_env.py              # Custom Gym environment
├── train_and_eval_rl.py         # RL training & evaluation script
└── README.md                    # Documentation (this file)
```

# Input & Output Files (Member 2 Scope)

| Type | File | Description |
|------|------|-------------|
| **Input** | `dc_motor_pid_baseline.m` | Baseline MATLAB control system to compare with RL |
| **Input** | `dc_motor_env.py` | Simulation model used for RL environment |
| **Output** | `baseline_pid_data.csv` | Error & response data collected for baseline PID |
| **Output** | `baseline_pid_response.png` | Graph showing baseline PID system performance |

# Research Papers Studied
**The Design of the PID Controller – Robert A. Paz (2001)**  Helped understand classical PID tuning mathematics and the effect of each gain on stability. 
**Introduction to Reinforcement Learning – Majid Ghasemi & Dariush Ebrahimi** Helped understand reward functions, agent-environment interaction, and policy learning for control tasks. 



# Member 2 Responsibilities
✔ Designed custom DC motor Gym environment  
✔ Developed RL training pipeline using DDPG  
✔ Integrated reward structure to reduce control error  
✔ Compared RL-tuned vs baseline PID performance  
✔ Studied research papers for strong theoretical base  

# Skills Gained
- Reinforcement Learning (DDPG Algorithm)
- Control Systems & PID Optimization
- Python, Gym API, Numerical Simulation Techniques
- Performance Evaluation using Plots & Metrics
- Reading & understanding research papers  
- Version control using GitHub


##  How to Run Training
Activate virtual environment:
```bash
.\rlpid\Scripts\activate
```

Train the RL agent:

```bash
python train_and_eval_rl.py
```


# What You Will See After Running
- RL agent tuning Kp, Ki, Kd automatically
- Response plots improving over episodes
- Logged PID gains + performance comparison


###  Outcome
RL successfully learns to reduce error and improves dynamic response compared to manually-tuned PID.

 

 **Note:** We are still working on this project. Sorry for any mistakes or incomplete parts.
## by Kaushiki Singh and Kaukab Erum
