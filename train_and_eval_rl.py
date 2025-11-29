# train_and_eval_rl.py
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from member2.dc_motor_env import DCMotorEnv   # relative import; ensure folder name 'member2'
import torch

# Directoriesp

MODELS_DIR = "models"
LOG_DIR = "logs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def make_env(randomize=True):
    env = DCMotorEnv(dt=0.001, episode_length=1.5, randomize=randomize)
    env = Monitor(env)
    return env

def train(total_timesteps=50000, use_cuda=False):
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    vec_env = DummyVecEnv([lambda: make_env(randomize=True)])
    model = SAC("MlpPolicy", vec_env,
                verbose=1,
                learning_rate=3e-4,
                batch_size=128 if device=="cuda" else 64,
                buffer_size=100000,
                tau=0.01,
                gamma=0.99,
                ent_coef='auto',
                device=device)

    checkpoint_cb = CheckpointCallback(save_freq=5000, save_path=MODELS_DIR, name_prefix='sac_dc')
    eval_env = DummyVecEnv([lambda: make_env(randomize=False)])
    eval_cb = EvalCallback(eval_env, best_model_save_path=os.path.join(MODELS_DIR, "best"),
                           log_path=LOG_DIR, eval_freq=5000, deterministic=True, render=False)

    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, eval_cb])
    model.save(os.path.join(MODELS_DIR, "sac_dc_final"))
    print("Training finished and model saved.")
    return model

def evaluate_and_plot(model_path=None, episodes=3):
    env = DCMotorEnv(dt=0.001, episode_length=1.5, randomize=False)
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "sac_dc_final.zip")
    model = SAC.load(model_path, env=env)

    # Evaluate episodes and capture omega traces
    omega_rl = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        arr = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            arr.append(env.omega_meas)  # measured speed in env
        omega_rl.append(np.array(arr))

    # Get baseline from Member-2's baseline function (we'll call the baseline helper in env)
    baseline_t, baseline_y = env.simulate_pid_baseline()  # env includes helper from member2

    # Plot comparison (use first RL episode)
    t_rl = np.arange(len(omega_rl[0])) * env.dt
    t_base = baseline_t
    plt.figure(figsize=(8,4))
    plt.plot(t_base, baseline_y, label='Baseline PID (Member-2)', linewidth=2)
    plt.plot(t_rl, omega_rl[0], label='RL-tuned PID (SAC)', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Measured speed (rad/s)')
    plt.title('Baseline PID vs SAC-tuned PID')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pid_vs_rl_comparison.png')
    print('Saved pid_vs_rl_comparison.png')
    plt.show()

if __name__ == "__main__":
    # Train then evaluate
    # Set use_cuda=True if you have GPU and want to accelerate training
    model = train(total_timesteps=50000, use_cuda=False)
    evaluate_and_plot()
