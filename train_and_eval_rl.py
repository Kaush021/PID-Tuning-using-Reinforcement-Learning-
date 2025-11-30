# /train_and_eval_rl.py
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# adjust import path to  package
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dc_motor_env import DCMotorEnv

# Directories
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def make_env(randomize=True):
    env = DCMotorEnv(dt=0.001, episode_length=1.5, randomize=randomize)
    env = Monitor(env)  # Monitor works with gymnasium
    return env

def train(total_timesteps=50000, use_cuda=False):
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    vec_env = DummyVecEnv([lambda: make_env(randomize=True)])
    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=128 if device == "cuda" else 64,
        buffer_size=100000,
        tau=0.01,
        gamma=0.99,
        ent_coef='auto',
        device=device,
    )

    checkpoint_cb = CheckpointCallback(save_freq=5000, save_path=MODELS_DIR, name_prefix='sac_dc')
    eval_env = DummyVecEnv([lambda: make_env(randomize=False)])
    eval_cb = EvalCallback(eval_env, best_model_save_path=os.path.join(MODELS_DIR, "best"),
                           log_path=LOG_DIR, eval_freq=5000, deterministic=True, render=False)

    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, eval_cb])
    final_path = os.path.join(MODELS_DIR, "sac_dc_final")
    model.save(final_path)
    print(f"Training finished. Model saved to {final_path}.zip")
    return model

def evaluate_and_plot(model_path=None, episodes=3):
    # create deterministic env for comparison
    env = DCMotorEnv(dt=0.001, episode_length=1.5, randomize=False)
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "sac_dc_final.zip")
    model = SAC.load(model_path, env=env)

    # run RL episodes
    omega_rl = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        arr = []
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            arr.append(env.omega_meas)
            if terminated or truncated:
                break
        omega_rl.append(np.array(arr))

    # get baseline from env helper
    baseline_t, baseline_y = env.simulate_pid_baseline()

    # plot comparison (use first RL episode)
    t_rl = np.arange(len(omega_rl[0])) * env.dt
    t_base = baseline_t
    plt.figure(figsize=(9,4))
    plt.plot(t_base, baseline_y, label='Baseline PID ', linewidth=2)
    plt.plot(t_rl, omega_rl[0], label='RL-tuned PID (SAC)', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Measured speed (rad/s)')
    plt.title('Baseline PID vs SAC-tuned PID')
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(os.path.dirname(__file__), "..", "pid_vs_rl_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved comparison plot to {out_path}")
    plt.show()

if __name__ == "__main__":
    # set use_cuda=True if you have GPU and want acceleration
    model = train(total_timesteps=50000, use_cuda=False)
    evaluate_and_plot()