# member2/dc_motor_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DCMotorEnv(gym.Env):
    """
    Full DC motor gymnasium environment compatible with gymnasium API.
    Action: [Kp, Ki, Kd] -- applied as PID gains for that episode
    Observation: [error, integral, derivative, omega_meas, omega_ref]
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, dt=0.001, episode_length=1.5, randomize=True):
        super().__init__()
        self.dt = float(dt)
        self.episode_length = float(episode_length)
        self.max_steps = int(self.episode_length / self.dt)
        self.randomize = bool(randomize)

        # Nominal motor parameters (match MATLAB model)
        self.R_nom = 1.0
        self.L_nom = 0.5e-3
        self.Kt_nom = 0.01
        self.Ke_nom = 0.01
        self.J_nom = 1e-4
        self.b_nom = 1e-4

        # Action space: Kp, Ki, Kd
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([200.0, 500.0, 5.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation: error, integral, derivative, omega_meas, omega_ref
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        # internal state placeholders
        self.seed_val = None
        self.reset()

    def _sample_params(self):
        if not self.randomize:
            self.R = self.R_nom; self.L = self.L_nom; self.Kt = self.Kt_nom
            self.Ke = self.Ke_nom; self.J = self.J_nom; self.b = self.b_nom
        else:
            # +/-10% variation
            factor = lambda x: x * (1.0 + 0.1*(np.random.rand()-0.5))
            self.R = factor(self.R_nom)
            self.L = factor(self.L_nom)
            self.Kt = factor(self.Kt_nom)
            self.Ke = factor(self.Ke_nom)
            self.J = factor(self.J_nom)
            self.b = factor(self.b_nom)

    def reset(self, *, seed=None, options=None):
        # gymnasium reset signature: return obs, info
        if seed is not None:
            self.seed_val = seed
            np.random.seed(seed)

        self._sample_params()
        self.i = 0.0
        self.omega = 0.0
        self.theta = 0.0
        self.step_count = 0
        self.integral = 0.0
        self.prev_error = 0.0

        # reference speed (rad/s) random step for robustness
        self.omega_ref = float(np.random.uniform(50.0, 120.0))

        # load schedule: step at 0.8s
        self.t_elapsed = 0.0
        self.tau_load = 0.0

        # measured speed (with noise)
        self.omega_meas = float(self.omega + 0.5 * np.random.randn())

        obs = np.array([self.omega_ref - self.omega, self.integral, 0.0, self.omega_meas, self.omega_ref], dtype=np.float32)
        info = {"omega_ref": self.omega_ref}
        return obs, info

    def motor_derivs(self, i, w, V, tauL):
        di = (-self.R*i - self.Ke*w + V) / self.L
        dw = (self.Kt*i - self.b*w - tauL) / self.J
        return di, dw

    def _rk4(self, i, w, V, tauL):
        dt = self.dt
        di1, dw1 = self.motor_derivs(i, w, V, tauL)
        di2, dw2 = self.motor_derivs(i + 0.5*dt*di1, w + 0.5*dt*dw1, V, tauL)
        di3, dw3 = self.motor_derivs(i + 0.5*dt*di2, w + 0.5*dt*dw2, V, tauL)
        di4, dw4 = self.motor_derivs(i + dt*di3, w + dt*dw3, V, tauL)
        i_n = i + (dt/6.0)*(di1 + 2*di2 + 2*di3 + di4)
        w_n = w + (dt/6.0)*(dw1 + 2*dw2 + 2*dw3 + dw4)
        return i_n, w_n

    def step(self, action):
        """
        Gymnasium step: returns (obs, reward, terminated, truncated, info)
        We treat episodes by fixed length -> use 'truncated' when max_steps reached.
        """
        # ensure numpy array
        action = np.asarray(action, dtype=np.float32).flatten()
        Kp = float(np.clip(action[0], 0.0, 200.0))
        Ki = float(np.clip(action[1], 0.0, 500.0))
        Kd = float(np.clip(action[2], 0.0, 5.0))

        # update external load after 0.8s
        if self.t_elapsed >= 0.8:
            self.tau_load = 0.001
        else:
            self.tau_load = 0.0

        # compute PID
        error = self.omega_ref - self.omega
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt if self.step_count > 0 else 0.0
        self.prev_error = error

        # anti-windup
        self.integral = float(np.clip(self.integral, -1e6, 1e6))

        V_unsat = Kp * error + Ki * self.integral + Kd * derivative
        V = float(np.clip(V_unsat, -24.0, 24.0))

        # integrate motor states
        i_n, w_n = self._rk4(self.i, self.omega, V, self.tau_load)
        self.i = i_n
        self.omega = w_n
        self.theta += self.omega * self.dt
        self.t_elapsed += self.dt
        self.step_count += 1

        # sensor noise on measurement
        self.omega_meas = float(self.omega + 0.5 * np.random.randn())

        # reward: negative squared error + small control penalty + gain magnitude penalty
        reward = - (error**2) - 1e-4*(V**2) - 1e-6*(abs(Kp) + abs(Ki) + abs(Kd))

        # termination flags
        terminated = False
        truncated = False
        if abs(self.omega) > 3000:
            reward -= 1000.0
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        obs = np.array([error, self.integral, derivative, self.omega_meas, self.omega_ref], dtype=np.float32)
        info = {'V': V, 'Kp': Kp, 'Ki': Ki, 'Kd': Kd, 'omega_true': self.omega}
        return obs, float(reward), terminated, truncated, info

    def render(self):
        print(f"t={self.t_elapsed:.4f}s omega={self.omega:.2f} ref={self.omega_ref:.2f}")

    def simulate_pid_baseline(self, Kp=25.0, Ki=50.0, Kd=0.0005):
        """
        Helper: simulate deterministic baseline PID (no randomization),
        returns (time_array, measured_omega_array)
        """
        saved_randomize = self.randomize
        self.randomize = False
        obs, _ = self.reset()
        i = self.i; w = self.omega
        prev_err = 0.0
        integral = 0.0
        t_list = []
        omega_meas = []
        # set load schedule local
        tau_load = 0.0
        for step in range(self.max_steps):
            error = self.omega_ref - w
            integral += error * self.dt
            derivative = (error - prev_err) / self.dt if step > 0 else 0.0
            prev_err = error
            V = Kp*error + Ki*integral + Kd*derivative
            V = float(np.clip(V, -24.0, 24.0))
            if step*self.dt >= 0.8:
                tau_load = 0.001
            i, w = self._rk4(i, w, V, tau_load)
            # measured with noise (small)
            omega_meas.append(w + 0.5*np.random.randn())
            t_list.append(step*self.dt)
        self.randomize = saved_randomize
        return np.array(t_list), np.array(omega_meas)
