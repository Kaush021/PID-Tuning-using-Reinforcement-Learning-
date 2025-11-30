% dc_motor_model.m
% Full DC motor dynamic model (armature circuit + mechanical) with external load & sensor noise
clc; clear; close all;

% ==== Parameters (nominal) ====
R = 1.0;           % Ohm
L = 0.5e-3;        % H
Kt = 0.01;         % N*m/A (torque constant)
Ke = 0.01;         % V/(rad/s) (back-emf)
J = 1e-4;          % kg*m^2 (rotor inertia)
b = 1e-4;          % N*m*s (viscous friction)

% External load torque (step + small disturbance)
tau_load_nom = 0.0;   % nominal constant load
tau_load_step = 0.001; % additional step disturbance magnitude

% Simulation settings
Tfinal = 2.0;      % seconds
dt = 0.0005;       % integration step
t = 0:dt:Tfinal;

% Input: voltage (we will simulate a step of 12V at t=0.02s)
V_in = zeros(size(t));
V_in(t >= 0.02) = 12.0;

% Preallocate
N = length(t);
i = zeros(1,N);
w = zeros(1,N);
theta = zeros(1,N);

% Initial conditions
i(1) = 0.0;
w(1) = 0.0;
theta(1) = 0.0;

% Add a timed load step at 0.8s
tau_load = tau_load_nom * ones(size(t));
tau_load(t>=0.8) = tau_load_nom + tau_load_step;

% Integrate using RK4
for k = 1:N-1
    % states: x = [i; w]
    x = [i(k); w(k)];
    V = V_in(k);
    tauL = tau_load(k);

    f = @(xv, Vv, tauLv) [ ( -R*xv(1) - Ke*xv(2) + Vv ) / L;
                           ( Kt*xv(1) - b*xv(2) - tauLv ) / J ];

    k1 = f(x, V, tauL);
    k2 = f(x + 0.5*dt*k1, V, tauL);
    k3 = f(x + 0.5*dt*k2, V, tauL);
    k4 = f(x + dt*k3, V, tauL);
    xnext = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);

    i(k+1) = xnext(1);
    w(k+1) = xnext(2);
    theta(k+1) = theta(k) + w(k)*dt;
end

% Add measurement noise to speed (sensor noise)
rng(0); % reproducible
noise_std = 0.5; % rad/s (tune as needed)
w_meas = w + noise_std * randn(size(w));

% Save to CSV for record
T = table(t', i', w', w_meas', theta', tau_load', V_in', 'VariableNames', ...
    {'time','current','omega','omega_meas','theta','tau_load','V_in'});
writetable(T, 'motor_model.csv');

% Plot
figure('Position',[200 200 900 400]);
subplot(2,1,1);
plot(t, V_in); ylabel('V (V)'); title('Input Voltage');
subplot(2,1,2);
plot(t, w, 'LineWidth',1.2); hold on;
plot(t, w_meas, ':'); xlabel('Time (s)'); ylabel('\omega (rad/s)');
legend('True \omega','Measured \omega (noisy)');
title('DC Motor Response (noisy measurement + load step)');
grid on;

saveas(gcf, 'motor_model_plot.png');
fprintf('Saved motor_model.csv and motor_model_plot.png\n');
