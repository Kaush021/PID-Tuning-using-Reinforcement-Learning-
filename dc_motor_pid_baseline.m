clc; clear; close all;

R = 1.0; L = 0.5e-3; Kt = 0.01; Ke = 0.01; J = 1e-4; b = 1e-4;

Tfinal = 1.5; dt = 0.001; t = 0:dt:Tfinal;
N = length(t);

Kp = 25.0;
Ki = 50.0;
Kd = 0.0005;

i = zeros(1,N);
w = zeros(1,N);
theta = zeros(1,N);
w_meas = zeros(1,N);
integral = 0; prev_err = 0;
ref = zeros(1,N);
ref(t >= 0.02) = 100.0; 

tau_load = zeros(size(t));
tau_load(t>=0.8) = 0.001;

for k = 1:N-1
    error = ref(k) - w(k);
    integral = integral + error*dt;
    derivative = (error - prev_err)/dt;
    prev_err = error;
    V = Kp*error + Ki*integral + Kd*derivative;
    V = max(min(V,24), -24);

    di = ( -R*i(k) - Ke*w(k) + V ) / L;
    dw = ( Kt*i(k) - b*w(k) - tau_load(k) ) / J;

    i(k+1) = i(k) + di*dt;
    w(k+1) = w(k) + dw*dt;
    theta(k+1) = theta(k) + w(k)*dt;
    w_meas(k) = w(k) + 0.5*randn(); 
end

w_meas(end) = w(end) + 0.5*randn();

figure('Position',[200 200 900 300]);
plot(t, ref, '--','DisplayName','ref'); hold on;
plot(t, w_meas,'DisplayName','measured \omega');
xlabel('Time (s)'); ylabel('\omega (rad/s)');
title('Baseline PID Response (Member-2)');
legend(); grid on;
saveas(gcf, 'baseline_pid_response.png');

final = ref(end);

overshoot = (max(w) - ref(find(ref>0,1,'first'))) / ref(find(ref>0,1,'first')) * 100;

T = table(t', w', w_meas', tau_load', 'VariableNames', {'time','omega','omega_meas','tau_load'});
writetable(T, 'baseline_pid_data.csv');
fprintf('Saved baseline_pid_response.png and baseline_pid_data.csv\n');