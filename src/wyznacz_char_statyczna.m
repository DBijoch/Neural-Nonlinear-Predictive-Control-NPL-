% Wyznaczanie charakterystyki statycznej y(u) metodą symulacyjną
clear all;
grubosc = 1.25; set(groot,'DefaultLineLineWidth',grubosc); set(groot,'DefaultStairLineWidth',grubosc);
colors = lines; set(groot, 'defaultAxesColorOrder', colors);

u_min = -1; u_max = 1; n_points = 100;
kmin = 5; k_sim = 1000;

u_values = linspace(u_min, u_max, n_points);
y_ss = zeros(size(u_values));

for i = 1:length(u_values)
    u_val = u_values(i);
    u = u_val * ones(1, k_sim);
    y = zeros(1, k_sim);
    x = zeros(1, k_sim);

    for k = kmin:k_sim
        [y(k), x(k)] = proces10_symulator(u(k-3), u(k-4), x(k-1), x(k-2));
    end

    n_avg = min(20, k_sim - kmin + 1);
    y_ss(i) = mean(y(end-n_avg+1:end));
end


figure;
plot(u_values, y_ss, 'LineWidth', 1.5);
grid on;
xlabel('u'); ylabel('y');
title('Charakterystyka statyczna');

fig_folder = fullfile('wykresy', 'cs');
if ~exist(fig_folder, 'dir')
    mkdir(fig_folder);
end

exportgraphics(gcf, fullfile(fig_folder, 'cs.pdf'), 'ContentType', 'vector');