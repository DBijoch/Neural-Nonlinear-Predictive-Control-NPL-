%% I. Generowanie zbiorów danych treningowych i walidacyjnych
% Skrypt generuje dane z symulatora procesu dla losowych sygnałów sterujących
% i zapisuje przygotowane wektory wejść/wyjść do plików .mat

[y_train, u_train] = generate_dataset(69, 100); 
[y_val, u_val] = generate_dataset(420, 100);

%%%
% Wykresy i zapis danych
%%%

fig_folder = fullfile('wykresy', 'data');
if ~exist(fig_folder, 'dir')
    mkdir(fig_folder);
end
data_folder = fullfile('data');
if ~exist(data_folder, 'dir')
    mkdir(data_folder);
end

k_min = 5; k_max_train = length(u_train); k_max_val = length(u_val);
N_train = k_max_train - k_min + 1;
N_val = k_max_val - k_min + 1;

X_train = zeros(4, N_train);
X_val = zeros(4, N_val); 
Y_train = zeros(1, N_train);
Y_val = zeros(1, N_val);

for i = k_min:N_train
    X_train(:, i) = [u_train(i-3); u_train(i-4); y_train(i-1); y_train(i-2)];
    Y_train(i) = y_train(i);
end

for i = k_min:N_val
    X_val(:, i) = [u_val(i-3); u_val(i-4); y_val(i-1); y_val(i-2)];
    Y_val(i) = y_val(i);
end

save(fullfile(data_folder, 'data_train.mat'), 'X_train', 'Y_train', 'N_train');
save(fullfile(data_folder, 'data_val.mat'), 'X_val', 'Y_val', 'N_val');

figure;
stairs(u_train);
xlabel('k'); ylabel('u(k)');
title('Sygnał sterujący u(k) - dane treningowe');

exportgraphics(gcf, fullfile(fig_folder, 'data_train_u.pdf'), 'ContentType', 'vector');

figure;
stairs(y_train);
xlabel('k'); ylabel('y(k)');
title('Odpowiedź procesu y(k) - dane treningowe');

exportgraphics(gcf, fullfile(fig_folder, 'data_train_y.pdf'), 'ContentType', 'vector');

figure;
stairs(u_val);
xlabel('k'); ylabel('u(k)');
title('Sygnał sterujący u(k) - dane weryfikujące');

exportgraphics(gcf, fullfile(fig_folder, 'data_val_u.pdf'), 'ContentType', 'vector');

figure;
stairs(y_val);
xlabel('k'); ylabel('y(k)');
title('Odpowiedź procesu y(k) - dane weryfikujące');

exportgraphics(gcf, fullfile(fig_folder, 'data_val_y.pdf'), 'ContentType', 'vector');

function [y, u] = generate_dataset(seed, n_points)
    if nargin < 1
        seed = 42;
        n_points = 100;
    elseif nargin < 2
        n_points = 100;
    end
    
    u_min = -1; u_max = 1;
    u_period = 40; % okres zmian u
    N = n_points * u_period; % całkowita liczba kroków symulacji

    u_values = zeros(1, n_points);

    rng(seed); % ustawienie seeda dla powtarzalności wyników
    for i = 2:n_points
        u_values(i) = u_min + (u_max - u_min) * rand();
    end

    u = zeros(1, N);
    for i = 0:(n_points-1)
        u(i*u_period + 1 : (i+1)*u_period) = u_values(i+1);
    end

    y = zeros(1, N);
    x = zeros(1, N); 

    for k = 5:N
        [y(k), x(k)] = proces10_symulator(u(k-3), u(k-4), x(k-1), x(k-2));
    end
end