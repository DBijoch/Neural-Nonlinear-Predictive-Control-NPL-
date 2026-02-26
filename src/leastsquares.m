%% I. Modelowanie liniowe procesu metodą najmniejszych kwadratów
% Skrypt wyznacza parametry modelu liniowego y(k) = w1*u(k-3) + w2*u(k-4) + w3*y(k-1) + w4*y(k-2)
% i wyznacza błędy MSE dla predykcji jednokrokowej i rekurencyjnej

load('data/data_train.mat');
load('data/data_val.mat');

Y_train = Y_train(:);
Y_val = Y_val(:);

M = X_train';
W = M\Y_train;
Ymod_train = model(W, X_train');
Ymod_val = model(W, X_val');
MSEtrain = mean((Y_train - Ymod_train).^2);
MSEval = mean((Y_val - Ymod_val).^2);

y_hist_train = [Y_train(2); Y_train(1)]; % y(k-1), y(k-2)
Yrec_train = recursive_prediction(W, X_train', N_train, y_hist_train);
MSErectrain = mean((Y_train - Yrec_train).^2);

y_hist_val = [Y_val(2); Y_val(1)]; % y(k-1), y(k-2)
Yrec_val = recursive_prediction(W, X_val', N_val, y_hist_val);
MSErecval = mean((Y_val - Yrec_val).^2);



function [Y] = model(w, X)
    Y = w(1) * X(:,1) + w(2) * X(:,2) + w(3) * X(:,3) + w(4) * X(:,4);
end

function [Yrec_train] = recursive_prediction(w, X, n, y_hist)
    Yrec_train = zeros(n,1);
    for j = 1:n
        x = X(j, :);
        x(3) = y_hist(1); % y(k-1)
        x(4) = y_hist(2); % y(k-2)
        y_mod = model(w, x);
        Yrec_train(j) = y_mod;
        y_hist(2) = y_hist(1);
        y_hist(1) = y_mod;
    end
end

%% Wykresy

fig_folder = fullfile('wykresy', 'leastsquares');
if ~exist(fig_folder, 'dir')
    mkdir(fig_folder);
end

%%%
% Training data
%%%

figure;
stairs(Y_train');
hold on;
stairs(Ymod_train', '--');
grid;
xlabel('k');
ylabel('y');
legend('Dane uczące','Model', 'Location', 'northeast');
title(sprintf('Dane uczące: MSEtrain=%e', MSEtrain));

exportgraphics(gcf, fullfile(fig_folder, 'leastsquares_train.pdf'), 'ContentType', 'vector');

figure;
stairs(Y_train');
hold on;
stairs(Yrec_train', '--');
grid;
xlabel('k');
ylabel('y');
legend('Dane uczące','Model', 'Location', 'northeast');
title(sprintf('Dane uczące (rekurencja): MSEtrain=%e', MSErectrain));

exportgraphics(gcf, fullfile(fig_folder, 'leastsquares_rec_train.pdf'), 'ContentType', 'vector');

figure;
plot(Y_train,Ymod_train,'.');
grid
xlabel('Dane uczące');
ylabel('Model');
title(sprintf('Dane uczące: MSEtrain=%e', MSEtrain));

exportgraphics(gcf, fullfile(fig_folder, 'leastsquares_scatter_train.png'), 'ContentType', 'image');

figure;
plot(Y_train,Yrec_train,'.');
grid
xlabel('Dane uczące');
ylabel('Model');
title(sprintf('Dane uczące (rekurencja): MSEtrain=%e', MSErectrain));
exportgraphics(gcf, fullfile(fig_folder, 'leastsquares_scatter_rec_train.png'), 'ContentType', 'image');

%%%
% Validation data
%%%
figure;
stairs(Y_val');
hold on;
stairs(Ymod_val', '--');
grid;
xlabel('k');
ylabel('y');
legend('Dane weryfikujące','Model', 'Location', 'northeast');
title(sprintf('Dane weryfikujące: MSEval=%e', MSEval));

exportgraphics(gcf, fullfile(fig_folder, 'leastsquares_val.pdf'), 'ContentType', 'vector');

figure;
stairs(Y_val');
hold on;
stairs(Yrec_val', '--');
grid;
xlabel('k');
ylabel('y');
legend('Dane weryfikujące','Model', 'Location', 'northeast');
title(sprintf('Dane weryfikujące (rekurencja): MSEval=%e', MSErecval));

exportgraphics(gcf, fullfile(fig_folder, 'leastsquares_rec_val.pdf'), 'ContentType', 'vector');

figure;
plot(Y_val,Ymod_val,'.');
grid
xlabel('Dane weryfikujące');
ylabel('Model');
title(sprintf('Dane weryfikujące: MSEval=%e', MSEval));

exportgraphics(gcf, fullfile(fig_folder, 'leastsquares_scatter_val.png'), 'ContentType', 'image');

figure;
plot(Y_val,Yrec_val,'.');
grid
xlabel('Dane weryfikujące');
ylabel('Model');
title(sprintf('Dane weryfikujące (rekurencja): MSEval=%e', MSErecval));

exportgraphics(gcf, fullfile(fig_folder, 'leastsquares_scatter_rec_val.png'), 'ContentType', 'image');