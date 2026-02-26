%% II. Modelowanie neuronowe procesu przy użyciu sieci ELM
clear;
linewidth = 1.5; set(0,'DefaultLineLineWidth', linewidth); set(0,'DefaultStairLineWidth', linewidth);
colors = lines; set(groot, 'defaultAxesColorOrder', colors);

% dane
load('data/data_train.mat'); % X_train, Y_train, N_train
load('data/data_val.mat'); % X_val, Y_val, N_val

% liczba neuronów w warstwie ukrytej
K = 70;
seed = 70; % ustalenie ziarna generatora liczb losowych
rng(seed);

inputSize = size(X_train, 1);
outputSize = size(Y_train, 1);
numParams=(inputSize+1)*K+(K+1)*outputSize;

Xtrain = X_train;
Ytrain = Y_train(:);
Xval = X_val;
Yval = Y_val(:);

% losowe wagi pierwszej warstwy
w10 = 2*(rand(K,1)-0.5);
w1 = 2*(rand(K,inputSize)-0.5);

% macierz pomiarów generowana blokowo
Vtrain = [ones(N_train,1) tanh(w10 + w1*Xtrain)'];

% obliczenie optymalnych wag drugiej warstwy
weightsW2 = Vtrain\Ytrain;
w20 = weightsW2(1);
w2 = weightsW2(2:K+1)';

% Zapis modelu
save('elm_params', 'w10', 'w1', 'w20', 'w2');

% obliczenie wyjścia modelu dla wszystkich próbek obu zbiorów (bez rekurencji i z rekurencją)
Ymod_train = model(w10, w1, w20, w2, Xtrain); Ymod_train = Ymod_train(:);
Ymod_val   = model(w10, w1, w20, w2, Xval);   Ymod_val   = Ymod_val(:);
MSEtrain = mean((Ytrain - Ymod_train).^2);
MSEval   = mean((Yval   - Ymod_val).^2);


y_hist_train = [Ytrain(2); Ytrain(1)];
y_hist_val   = [Yval(2);   Yval(1)];
Yrec_train  = recursive_prediction(w10,w1,w20,w2,Xtrain,N_train, y_hist_train);
Yrec_val    = recursive_prediction(w10,w1,w20,w2,Xval,N_val,   y_hist_val);
MSErectrain = mean((Ytrain - Yrec_train).^2);
Mserecval   = mean((Yval   - Yrec_val  ).^2);

fprintf("Extreme learining machine\n");
fprintf(" network architecture K=%d\n", K);
fprintf(" samples %d, weights %d,  MSEtrain=%e, MSEval=%e, MSErectrain=%e, MSErecval=%e\n", N_train, numParams, MSEtrain, MSEval, MSErectrain, Mserecval);

 %%% funkcja wyjście modelu
function Ymod = model(w10, w1, w20, w2, X)
    Ymod = w20+w2*tanh(w10+w1*X);
end
%%% funkcja predykcji rekurencyjnej
function [Yrec_train] = recursive_prediction(w10,w1,w20,w2,X,n,y_hist)
Yrec_train = zeros(n,1);
for j = 1:n
    x = X(:,j);      
    x(3) = y_hist(1);    
    x(4) = y_hist(2);   
    y_mod = model(w10,w1,w20,w2,x);
    Yrec_train(j) = y_mod;
    y_hist = [y_mod; y_hist(1)];
end
end

%% Wykresy
figs = true;
 
fig_folder = fullfile('wykresy', 'elm');
if ~exist(fig_folder, 'dir')
    mkdir(fig_folder);
end

%%%
% Training data
%%%
if figs
figure;
stairs(Ytrain');
hold on;
stairs(Ymod_train', '--');
grid;
xlabel('k');
ylabel('y');
legend('Dane treningowe','Model', 'Location', 'southeast');
title(sprintf('Training data: MSEtrain=%e', MSEtrain));

exportgraphics(gcf, fullfile(fig_folder, ['elm_train_', num2str(K), 'k_' ,num2str(seed), '.pdf']), 'ContentType', 'vector');

figure;
stairs(Ytrain');
hold on;
stairs(Yrec_train', '--');
grid;
xlabel('k');
ylabel('y');
legend('Dane treningowe','Model', 'Location', 'southeast');
title(sprintf('Dane treningowe (Rekurencja): MSEtrain=%e', MSErectrain));

exportgraphics(gcf, fullfile(fig_folder, ['elm_rec_train_', num2str(K), 'k_' ,num2str(seed), '.pdf']), 'ContentType', 'vector');

figure;
plot(Ytrain,Ymod_train,'.');
grid
xlabel('Dane treningowe');
ylabel('Model');
title(sprintf('Dane treningowe: MSEtrain=%e', MSEtrain));

exportgraphics(gcf, fullfile(fig_folder, ['elm_scatter_train_', num2str(K), 'k_' ,num2str(seed), '.png']), 'ContentType', 'image');

figure;
plot(Ytrain,Yrec_train,'.');
grid
xlabel('Dane treningowe');
ylabel('Model');
title(sprintf('Dane treningowe (Rekurencja): MSEtrain=%e', MSErectrain));

exportgraphics(gcf, fullfile(fig_folder, ['elm_scatter_rec_train_', num2str(K), 'k_' ,num2str(seed), '.png']), 'ContentType', 'image');

%%%
% Validation data
%%%

figure;
stairs(Yval');
hold on;
stairs(Ymod_val', '--');
grid;
xlabel('k');
ylabel('y');
legend('Dane weryfikujące','Model', 'Location', 'southeast');
title(sprintf('Dane weryfikujące: MSEval=%e', MSEval));

exportgraphics(gcf, fullfile(fig_folder, ['elm_val_', num2str(K), 'k_' ,num2str(seed), '.pdf']), 'ContentType', 'vector');

figure;
stairs(Yval');
hold on;
stairs(Yrec_val', '--');
grid;
xlabel('k');
ylabel('y');
legend('Dane weryfikujące','Model', 'Location', 'southeast');
title(sprintf('Dane weryfikujące (Rekurencja): MSEval=%e', Mserecval));

exportgraphics(gcf, fullfile(fig_folder, ['elm_rec_val_', num2str(K), 'k_' ,num2str(seed), '.pdf']), 'ContentType', 'vector');

figure;
plot(Yval,Ymod_val,'.');
grid
xlabel('Dane weryfikujące');
ylabel('Model');
title(sprintf('Dane weryfikujące: MSEval=%e', MSEval));

exportgraphics(gcf, fullfile(fig_folder, ['elm_scatter_val_', num2str(K), 'k_' ,num2str(seed), '.png']), 'ContentType', 'image');

figure;
plot(Yval,Yrec_val,'.');
grid
xlabel('Dane weryfikujące');
ylabel('Model');
title(sprintf('Dane weryfikujące (Rekurencja): MSEval=%e', Mserecval));

exportgraphics(gcf, fullfile(fig_folder, ['elm_scatter_rec_val_', num2str(K), 'k_' ,num2str(seed), '.png']), 'ContentType', 'image');
end

