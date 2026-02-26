%% 
%% III. Modelowanie neuronowe procesu przy użyciu Deep Learning Toolbox 
%% Testowane w MATLABie 2025b

clear;
linewidth = 1.5; set(0,'DefaultLineLineWidth',linewidth); set(0,'DefaultStairLineWidth',linewidth);
colors = lines; set(groot, 'defaultAxesColorOrder', colors);

%% dane
load('data/data_train.mat');
load('data/data_val.mat');

K=4;%liczba neuronów w warstwach ukrytych

% konfiguracja algorytmu uczącego
algorithm = 'lbfgs'; % adam, 'lbfgs'
numEpochs = 200;

% ustalenie ziarna generatora liczb losowych
seed = 30;
rng(seed);

% tylko dla 'adam'
learningRate = 0.01;

inputSize = size(X_train, 1);
outputSize = size(Y_train, 1);

Xtrain = dlarray(X_train);
Ytrain = dlarray(Y_train);
Xval = dlarray(X_val);
Yval = dlarray(Y_val);

% Inicjalizacja wag
net.w1 = dlarray(randn(K, inputSize) * 0.1);
net.w10 = dlarray(zeros(K, 1));
net.w2 = dlarray(randn(outputSize, K) * 0.1);
net.w20 = dlarray(zeros(outputSize, 1));
net.w11 = dlarray(randn(inputSize, 1) * 0.1);
net.w22 = dlarray(randn(inputSize, 1) * 0.1);

numParams = (inputSize+1)*K+(K+1)*outputSize+2*inputSize;

% błędy dla całych zbiorów
lossTrainGlobal = zeros(numEpochs, 1);
lossValGlobal = zeros(numEpochs, 1);

fprintf("algorithm epoch/epochmax MSEtrainGlobal MSEvalGlobal\n");
fprintf('%s\n', repmat(char(9472), 1, 45));

% inicjalizacja stanu dla optymalizatorów
trailingAvg = [];
trailingAvgSq = [];
stateLBFGS = [];
iteration = 0;

tic;
switch algorithm
    case 'lbfgs'
        lossFcn =  @(net) dlfeval(@modelLoss, net, Xtrain, Ytrain);
        for epoch = 1:numEpochs            
            [net, stateLBFGS] = lbfgsupdate(net, lossFcn, stateLBFGS);
            
            % Globalny błąd na całym zbiorze uczących i weryfikującym                        
            lossTrain = modelLoss(net, Xtrain, Ytrain);
            lossVal = modelLoss(net, Xval, Yval);
            lossTrainGlobal(epoch) = 2*extractdata(lossTrain);
            lossValGlobal(epoch) = 2*extractdata(lossVal);
            % fprintf("%s: epoch %d/%d, MSEtrain=%e, MSEval=%e\n", ...
                % algorithm, epoch, numEpochs, lossTrainGlobal(epoch), lossValGlobal(epoch));
        end
    case 'adam'
        for epoch = 1:numEpochs 
            iteration = iteration + 1;
            [loss, gradients] = dlfeval(@modelLoss, net, Xtrain, Ytrain);
            [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradients, ...
                trailingAvg, trailingAvgSq, iteration, learningRate);
            % Globalny błąd na całym zbiorze uczących i weryfikującym                        
            lossTrain = modelLoss(net, Xtrain, Ytrain);
            lossVal = modelLoss(net, Xval, Yval);
            lossTrainGlobal(epoch) = 2*extractdata(lossTrain);
            lossValGlobal(epoch) = 2*extractdata(lossVal);
            fprintf("%s: epoch %d/%d, MSEtrain=%e, MSEval=%e\n", ...
                algorithm, epoch, numEpochs, lossTrainGlobal(epoch), lossValGlobal(epoch));
        end
            fprintf("%s: epoch %d/%d, iteration %d, MSEtrain=%e, MSEval=%e\n", ...
                algorithm, epoch, numEpochs, iteration, lossTrainGlobal(epoch), lossValGlobal(epoch));
end
time = toc;

% zapis modelu
save('dl_tbx_params', 'net');

Ymod_train = model(net, Xtrain);
Ymod_val = model(net, Xval);
y_hist_train = [Y_train(2); Y_train(1)];
y_hist_val   = [Y_val(2);   Y_val(1)];
Yrec_train = recursive_prediction(net, Xtrain, N_train, y_hist_train)';
Yrec_val = recursive_prediction(net, Xval, N_val, y_hist_val)';

% extractdata: dlarray -> zwykła macierz
Xtrain = extractdata(Xtrain);
Ytrain = extractdata(Ytrain);
Xval = extractdata(Xval);
Yval = extractdata(Yval);
Ymod_train = extractdata(Ymod_train);
Ymod_val = extractdata(Ymod_val);

MSEtrain = mean((Ytrain - Ymod_train).^2);
MSEval = mean((Yval - Ymod_val).^2);
MSErectrain = mean((Ytrain - Yrec_train).^2);
Mserecval = mean((Yval - Yrec_val).^2);

%%% forward model: wyjście modelu
function Ymod = model(net, X)
z = net.w1*X+net.w10;
z(1,:) = z(1,:)+net.w11(1)*sin(X(2,:));
v = tanh(z);
Ymod = net.w22(1)*cos(X(1,:)) + net.w22(2)*X(2,:) + net.w20 + net.w2*v;
end

%%5 funkcja predykcji rekurencyjnej
function [Yrec_train] = recursive_prediction(net, X, n, y_hist)
Yrec_train = zeros(n,1);
for j = 1:n
    x = X(:,j);
    x(3) = y_hist(1); % y(k-1)
    x(4) = y_hist(2); % y(k-2)
    y_mod = model(net, x);
    Yrec_train(j) = y_mod;
    y_hist = [y_mod; y_hist(1)];
end
end

%%% błąd modelu
function [loss, gradients] = modelLoss(net, X, Y)
Ypred = model(net, X);
loss = mse(Ypred, Y, 'DataFormat', 'CB');

if nargout > 1
    gradients = dlgradient(loss, net);
end
end

fprintf("deep learning toolbox custom loop i własna siec\n");
fprintf(" network architecture K=%d\n", K);
fprintf(" algoritm %s\n", algorithm);
fprintf(" samples %d, weights %d, time=%e\n", N_train, numParams, time);
fprintf(" MSEtrain=%e, MSEval=%e, MSErectrain=%e, MSErecval=%e\n", MSEtrain, MSEval, MSErectrain, Mserecval);


%% Wykresy
figs = true;
figs_save = true;

if figs

fig_folder = fullfile('wykresy', 'dl_tbx');
if ~exist(fig_folder, 'dir')
    mkdir(fig_folder);
end

figure;
plot(1:numEpochs, lossTrainGlobal, '-o', 'DisplayName','Treningowe');
hold on;
plot(1:numEpochs, lossValGlobal, '--s', 'DisplayName','Weryfikujące');
xlabel('Epoka'); ylabel('MSE'); legend; grid on; set(gca,'YScale','log');

if figs_save
exportgraphics(gcf, fullfile(fig_folder, ['dl_tbx_loss_', num2str(K), 'k_' ,algorithm, '_', num2str(seed), '.png']), 'ContentType', 'image');
end


%%%
% Training data
%%%

figure;
stairs(Ytrain');
hold on;
stairs(Ymod_train', '--');
grid;
xlabel('k');
ylabel('y');
legend('Dane treningowe','Model', 'Location', 'southeast');
title(sprintf('Dane treningowe: MSEtrain=%e', MSEtrain));

if figs_save
exportgraphics(gcf, fullfile(fig_folder, ['dl_tbx_train_', num2str(K), 'k_' ,algorithm, '_', num2str(seed), '.pdf']), 'ContentType', 'vector');
end

figure;
stairs(Ytrain');
hold on;
stairs(Yrec_train', '--');
grid;
xlabel('k');
ylabel('y');
legend('Dane treningowe','Model', 'Location', 'southeast');
title(sprintf('Dane treningowe (Rekurencja): MSEtrain=%e', MSErectrain));

if figs_save
exportgraphics(gcf, fullfile(fig_folder, ['dl_tbx_rec_train_', num2str(K), 'k_' ,algorithm, '_', num2str(seed), '.pdf']), 'ContentType', 'vector');
end

figure;
plot(Ytrain,Ymod_train,'.');
grid
xlabel('Dane treningowe');
ylabel('Model');
title(sprintf('Dane treningowe: MSEtrain=%e', MSEtrain));

if figs_save
exportgraphics(gcf, fullfile(fig_folder, ['dl_tbx_scatter_train_', num2str(K), 'k_' ,algorithm, '_', num2str(seed), '.png']), 'ContentType', 'image');
end

figure;
plot(Ytrain,Ymod_train,'.');
grid
xlabel('Dane treningowe');
ylabel('Model');
title(sprintf('Dane treningowe (Rekurencja): MSEtrain=%e', MSErectrain));

if figs_save
exportgraphics(gcf, fullfile(fig_folder, ['dl_tbx_scatter_rec_train_', num2str(K), 'k_' ,algorithm, '_', num2str(seed), '.png']), 'ContentType', 'image');
end

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

if figs_save
exportgraphics(gcf, fullfile(fig_folder, ['dl_tbx_val_', num2str(K), 'k_' ,algorithm, '_', num2str(seed), '.pdf']), 'ContentType', 'vector');
end

figure;
stairs(Yval');
hold on;
stairs(Yrec_val', '--');
grid;
xlabel('k');
ylabel('y');
legend('Dane weryfikujące','Model', 'Location', 'southeast');
title(sprintf('Dane weryfikujące (Rekurencja): MSEval=%e', Mserecval));

if figs_save
exportgraphics(gcf, fullfile(fig_folder, ['dl_tbx_rec_val_', num2str(K), 'k_' ,algorithm, '_', num2str(seed), '.pdf']), 'ContentType', 'vector');
end

figure;
plot(Yval,Ymod_val,'.');
grid
xlabel('Dane weryfikujące');
ylabel('Model');
title(sprintf('Dane weryfikujące: MSEval=%e', MSEval));

if figs_save
exportgraphics(gcf, fullfile(fig_folder, ['dl_tbx_scatter_val_', num2str(K), 'k_' ,algorithm, '_', num2str(seed), '.png']), 'ContentType', 'image');
end

figure;
plot(Yval,Yrec_val,'.');
grid
xlabel('Dane weryfikujące');
ylabel('Model');
title(sprintf('Dane weryfikujące (Rekurencja): MSEval=%e', Mserecval));

if figs_save
exportgraphics(gcf, fullfile(fig_folder, ['dl_tbx_scatter_rec_val_', num2str(K), 'k_' ,algorithm, '_', num2str(seed), '.png']), 'ContentType', 'image');
end

end
