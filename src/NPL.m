%% IV. Nieliniowe sterowanie predykcyjne (NPL) z użyciem modeli neuronowych
% Skrypt implementuje algorytm MPC z linearyzacją modelu neuronowego (ELM lub hybrydowego)
% w każdym kroku sterowania i testuje regulator na zadanej trajektorii

warning off;

elm = load('data/elm_params');
dl_tbx = load('data/dl_tbx_params');
u_min = -1; u_max = 1;

kk = 250;
kp = 5; 

nA = 2;
nB = 4;

N = 25;
Nu = 4;
lambda = 10;

yzad = zeros(1, kk);
yzad(1:20) = 0;
yzad(21:80) = 0.4;
yzad(81:140) = -1.5;
yzad(141:190) = -0.5;
yzad(191:250) = 0.7;


U = zeros(1, kk);
Y = zeros(1, kk);
X = zeros(1, kk);

Yelm = zeros(1, kk);
Ydl = zeros(1, kk);

model = 'hybrydowy'; % 'elm' lub 'hybrydowy'


for k=kp:kk
    
    %%%
    % 1. Pomiar y(k)
    %%%

    [Y(k), X(k)] = proces10_symulator(U(k-3), U(k-4), X(k-1), X(k-2));
    
    %%%
    % 2. Obliczenie błędu modelu
    %%%
    switch model
        case 'elm'
            % 2. Model i błąd
            Yelm(k) = model_elm(elm.w10, elm.w1, elm.w20, elm.w2, [U(k-3); U(k-4); Y(k-1); Y(k-2)]);
            dmod = Y(k) - Yelm(k);

            % 3. Odpowiedź swobodna
            Y0 = zeros(N, 1);
            Y0(1) = model_elm(elm.w10, elm.w1, elm.w20, elm.w2, [U(k-2); U(k-3); Yelm(k); Yelm(k-1)]) + dmod;
            Y0(2) = model_elm(elm.w10, elm.w1, elm.w20, elm.w2, [U(k-1); U(k-2); Y0(1); Yelm(k)]) + dmod;
            for p = 3:N
                Y0(p) = model_elm(elm.w10, elm.w1, elm.w20, elm.w2, [U(k-1); U(k-1); Y0(p-1); Y0(p-2)]) + dmod;
            end

            % 4. Linearyzacja
            [a1, a2, b3, b4] = linearize_model_elm(elm, U(k-3), U(k-4), Yelm(k-1), Yelm(k-2));

        case 'hybrydowy'
            % 2. Model i błąd
            Ydl(k) = model_dl(dl_tbx.net, [U(k-3); U(k-4); Y(k-1); Y(k-2)]);
            dmod = Y(k) - Ydl(k);

            % 3. Odpowiedź swobodna
            Y0 = zeros(N, 1);
            Y0(1) = model_dl(dl_tbx.net, [U(k-2); U(k-3); Ydl(k); Ydl(k-1)]) + dmod;
            Y0(2) = model_dl(dl_tbx.net, [U(k-1); U(k-2); Y0(1); Ydl(k)]) + dmod;
            for p = 3:N
                Y0(p) = model_dl(dl_tbx.net, [U(k-1); U(k-1); Y0(p-1); Y0(p-2)]) + dmod;
            end

            % 4. Linearyzacja
            [a1, a2, b3, b4] = linearize_model_dl(dl_tbx, U(k-3), U(k-4), Ydl(k-1), Ydl(k-2));
    end

    % 5. Obliczenie s(k)
    S = zeros(1, N);
    a = [a1, a2];
    b = [0, 0, b3, b4];

    for q = 1:N
        for i = 1:min(q, nB)
            S(q) = S(q) + b(i);
        end
        for i = 1:min(q - 1, nA)
            S(q) = S(q) - a(i) * S(q - i);
        end
    end

    % 6. M(k)
    M = zeros(N, Nu);
    for row = 1:N
        for col = 1:Nu
            if row >= col
                M(row, col) = S(row - col + 1);
            end
        end
    end

    % 7. K(k)
    K = (M' * M + lambda * eye(Nu)) \ (M');

    % 8. delta U(k)
    dU = K * (yzad(k) * ones(N, 1) - Y0);

    %%%
    % 9. Sterowanie 
    %%%
    
    U(k) = max(min(U(k - 1) + dU(1), u_max), u_min);
    
end



function Ymod = model_elm(w10, w1, w20, w2, X)
    Ymod = w20+w2*tanh(w10+w1*X);
end

function Ymod = model_dl(net, X)
    z = net.w1*X+net.w10;
    z(1,:) = z(1,:)+net.w11(1)*sin(X(2,:));
    v = tanh(z);
    Ymod = net.w22(1)*cos(X(1,:)) + net.w22(2)*X(2,:) + net.w20 + net.w2*v;
end

function [a1, a2, b3, b4] = linearize_model_elm(elm, ukm3, ukm4, ykm1, ykm2)
    delta = 1e-5;
    f0 = model_elm(elm.w10, elm.w1, elm.w20, elm.w2, [ukm3; ukm4; ykm1; ykm2]);
    
    f_ukm3 = model_elm(elm.w10, elm.w1, elm.w20, elm.w2, [ukm3 + delta; ukm4; ykm1; ykm2]);
    b3 = (f_ukm3 - f0) / delta;
    
    f_ukm4 = model_elm(elm.w10, elm.w1, elm.w20, elm.w2, [ukm3; ukm4 + delta; ykm1; ykm2]);
    b4 = (f_ukm4 - f0) / delta;

    f_ykm1 = model_elm(elm.w10, elm.w1, elm.w20, elm.w2, [ukm3; ukm4; ykm1 + delta; ykm2]);
    a1 = - (f_ykm1 - f0) / delta;

    f_ykm2 = model_elm(elm.w10, elm.w1, elm.w20, elm.w2, [ukm3; ukm4; ykm1; ykm2 + delta]);
    a2 = - (f_ykm2 - f0) / delta;
end

function [a1, a2, b3, b4] = linearize_model_dl(dl_tbx, ukm3, ukm4, ykm1, ykm2)
    delta = 1e-5;
    f0 = model_dl(dl_tbx.net, [ukm3; ukm4; ykm1; ykm2]);
    
    f_ukm3 = model_dl(dl_tbx.net, [ukm3 + delta; ukm4; ykm1; ykm2]);
    b3 = (f_ukm3 - f0) / delta;
    
    f_ukm4 = model_dl(dl_tbx.net, [ukm3; ukm4 + delta; ykm1; ykm2]);
    b4 = (f_ukm4 - f0) / delta;

    f_ykm1 = model_dl(dl_tbx.net, [ukm3; ukm4; ykm1 + delta; ykm2]);
    a1 = - (f_ykm1 - f0) / delta;

    f_ykm2 = model_dl(dl_tbx.net, [ukm3; ukm4; ykm1; ykm2 + delta]);
    a2 = - (f_ykm2 - f0) / delta;
end

%% Wykresy

figs = true;
figs_save = true;

if figs
    if figs_save
        fig_folder = fullfile('wykresy', ['npl_', model]);
        if ~exist(fig_folder, 'dir')
            mkdir(fig_folder);
        end
    end

    figure;
    stairs(Y, 'LineWidth', 1.5);
    hold on;
    stairs(yzad, '--', 'LineWidth', 1.5);
    grid;
    xlabel('k');
    ylabel('y, y_{zad}');
    legend('y', 'y_{zad}', 'Location', 'best');
    title(['Wyjście procesu - model ',  model, '- N = ', num2str(N), ', Nu = ', num2str(Nu), ', \lambda = ', num2str(lambda)]);
    xlim([0 kk]);
    ylim([-1.7 1]);

    if figs_save
        exportgraphics(gcf, fullfile(fig_folder, [model, '_', num2str(N), '_', num2str(Nu), '_', num2str(lambda), '_y.pdf']), 'ContentType', 'vector');
    end

    figure;
    stairs(U, 'LineWidth', 1.5);
    grid;
    xlabel('k');
    ylabel('u');
    title(['Sterowanie u - model ',  model, '- N = ', num2str(N), ', Nu = ', num2str(Nu), ', \lambda = ', num2str(lambda)]);
    xlim([0 kk]);
    ylim([-1 1]);

    if figs_save
        exportgraphics(gcf, fullfile(fig_folder, [model, '_', num2str(N), '_', num2str(Nu), '_', num2str(lambda), '_u.pdf']), 'ContentType', 'vector');
    end


end
