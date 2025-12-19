%% (a) D1 + Strategy 1- Predictive distribution
clear; clc;

% 0) 檔案載入
load('TrainingSamplesDCT_subsets_8.mat');   % D1_FG, D1_BG, ...
load('Prior_1.mat');                        % mu0_FG, mu0_BG, W0
load('Alpha.mat');                          % alpha
test_img = im2double(imread('cheetah.bmp'));
mask     = im2double(imread('cheetah_mask.bmp'));   % 1=FG, 0=BG

% 1) 用 D1 當訓練資料
TrainsampleDCT_FG = D1_FG;
TrainsampleDCT_BG = D1_BG;

N_FG = size(TrainsampleDCT_FG, 1);
N_BG = size(TrainsampleDCT_BG, 1);
Py_cheetah = N_FG / (N_FG + N_BG);
Py_grass = 1 - Py_cheetah;

% ML 算D1統計量（每類）：均值 / 協方差
mu_FG   = mean(TrainsampleDCT_FG, 1)';     % 64x1
mu_BG   = mean(TrainsampleDCT_BG, 1)';     % 64x1
cov_FG = cov(TrainsampleDCT_FG, 1);       % 64x64
cov_BG = cov(TrainsampleDCT_BG, 1);       % 64x64

% 類別先驗的 log（之後迴圈裡一直用，先算好）
logPy_cheetah = log(Py_cheetah);
logPy_grass   = log(Py_grass);

% 2) 先驗（Strategy 1 prior）：mu0 與 Σ0 = α * diag(w)
mu0_FG = mu0_FG(:);      % 64x1
mu0_BG = mu0_BG(:);      % 64x1
w0 = W0(:);               % 64x1  

%% 4) 測試新模型 準備 zig-zag 對應
zig_zag_array = load('Zig-Zag Pattern.txt');  % 內容多半是 0..63
zig_zag_array = double(zig_zag_array) + 1;    % 轉成 1..64

order_coef = zeros(1, 64);
for r = 1:8
    for c = 1:8
        k = zig_zag_array(r,c);               % zigzag 序號 1..64
        order_coef(k) = sub2ind([8,8], r, c); % 第 k 個要取 D(r,c) 的線性索引
    end
end

% 影像與滑窗參數
[H,W] = size(test_img);

% Ground truth 對齊：只取到能滑動 block 的範圍
GT = mask(1:(H-7), 1:(W-7)) > 0.5;

%% 3) 掃 alpha(不同α的迴圈)
alpha_list = alpha(:);
PD_error   = zeros(length(alpha_list),1);

for a = 1:length(alpha_list)

    % 設定經驗模糊程度: 先驗共變異數 cov0 = alpha * diag(w0)
    cov0 = diag(alpha_list(a) * w0);   % 64x64

    % 3.1 計算「折衷」後的獵豹長相 (後驗均值 muN)
    
    % ===== BG 類別 (grass) 的後驗均值 / 共變異數 =====

    W1BG  = (N_BG * cov0) / (cov_BG + N_BG * cov0);  % W1BG, W2BG 為對應權重
    W2BG  = cov_BG / (cov_BG + N_BG * cov0);           % 64x64

    % 後驗均值 muN_BG 
    muN_BG = W1BG * mu_BG + W2BG * mu0_BG;

    % 後驗共變異數 SigmaN_BG
    SigmaN_BG = (cov_BG * cov0) / (cov_BG + N_BG * cov0);

    % Predictive 共變異數：Cov_BG + SigmaN_BG，順便 symmetrize 一下
    SigmaN_BG_PD = ((cov_BG + SigmaN_BG) + (cov_BG + SigmaN_BG)') / 2;

    % ===== FG 類別 (cheetah) 的後驗均值 / 共變異數 =====

    W1FG  = (N_FG * cov0) / (cov_FG + N_FG * cov0);
    W2FG  = cov_FG / (cov_FG + N_FG * cov0);

    muN_FG = W1FG * mu_FG + W2FG * mu0_FG;

    SigmaN_FG = (cov_FG * cov0) / (cov_FG + N_FG * cov0);
    SigmaN_FG_PD = ((cov_FG + SigmaN_FG) + (cov_FG + SigmaN_FG)') / 2;


    % 4)用新模型 (muN, SigmaN_PD) 去看測試圖片

    % 用一個 log Gaussian 函式 ---
    maskMatrix = zeros(H-7, W-7);   % (i,j) 是 block 左上角
    
    for i = 1:(H-7)
        for j = 1:(W-7)
    
            % 取 8x8 block, 做 DCT, zig-zag 展開成 64x1
            blk = test_img(i:i+7, j:j+7);
            D   = dct2(blk);
            x64 = D(order_coef).';      % 64x1 column
    
            % BG: log p(x|BG) + log P(BG)
            BDR_grass = log_gauss(x64, muN_BG, SigmaN_BG_PD) + log(Py_grass);
    
            % FG: log p(x|FG) + log P(FG)
            BDR_cheetah = log_gauss(x64, muN_FG, SigmaN_FG_PD) + log(Py_cheetah);
    
            % Bayes decision：誰的 log posterior 比較大就選誰
            if BDR_cheetah > BDR_grass
                maskMatrix(i,j) = 1;   % 1 = cheetah
            else
                maskMatrix(i,j) = 0;   % 0 = grass
            end
        end
    end


    % (對答案)計算這個 alpha 下的錯誤率 PE 
    PD_error(a) = mean( maskMatrix(:) ~= GT(:) );
    fprintf('(a) Alpha=%g, Probability error=%.4f\n', alpha_list(a), PD_error(a));
end

% 6) 畫 PE(α)
% figure;
% semilogx(alpha_list, PD_error, '-o', 'LineWidth', 1.4); grid on;
% xlabel('\alpha'); ylabel('PE');
% title('(a) Predictive — D1, Strategy 1');

%% (b) D1 + Strategy 1- ML
% 0) ML 分類器
mask_ML = zeros(H-7, W-7);
for i = 1:(H-7)
    for j = 1:(W-7)
        blk = test_img(i:i+7, j:j+7);    % 8x8 block
        D   = dct2(blk);
        x64 = D(order_coef).';           % 64x1 feature

        % BG 的 log posterior ~ log p(x|BG) + log P(BG)
        ll_BG = log_gauss(x64, mu_BG, cov_BG) + log(Py_grass);

        % FG 的 log posterior ~ log p(x|FG) + log P(FG)
        ll_FG = log_gauss(x64, mu_FG, cov_FG) + log(Py_cheetah);

        if ll_FG > ll_BG
            mask_ML(i,j) = 1;    % cheetah
        else
            mask_ML(i,j) = 0;    % grass
        end
    end
end

% 1) 計算 ML 的 PE（單一數值）
ML_error = mean( mask_ML(:) ~= GT(:) );
fprintf('(b) ML error = %.4f\n', ML_error);

% 2) 畫 PE(α) — ML 是一條水平線
ML_error_vec = ML_error * ones(size(alpha_list));  % 複製成和 alpha 一樣長的向量

% figure;
% semilogx(alpha_list, ML_error_vec, '--', 'LineWidth', 1.4); grid on;
% xlabel('\alpha'); ylabel('PE');
% title('(b) ML — D1');

%% (c) D1 + Strategy 1- MAP
% 0) prior 參數
MAP_error = zeros(length(alpha_list),1);

% 1) 對每個 alpha 做一次 MAP 分類
for a = 1:length(alpha_list)

    % 5.1 先驗 cov0 = alpha * diag(w0)
    cov0 = diag(alpha_list(a) * w0);   % 64x64

    % ---- BG：posterior mean = mu_MAP_BG ----
    W1BG   = (N_BG * cov0) / (cov_BG + N_BG * cov0);
    W2BG   =  cov_BG        / (cov_BG + N_BG * cov0);
    muMAP_BG = W1BG * mu_BG + W2BG * mu0_BG;

    % ---- FG：posterior mean = mu_MAP_FG ----
    W1FG   = (N_FG * cov0) / (cov_FG + N_FG * cov0);
    W2FG   =  cov_FG        / (cov_FG + N_FG * cov0);
    muMAP_FG = W1FG * mu_FG + W2FG * mu0_FG;

    % 重要：MAP classifier 的 covariance 用 ML 的 cov_FG / cov_BG
    % (a) Predictive 用的是：變寬的 SigmaN_BG_PD ; (c) MAP 用的是：原始窄窄的 cov_BG
    mask_MAP = zeros(H-7, W-7);

    for i = 1:(H-7)
        for j = 1:(W-7)
            blk = test_img(i:i+7, j:j+7);
            D   = dct2(blk);
            x64 = D(order_coef).';      % 64x1

            % BG: log p(x | muMAP_BG, cov_BG) + log P(BG)
            ll_BG = log_gauss(x64, muMAP_BG, cov_BG) + log(Py_grass);

            % FG: log p(x | muMAP_FG, cov_FG) + log P(FG)
            ll_FG = log_gauss(x64, muMAP_FG, cov_FG) + log(Py_cheetah);

            if ll_FG > ll_BG
                mask_MAP(i,j) = 1;
            else
                mask_MAP(i,j) = 0;
            end
        end
    end

    % 5.2 計算這個 alpha 下的 MAP PE
    MAP_error(a) = mean( mask_MAP(:) ~= GT(:) );
    fprintf('(c) Alpha = %g, Probability error = %.4f\n', alpha_list(a), MAP_error(a));
end

% 3) 畫 MAP 的 PE(alpha)
% figure;
% semilogx(alpha_list, MAP_error, '-o', 'LineWidth', 1.4); grid on;
% xlabel('\alpha'); ylabel('PE');
% title('(c) MAP — D_1, Strategy 1');

%% D1 + Strategy 1- Predictive distribution + ML + MAP plot

figure; hold on; grid on;
semilogx(alpha_list, PD_error, 'b',  'LineWidth', 1.4);      % Predictive
semilogx(alpha_list, MAP_error, 'g', 'LineWidth', 1.4);      % MAP
semilogx(alpha_list, ML_error_vec, 'r', 'LineWidth', 1.4);   % ML (horizontal)


% ==== 強制 x 軸為 log（避免 axes 被上一張圖污染）====
set(gca, 'XScale', 'log');

% ==== 設定 x 軸的 tick 完全和 alpha_list 對齊 ====
set(gca, 'XTick', alpha_list);

% ==== 設定 tick label =====
set(gca, 'XTickLabel', {'10^{-4}','10^{-3}','10^{-2}','10^{-1}', ...
                        '1','10','10^{2}','10^{3}','10^{4}'});

xlabel('\alpha'); 
ylabel('PE');
title('D1, Strategy 1');
legend('Predictive', 'MAP', 'ML', 'Location', 'best');

%% log function
function log_p = log_gauss(x, mu, Sigma)
    % 把 row/col 都轉成 column vector
    x  = x(:);
    mu = mu(:);

    diff = x - mu;
    % (x - mu)' * inv(Sigma) * (x - mu)
    quad = diff' * (Sigma \ diff);

    % log(det(Sigma))
    ld = log(det(Sigma));

    % d = 維度（64）
    d = length(x);

    % 直接套公式： -0.5 * [ quadratic + logdet + d*log(2π) ]
    log_p = -0.5 * (quad + ld + d*log(2*pi));
end