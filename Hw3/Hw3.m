%% (a) D1 + Strategy 1- Predictive distribution
clear; clc;

% 0) 檔案載入
load('TrainingSamplesDCT_subsets_8.mat');   % D1_FG, D1_BG, ...
load('Alpha.mat');                          % alpha
test_img = im2double(imread('cheetah.bmp'));
mask     = im2double(imread('cheetah_mask.bmp'));   % 1=FG, 0=BG
alpha_list = alpha(:);

%% 4) 測試新模型 準備 zig-zag 對應
zig_zag_array = load('Zig-Zag Pattern.txt');  
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

% 四個 dataset 
cheetah_data   = {D1_FG, D2_FG, D3_FG, D4_FG};
grass_data   = {D1_BG, D2_BG, D3_BG, D4_BG};
data_set  = {'D1','D2','D3','D4'};

% 兩種 Strategy 的 prior 
prior_files = {'Prior_1.mat','Prior_2.mat'};
prior_names = {'Strategy 1','Strategy 2'};

%% 外層迴圈：Strategy 1 / Strategy 2
for strat = 1:2

    % 載入對應的 prior 檔案（mu0_FG, mu0_BG, W0）
    load(prior_files{strat});    % 會產生 mu0_FG, mu0_BG, W0

    mu0_FG = mu0_FG(:);    % 64x1
    mu0_BG = mu0_BG(:);    % 64x1
    w0     = W0(:);        % 64x1

    fprintf('\n===== %s =====\n', prior_names{strat});

    % ---- 內層迴圈：D1, D2, D3, D4 ----
    for ds = 1:4

        fprintf("\n< Running %s + %s >\n", data_set{ds}, prior_names{strat});

        TrainsampleDCT_FG = cheetah_data{ds};
        TrainsampleDCT_BG = grass_data{ds};

        % Number of samples
        N_FG = size(TrainsampleDCT_FG, 1);
        N_BG = size(TrainsampleDCT_BG, 1);

        Py_cheetah = N_FG / (N_FG + N_BG);
        Py_grass   = 1 - Py_cheetah;

        % ML mean/cov
        mu_FG = mean(TrainsampleDCT_FG, 1)';
        mu_BG = mean(TrainsampleDCT_BG, 1)';
        cov_FG = cov(TrainsampleDCT_FG, 1);
        cov_BG = cov(TrainsampleDCT_BG, 1);

        %% (a) Predictive Distribution 

        PD_error = zeros(length(alpha_list),1);

        for a = 1:length(alpha_list)

            cov0 = diag(alpha_list(a) * w0);

            % BG posterior
            W1BG = (N_BG * cov0) / (cov_BG + N_BG*cov0);
            W2BG =  cov_BG        / (cov_BG + N_BG*cov0);
            muN_BG = W1BG*mu_BG + W2BG*mu0_BG;
            SigmaN_BG = (cov_BG * cov0) / (cov_BG + N_BG*cov0);
            SigmaN_BG_PD = ((cov_BG + SigmaN_BG) + (cov_BG + SigmaN_BG)') / 2;

            % FG posterior
            W1FG = (N_FG * cov0) / (cov_FG + N_FG*cov0);
            W2FG =  cov_FG        / (cov_FG + N_FG*cov0);
            muN_FG = W1FG*mu_FG + W2FG*mu0_FG;
            SigmaN_FG = (cov_FG * cov0) / (cov_FG + N_FG*cov0);
            SigmaN_FG_PD = ((cov_FG + SigmaN_FG) + (cov_FG + SigmaN_FG)') / 2;

            % Classification
            mask_pred = zeros(H-7, W-7);
            for i = 1:(H-7)
                for j = 1:(W-7)
                    blk = test_img(i:i+7, j:j+7);
                    D   = dct2(blk);
                    x64 = D(order_coef).';

                    ll_BG = log_gauss(x64, muN_BG, SigmaN_BG_PD) + log(Py_grass);
                    ll_FG = log_gauss(x64, muN_FG, SigmaN_FG_PD) + log(Py_cheetah);

                    mask_pred(i,j) = (ll_FG > ll_BG);
                end
            end

            PD_error(a) = mean(mask_pred(:) ~= GT(:));
            fprintf(" (a) Alpha=%g, PD error=%.4f\n", alpha_list(a), PD_error(a));
        end

        %% (b) ML 
        mask_ML = zeros(H-7, W-7);
        for i = 1:(H-7)
            for j = 1:(W-7)
                blk = test_img(i:i+7, j:j+7);
                D   = dct2(blk);
                x64 = D(order_coef).';

                ll_BG = log_gauss(x64, mu_BG, cov_BG) + log(Py_grass);
                ll_FG = log_gauss(x64, mu_FG, cov_FG) + log(Py_cheetah);
                mask_ML(i,j) = (ll_FG > ll_BG);
            end
        end
        ML_error = mean(mask_ML(:) ~= GT(:));
        ML_error_vec = ML_error * ones(size(alpha_list));
        fprintf(" (b) ML error = %.4f\n", ML_error);

        %% (c) MAP 

        MAP_error = zeros(length(alpha_list),1);

        for a = 1:length(alpha_list)

            cov0 = diag(alpha_list(a) * w0);

            W1BG = (N_BG * cov0) / (cov_BG + N_BG*cov0);
            W2BG =  cov_BG        / (cov_BG + N_BG*cov0);
            muMAP_BG = W1BG*mu_BG + W2BG*mu0_BG;

            W1FG = (N_FG * cov0) / (cov_FG + N_FG*cov0);
            W2FG =  cov_FG        / (cov_FG + N_FG*cov0);
            muMAP_FG = W1FG*mu_FG + W2FG*mu0_FG;

            mask_MAP = zeros(H-7, W-7);
            for i = 1:(H-7)
                for j = 1:(W-7)
                    blk = test_img(i:i+7, j:j+7);
                    D   = dct2(blk);
                    x64 = D(order_coef).';

                    ll_BG = log_gauss(x64, muMAP_BG, cov_BG) + log(Py_grass);
                    ll_FG = log_gauss(x64, muMAP_FG, cov_FG) + log(Py_cheetah);

                    mask_MAP(i,j) = (ll_FG > ll_BG);
                end
            end

            MAP_error(a) = mean(mask_MAP(:) ~= GT(:));
            fprintf(" (c) Alpha=%g, MAP error=%.4f\n", alpha_list(a), MAP_error(a));
        end

        %% Predictive distribution + ML + MAP plot
        figure; hold on; grid on;
        semilogx(alpha_list, PD_error, 'b', 'LineWidth', 1.4);
        semilogx(alpha_list, MAP_error, 'g', 'LineWidth', 1.4);
        semilogx(alpha_list, ML_error_vec, 'r', 'LineWidth', 1.4);

        % ==== 強制 x 軸為 log（避免 axes 被上一張圖污染）====
        set(gca, 'XScale', 'log');
        
        % ==== 設定 x 軸的 tick 完全和 alpha_list 對齊 ====
        set(gca, 'XTick', alpha_list);
        
        % ==== 設定 tick label =====
        set(gca, 'XTickLabel', {'10^{-4}','10^{-3}','10^{-2}','10^{-1}', ...
                                '1','10','10^{2}','10^{3}','10^{4}'});

        xlabel('\alpha'); ylabel('Probability of error');
        title(sprintf('%s + %s', data_set{ds}, prior_names{strat}));
        legend('PD','MAP','ML');
    end
end
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