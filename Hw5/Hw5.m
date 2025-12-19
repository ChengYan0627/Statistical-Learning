clear; clc;

%% 0) Load data & images
load('TrainingSamplesDCT_8_new.mat')
test_img = im2double(imread('cheetah.bmp'));
ground_truth = imread('cheetah_mask.bmp');
ground_truth = im2double(ground_truth);     % 轉 double 到 [0,1]
ground_truth = ground_truth > 0.5;          % 二值化 (true/false)

% class priors from counts
N_FG = size(TrainsampleDCT_FG, 1);
N_BG = size(TrainsampleDCT_BG, 1);
Py_cheetah = N_FG / (N_FG + N_BG);
Py_grass = 1 - Py_cheetah;

%% 1) Zig-zag 對應序號
zig_zag_array = load('Zig-Zag Pattern.txt');    % 內容 0..63
zig_zag_array = zig_zag_array + 1;              % 轉成 1..64           
order_coef = zeros(1,64);
for r = 1:8
    for c = 1:8
        k = zig_zag_array(r,c);               % zigzag 序號 1..64
        order_coef(k) = sub2ind([8,8], r, c); % 第 k 個要取 D(r,c) 的線性索引
    end
end

%% 2) Sliding-window
% 取整張 cheetah 圖的所有 8x8 區塊，做 DCT + zig-zag → (H-7)*(W-7) x 64 的矩陣
[H,W] = size(test_img); % 取得影像尺寸

% 預先建立一個大矩陣來存所有測試特徵
% 總共有 (H-7)*(W-7) 個區塊，每個區塊有 64 個特徵
num_blocks = (H-7) * (W-7);
Test_Features = zeros(num_blocks, 64); 

% 矩陣來存正確答案 (Labels)，等一下算錯誤率
Test_Labels = zeros(num_blocks, 1);

% 用一個計數器來記錄現在存到第幾列了
row_count = 0;

for i = 1:(H-7)
    for j = 1:(W-7)
        row_count = row_count + 1; % 換下一列
        block = test_img(i:i+7, j:j+7);
        DCT = dct2(block);         
        
        % 將 DCT 係數按 Zig-zag 順序拉直
        vec = DCT(order_coef); 
        Test_Features(row_count, :) = vec;
        Test_Labels(row_count, 1) = ground_truth(i+4, j+4);
    end
end

%% (a) C=8, FG/BG 各learn 5 次

C = 8;        % 題目要求 Component 數量
runs = 5;     % 題目要求跑 5 次不同的初始化
dims = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]; % 要測試的維度列表

% 1) 初始化儲存空間，各自訓練 FG/BG 的 5 個 GMM（對角共變）
pi_BG = cell(runs,1);  
mu_BG = cell(runs,1);  
var_BG = cell(runs,1);
pi_FG = cell(runs,1);  
mu_FG = cell(runs,1);  
var_FG = cell(runs,1);

% 2) 訓練 FG (獵豹) 模型
for i = 1:runs
    [pi_FG{i}, mu_FG{i}, var_FG{i}] = EM(TrainsampleDCT_FG, C);
end

% 3) 訓練 BG (草地) 模型
for i = 1:runs
    [pi_BG{i}, mu_BG{i}, var_BG{i}] = EM(TrainsampleDCT_BG, C);
end

% 4) 25 種組合的測試與作圖
for j = 1:runs                       % 固定第 j 個 BG mixture
    figure; hold on; grid on;
    title(sprintf('Probability of Error vs. Dimension (BG %d, C=%d)', j, C));
    xlabel('Dimension'); 
    ylabel('Probability of Error');

    for i = 1:runs                   % 內層迴圈：配對第 i 個 FG 模型
            current_errors = zeros(1, numel(dims));
            
            for k = 1:numel(dims)
                d = dims(k);             % 當前使用的特徵維度
                X = Test_Features(:, 1:d); % 取出測試資料的前 d 維
                
                % === 1. 計算 FG (獵豹) 總機率 ===
                % 使用第 i 組參數
                prob_FG = zeros(size(X, 1), 1);
                for c = 1:C
                    % 取出第 c 個高斯的參數
                    mu = mu_FG{i}(c, 1:d);
                    sigma = var_FG{i}(c, 1:d); % 變異數向量
                    weight = pi_FG{i}(c);
                    
                    % 累加: 權重 * 高斯機率
                    prob_FG = prob_FG + weight * multv_npdf(X, mu, sigma);
                end
                % 乘上先驗機率 P(Cheetah)
                total_FG = prob_FG * Py_cheetah;
                
                
                % === 2. 計算 BG (草地) 總機率 ===
                % 使用第 j 組參數
                prob_BG = zeros(size(X, 1), 1);
                for c = 1:C
                    mu = mu_BG{j}(c, 1:d);
                    sigma = var_BG{j}(c, 1:d);
                    weight = pi_BG{j}(c);
                    
                    prob_BG = prob_BG + weight * multv_npdf(X, mu, sigma);
                end
                % 乘上先驗機率 P(Grass)
                total_BG = prob_BG * Py_grass;
                
                
                % === 3. Bayes 決策與錯誤率 ===
                pred = (total_FG > total_BG);          % 1=Cheetah, 0=Grass
                current_errors(k) = mean(pred ~= Test_Labels);
            end

        plot(dims, current_errors, '-o', 'LineWidth', 1.2, 'DisplayName', sprintf('FG #%d', i));
    end

    legend('show', 'Location', 'best');
end

%% (b) C ∈ {1,2,4,8,16,32}，每個 C 一條曲線

% 1. 設定參數
C_list = [1, 2, 4, 8, 16, 32];  % 題目要測試的 C (沙堆數量)
dims = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]; % 維度列表

% 準備畫布
figure; 
hold on; grid on;
title('PoE vs. Dimension (Different C)');
xlabel('Dimension');
ylabel('Probability of Error');

% 設定顏色 (讓每一條線不一樣)
colors = {'k', 'b', 'c', 'g', 'm', 'r'}; % 黑藍青綠紫紅

% 2. 外層迴圈：測試不同的 C (沙堆數量)
for i = 1:length(C_list)
    C = C_list(i);
    
    % --- 步驟 A: 訓練模型 ---
    % 呼叫 EM 函數算出參數 (pi, mu, sigma)
    % 為了簡單，我們只訓練一次
    [pi_fg, mu_fg, sig_fg] = EM(TrainsampleDCT_FG, C);
    [pi_bg, mu_bg, sig_bg] = EM(TrainsampleDCT_BG, C);
    
    % --- 步驟 B: 測試 (計算錯誤率) ---
    current_errors = zeros(1, length(dims));
    
    % 針對每個維度進行測試
    for k = 1:length(dims)
        d = dims(k); % 目前是用前 d 個特徵
        
        % 取出測試資料的前 d 個特徵
        X = Test_Features(:, 1:d);
        
        % === 這裡用最直白的方式算機率 (類似學長的寫法) ===
        
        % 1. 算獵豹 (FG) 的總機率
        prob_FG = zeros(size(X, 1), 1); % 先開一個全 0 的計分板
        for c = 1:C
            % 公式：權重 * 高斯機率
            % multv_npdf 是 Matlab 內建函數，幫你算高斯分佈
            pdf_val = multv_npdf(X, mu_fg(c, 1:d), sig_fg(c, 1:d));
            prob_FG = prob_FG + pi_fg(c) * pdf_val;
        end
        % 乘上先驗機率 P(Cheetah)
        total_score_FG = prob_FG * Py_cheetah;
        
        
        % 2. 算草地 (BG) 的總機率
        prob_BG = zeros(size(X, 1), 1);
        for c = 1:C
            pdf_val = multv_npdf(X, mu_bg(c, 1:d), sig_bg(c, 1:d));
            prob_BG = prob_BG + pi_bg(c) * pdf_val;
        end
        % 乘上先驗機率 P(Grass)
        total_score_BG = prob_BG * Py_grass;
        
        
        % 3. 比較分數 (BDR)
        prediction = total_score_FG > total_score_BG;
        
        % 4. 算錯了幾個
        current_errors(k) = sum(prediction ~= Test_Labels) / length(Test_Labels);
    end
    
    % --- 步驟 C: 畫圖 ---
    plot(dims, current_errors, '-o', 'Color', colors{i}, 'LineWidth', 1.2) 

end

% 加上圖例
legend('C=1', 'C=2', 'C=4', 'C=8', 'C=16', 'C=32');


%% 輔助函數 
% EM 演算法
function [pi_c, mu_c, sigma_c] = EM(data, C)
    [N, ~] = size(data);
    
    % --- 初始化 (Initialization) ---
    % 隨機選 C 個樣本點作為初始中心 (比 rand() 更好)
    rand_idx = randperm(N, C);
    mu_c = data(rand_idx, :);
    % 初始變異數設為整體的變異數，避免太小
    sigma_c = repmat(var(data) + 0.001, C, 1);
    % 權重均分
    pi_c = ones(1, C) / C;
    
    for iter = 1:50 % 迭代 50 次通常夠了
        % --- E-Step ---
        % 計算 Log Likelihood (N x C)
        log_prob = zeros(N, C);
        for c = 1:C
             % 高斯公式的 Log 版: -0.5 * log(2pi*sigma) - (x-mu)^2 / 2sigma
             % 省略常數項 log(2pi) 因為不影響後續權重計算
             diff = data - mu_c(c, :);
             term1 = sum(log(sigma_c(c, :))); 
             term2 = sum((diff.^2) ./ sigma_c(c, :), 2);
             log_prob(:, c) = log(pi_c(c)) - 0.5 * (term1 + term2);
        end
        
        % Log-Sum-Exp 技巧 (防止數值變成 0)
        max_log = max(log_prob, [], 2);
        prob_norm = exp(log_prob - max_log);
        responsibilities = prob_norm ./ sum(prob_norm, 2);
        
        % --- M-Step ---
        Nk = sum(responsibilities, 1);
        for c = 1:C
            pi_c(c) = Nk(c) / N;
            mu_c(c, :) = (responsibilities(:, c)' * data) / Nk(c);
            % 更新 Variance (Diagonal)
            diff = data - mu_c(c, :);
            sigma_c(c, :) = (responsibilities(:, c)' * (diff.^2)) / Nk(c);
            sigma_c(c, sigma_c(c,:) < 1e-4) = 1e-4; % 防止變異數過小
        end
    end
end

% Multivariate Normal PDF funtion
function y = multv_npdf(X, mu, var_vec)
    % X: 資料矩陣 (N x D)
    % mu: 平均值向量 (1 x D)
    % var_vec: 變異數向量 (1 x D)，對應對角矩陣的對角線元素
    
    [N, D] = size(X); % 
    
    % 1. 計算常數項
    % 使用 prod 計算對角矩陣的行列式 (Determinant)
    % 公式: 1 / sqrt((2*pi)^D * det(Sigma))
    det_sigma = prod(var_vec); % 
    const = 1 / sqrt((2*pi)^D * det_sigma); 
    
    y = zeros(N, 1); 
    
    % 2. 計算指數項
    % 由於不能確定 sum 是否支援維度參數 (dim)，使用 for 迴圈最保險 
    for i = 1:N
        % 取出一筆資料
        x_row = X(i, :);
        
        % 計算差值 (x - mu)
        diff = x_row - mu;
        
        % 計算 Mahalanobis Distance 的平方部分
        % 因為是 Diagonal Matrix，可以直接用元素除法 ./
        % 公式: (x-mu) * inv(Sigma) * (x-mu)' 簡化為 sum((diff^2) ./ var)
        exponent = -0.5 * sum((diff.^2) ./ var_vec); 

        y(i) = const * exp(exponent); 
    end
end
