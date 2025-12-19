%% a) 
load('TrainingSamplesDCT_8_new.mat')
N_FG = size(TrainsampleDCT_FG);
N_BG = size(TrainsampleDCT_BG);
N = N_FG(1) + N_BG(1);
Py_cheetah = N_FG(1)/ N;
Py_grass = N_BG(1)/ N;


%% b) Visual Feature Selection with subplots

% 1) Estimate class-conditional Gaussian parameters (MLE)
u_FG  = mean(TrainsampleDCT_FG, 1);            % sample mean for cheetah(對每一特徵維取mean)
u_BG  = mean(TrainsampleDCT_BG, 1);            % sample mean for grass(對每一特徵維取mean)

cov_FG = cov(TrainsampleDCT_FG, 1);            % covariance for cheetah
cov_BG = cov(TrainsampleDCT_BG, 1);            % covariance for grass

var_FG = diag(cov_FG)';    % foreground 每維的 variance
var_BG = diag(cov_BG)';    % background 每維的 variance

% 2) Loop all 64 features - plot one by one for CLEAR visual inspection
%    （這一段幫助你"看清楚"每一維，人工挑選 best/worst 候選）
figure(1); clf;
sgtitle('Marginal Gaussian PDFs for All 64 DCT Features');
for k = 1:64
    subplot(8,8,k);
    hold on; grid on; box on;

    %這四個數（mF, mB, sF, sB）就是兩條 1D 高斯的參數
    sd_Fx = sqrt(var_FG(k));
    sd_Bx = sqrt(var_BG(k));
    u_Fx = u_FG(k); 
    u_Bx = u_BG(k);

    %設定 繪圖的 x 範圍
    lo_bound = min([u_Fx-4*sd_Fx, u_Bx-4*sd_Bx]);
    hi_bound = max([u_Fx+4*sd_Fx, u_Bx+4*sd_Bx]);
    x  = linspace(lo_bound, hi_bound, 300);

    %Calculate the Gaussian PDFs for the current feature
    G_Fx = normpdf(x, u_Fx, sd_Fx);
    G_Bx = normpdf(x, u_Bx, sd_Bx);

    % Different line styles as required
    plot(x, G_Fx, 'r-',  'LineWidth', 1.6); % FG: red solid
    plot(x, G_Bx, 'b--', 'LineWidth', 1.6); % BG: blue dashed

    % --- 子圖標題與外觀 ---
    title(sprintf('Feature #%d', k), 'FontSize', 7, 'FontWeight', 'bold');
    set(gca, 'XTick', [], 'YTick', []);  % 移除刻度
    axis tight;

end


% ====== 在這裡人工挑選 ======
% 看完 1..64 的單張圖，手動填入你選定的 index：
best_8  = [ 1 17 18 19 26 27 32 33 ];   % <--- 例子：請換成你"看圖後"的選擇
worst_8 = [ 3 4 5 6 59 62 63 64 ]; % <--- 例子：請換成你"看圖後"的選擇

% 3) 交卷圖：用 subplot 整理 Best-8、Worst-8

% === 繪製 Best 8 Features ===
figure(2); clf;
sgtitle('Best 8 Features (Least Overlap)', 'FontWeight','bold', 'FontSize', 12);
for i = 1:8
    k = best_8(i);
    subplot(2,4,i);
    hold on; grid on; box on;

    %這四個數（mF, mB, sF, sB）就是兩條 1D 高斯的參數
    sd_Fx = sqrt(var_FG(k));
    sd_Bx = sqrt(var_BG(k));
    u_Fx = u_FG(k); 
    u_Bx = u_BG(k);

    %設定 繪圖的 x 範圍
    lo_bound = min([u_Fx-4*sd_Fx, u_Bx-4*sd_Bx]);
    hi_bound = max([u_Fx+4*sd_Fx, u_Bx+4*sd_Bx]);
    x  = linspace(lo_bound, hi_bound, 300);

    %Calculate the Gaussian PDFs for the current feature
    G_Fx = normpdf(x, u_Fx, sd_Fx);
    G_Bx = normpdf(x, u_Bx, sd_Bx);

    % Different line styles as required
    plot(x, G_Fx, 'r-',  'LineWidth', 1.6); % FG: red solid
    plot(x, G_Bx, 'b--', 'LineWidth', 1.6); % BG: blue dashed

    % --- 子圖標題與外觀 ---
    title(sprintf('Feature #%d', k), 'FontSize', 7, 'FontWeight', 'bold');
    set(gca, 'XTick', [], 'YTick', []);  % 移除刻度
    axis tight;

end

% === 繪製 Worst 8 Features ===
figure(3); clf;
sgtitle('Worst 8 Features (Most Overlap)', 'FontWeight','bold', 'FontSize', 12);
for i = 1:8
    k = worst_8(i);
    subplot(2,4,i);
    hold on; grid on; box on;

    %這四個數（mF, mB, sF, sB）就是兩條 1D 高斯的參數
    sd_Fx = sqrt(var_FG(k));
    sd_Bx = sqrt(var_BG(k));
    u_Fx = u_FG(k); 
    u_Bx = u_BG(k);

    %設定 繪圖的 x 範圍
    lo_bound = min([u_Fx-4*sd_Fx, u_Bx-4*sd_Bx]);
    hi_bound = max([u_Fx+4*sd_Fx, u_Bx+4*sd_Bx]);
    x  = linspace(lo_bound, hi_bound, 300);

    %Calculate the Gaussian PDFs for the current feature
    G_Fx = normpdf(x, u_Fx, sd_Fx);
    G_Bx = normpdf(x, u_Bx, sd_Bx);

    % Different line styles as required
    plot(x, G_Fx, 'r-',  'LineWidth', 1.6); % FG: red solid
    plot(x, G_Bx, 'b--', 'LineWidth', 1.6); % BG: blue dashed

    % --- 子圖標題與外觀 ---
    title(sprintf('Feature #%d', k), 'FontSize', 7, 'FontWeight', 'bold');
    set(gca, 'XTick', [], 'YTick', []);  % 移除刻度
    axis tight;

end

%% c) MAP classification using previous sliding-window framework

% 0) 讀測試影像
test_img = im2double(imread('cheetah.bmp'));
[H, W] = size(test_img);

% 先驗機率取log
logPy_cheetah = log(Py_cheetah);
logPy_grass = log(Py_grass);

inv_FG  = inv(cov_FG);  
inv_BG  = inv(cov_BG);
log_detFG = log(det(cov_FG)); 
log_detBG = log(det(cov_BG));


% 挑好的 best8 / worst8（這裡填你視覺挑選的）
best8  = [1 17 18 19 26 27 32 33];

% 8D 子模型參數
u_FG8   = u_FG(best8).';                  % 8x1
u_BG8   = u_BG(best8).';                  % 8x1
cov_FG8 = cov_FG(best8, best8);
cov_BG8 = cov_BG(best8, best8);
inv_FG8 = inv(cov_FG8);
inv_BG8 = inv(cov_BG8);
log_detFG8 = log(det(cov_FG8) + realmin);
log_detBG8 = log(det(cov_BG8) + realmin);

% (1) 準備輸出遮罩（用你上一版的大小寫到左上角像素）
A64 = false(H, W);   % 64D 結果
A8  = false(H, W);   % best-8 結果

% (2) 準備 zig-zag 對應（沿用你上次的方法）
zig_zag_array = load('Zig-Zag Pattern.txt');  % 內容 0..63
zig_zag_array = zig_zag_array + 1;            % 轉成 1..64
order_coef = zeros(1, 64);
for r = 1:8
    for c = 1:8
        k = zig_zag_array(r,c);               % zigzag 序號 1..64
        order_coef(k) = sub2ind([8,8], r, c); % 第 k 個要取 D(r,c) 的線性索引
    end
end

% (3) 滑動視窗，逐區塊 QDA 打分
for i = 1:(H-7)
    for j = 1:(W-7)
        D = dct2(test_img(i:i+7, j:j+7));   % 8x8
        x64 = D(order_coef).';              % 轉置符號 (') 使其成為 64x1 欄向量

        % 64D
        xuF = x64 - u_FG(:);                % 64x1
        xuB = x64 - u_BG(:);                % 64x1
        gFG = -0.5*(xuF.'*inv_FG*xuF) - 0.5*log_detFG + logPy_cheetah;   % 純量
        gBG = -0.5*(xuB.'*inv_BG*xuB) - 0.5*log_detBG + logPy_grass;   % 純量
        
        % 把結果0或1寫到「中心像素」；若想寫左上角就改成 A(i,j)
        if gFG > gBG
            A64(i, j) = 1;        % 1 = cheetah
        else
            A64(i, j) = 0;        % 0 = grass
        end

        % Best-8D
        x8  = x64(best8);                   % 8x1
        xuF8 = x8 - u_FG8;                  % 8x1
        xuB8 = x8 - u_BG8;                  % 8x1
        gFG8 = -0.5*(xuF8.'*inv_FG8*xuF8) - 0.5*log_detFG8 + logPy_cheetah;
        gBG8 = -0.5*(xuB8.'*inv_BG8*xuB8) - 0.5*log_detBG8 + logPy_grass;

        % 把結果0或1寫到「中心像素」；若想寫左上角就改成 A(i,j)
        if gFG8 > gBG8
            A8(i, j) = 1;        % 1 = cheetah
        else
            A8(i, j) = 0;        % 0 = grass
        end
    end
end

% (4) 顯示結果
figure; 

subplot(1, 2, 1);
imshow(A64);
title('64D Classification Mask');

subplot(1, 2, 2);
imshow(A8);
title('Best 8D Classification Mask');

%% d) 評估錯誤率

% (1) 讀入並準備 ground truth mask
ground_truth = imread('cheetah_mask.bmp');
ground_truth = im2double(ground_truth);     % 轉 double 到 [0,1]
ground_truth = ground_truth > 0.5;          % 二值化 (true/false)

% (2) 定義要評估的區域 (H 和 W 已經在前面定義過了)
rows = 1:(H-7);
cols = 1:(W-7);

% (3) 裁剪 ground truth 遮罩(從 ground_truth 中裁剪出與您的分類結果完全相同大小和位置的區域)
ground_truth_eval = ground_truth(rows, cols);

% (4) 評估 64-D 遮罩 (A64)
A64_eval  = A64(rows, cols);  % A64 已經是 0/1，不需 >0.5
err_rate_64 = mean(A64_eval(:) ~= ground_truth_eval(:));

% (5) 評估 Best-8D 遮罩 (A8)
A8_eval  = A8(rows, cols);    % A8 已經是 0/1
err_rate_8 = mean(A8_eval(:) ~= ground_truth_eval(:));

% (6) 在命令視窗顯示結果
fprintf('Error Rate (64-D Classifier): %.4f (%.2f%%)\n', err_rate_64, 100*err_rate_64);
fprintf('Error Rate (Best-8D Classifier): %.4f (%.2f%%)\n', err_rate_8, 100*err_rate_8);

% (7) 視覺化比較
figure; 
colormap(gray(255)); % 設定顏色表為灰階

subplot(1, 3, 1);
imagesc(ground_truth_eval); 
axis image off; 
title('Ground Truth (Cropped)');

subplot(1, 3, 2); 
imagesc(A64_eval);  
axis image off; 
title(sprintf('64D Prediction (Err: %.2f%%)', 100*err_rate_64));

subplot(1, 3, 3); 
imagesc(A8_eval);  
axis image off; 
title(sprintf('Best 8D Prediction (Err: %.2f%%)', 100*err_rate_8));

