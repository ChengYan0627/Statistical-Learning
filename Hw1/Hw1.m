%% a) 
load('TrainingSamplesDCT_8.mat')
N_FG = size(TrainsampleDCT_FG);
N_BG = size(TrainsampleDCT_BG);
N = N_FG(1) + N_BG(1);
Py_cheetah = N_FG(1)/ N;
Py_grass = N_BG(1)/ N;
Test = 1-Py_cheetah

%% b)
data_FG = TrainsampleDCT_FG;   % 每列 = 一筆樣本 (1×64)
numSamples = size(data_FG, 1); % 樣本數
X_FG = zeros(numSamples, 1);   % 預先建立一個空的結果向量
N_block_FG = N_FG(1);

data_BG = TrainsampleDCT_BG;   % 每列 = 一筆樣本 (1×64)
numSamples = size(data_BG, 1); % 樣本數
X_BG = zeros(numSamples, 1);   % 預先建立一個空的結果向量
N_block_BG = N_BG(1);


for i = 1:N_block_FG
    % 取出第 i 筆樣本的 64 維向量
    vec = data_FG(i, :);

    % 計算每個係數的絕對值（因為要比較能量大小）
    absVec = abs(vec);

    % 找出前兩大的索引,[B,I] = sort(___) 还会为上述任意语法返回一个索引向量的集合
    [B, sortedIndex] = sort(absVec, 'descend');

    % 第二大的係數位置 = sortedIndex(2),histogram 指令會自動把相同的值相加，幫你統計每個值出現的次數
    X_FG(i) = sortedIndex(2);
end


for i = 1:N_block_BG
    % 取出第 i 筆樣本的 64 維向量
    vec = data_BG(i, :);

    % 計算每個係數的絕對值（因為要比較能量大小）
    absVec = abs(vec);

    % 找出前兩大的索引,[B,I] = sort(___) 还会为上述任意语法返回一个索引向量的集合
    [B, sortedIndex] = sort(absVec, 'descend');

    % 第二大的係數位置 = sortedIndex(2),histogram 指令會自動把相同的值相加，幫你統計每個值出現的次數
    X_BG(i) = sortedIndex(2);
end

% 直方圖 -> 機率（總和=1）
edges = 0.5:1:64.5; bins = 1:64;
P_X_given_FG = histcounts(X_FG, edges, 'Normalization','probability');
P_X_given_BG = histcounts(X_BG, edges, 'Normalization','probability');

% (可視化，非必需)
figure; bar(bins, P_X_given_FG); title('P(X|cheetah)'); xlim([1 64]);
figure; bar(bins, P_X_given_BG); title('P(X|grass)');   xlim([1 64]);

%% c)
% 0) 讀取測試影像，轉 double
test_img = im2double(imread('cheetah.bmp')); % 轉成 double
[H, W] = size(test_img);                     % 取得影像尺寸

% (1) 準備輸出遮罩 A（1=cheetah, 0=grass）
A = zeros(H, W);

% (2) 讀 zig-zag 對應（要 +1 才是 MATLAB index從1開始）/ zig_zag_array(r, c) 就是第 r 列、第 c 行的 zig-zag 編號
zig_zag_array = load('Zig-Zag Pattern.txt');   % 內容是 0..63
zig_zag_array = zig_zag_array + 1;             % 轉成 1..64


% 把第 k 個 zig-zag 位置，對應到 DCT 矩陣中的「行列位置 (r, c)」
order_coef = zeros(1, 64);
for r = 1:8
    for c = 1:8
        k = zig_zag_array(r,c);               % zig-zag 序號 1..64
        order_coef(k) = sub2ind([8,8], r, c); % 把第 k 個要抓哪個 D(r,c) 的線性索引
    end
end

% (3) 滑動視窗（stride=1）：對每個 8×8 區塊執行 DCT → zig-zag → 取 feature X →（可選）分類
for i = 1:(H-7)
    for j = 1:(W-7)
        % 4.1 擷取 8×8 區塊
        block = test_img(i:i+7, j:j+7);

        % 4.2 DCT：空間域 → 頻率域
        DCT = dct2(block);             % 8x8

        % 4.3 依 zig-zag 順序壓成 1x64 向量
        % 用事先算好的 order_coef 直接抓：vec(k) = D(對應(r,c))
        vec = DCT(order_coef);         % 1x64，已是 zig-zag 順序

        % 4.4 取絕對值後排序找「第二大」的 index（真正第2大，不假設 DC 最大）
        absVec = abs(vec);
        [~, sortedIndex] = sort(absVec, 'descend');
        X = sortedIndex(2);   % 第二大的係數位置

        % ===== 到這裡為止，就是你說的整套「算 DCT → zig-zag → 壓扁 → sort 找 feature」 =====

        % 使用特徵 X 來索引，取得對應的概似率值
        likelihood_FG = P_X_given_FG(X);
        likelihood_BG = P_X_given_BG(X);

        % 4.5（可選）MAP 比較：用 (a)(b) 的機率做分類
        score_FG = likelihood_FG * Py_cheetah;
        score_BG = likelihood_BG * Py_grass;

        % 4.6 把結果0或1寫到「中心像素」；若想寫左上角就改成 A(i,j)
        if score_FG > score_BG
            A(i, j) = 1;        % 1 = cheetah
        else
            A(i, j) = 0;        % 0 = grass
        end
    end
end

% (5) 顯示結果
figure; imagesc(A); axis image off; colormap(gray(255));
title('Segmentation mask');

%% d) Evaluate error rate against ground truth
% 讀入 ground truth mask（0=grass, 255=cheetah 或 0/1）
ground_truth = imread('cheetah_mask.bmp');
ground_truth = im2double(ground_truth);     % 轉 double 到 [0,1]
ground_truth = ground_truth > 0.5;          % 二值化（>0.5 當 1)

% 與 (c) 對齊
[H, W] = size(A);
rows = 1:(H-7);
cols = 1:(W-7);

A_eval  = A(rows, cols) > 0.5;      % 保險二值化
ground_truth_eval = ground_truth(rows, cols) > 0.5;

% 總錯誤率（0/1 loss）
err_rate = mean(A_eval(:) ~= ground_truth_eval(:));

% 顯示結果
fprintf('Error rate = %.4f (%.2f%%)\n', err_rate, 100*err_rate);

% 視覺化（可選）：GT、Pred、差異
figure; 
subplot(1,2,1); imagesc(ground_truth_eval); axis image off; colormap(gray(255)); title('Ground Truth');
subplot(1,2,2); imagesc(A_eval);  axis image off; colormap(gray(255)); title(sprintf('Prediction (error rate = %.2f%%)', 100*err_rate));
