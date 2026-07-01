%% ============================================================
%  EUROPEAN BANKING SYSTEM - CLUSTERING ANALYSIS
%  Methods: K-Means, Fuzzy C-Means, Self-Organizing Maps (SOM)
%  Author: [Your Name]
% ============================================================

clc;
clear;
close all;
format compact;

%% ============================================================
%  1. DATA LOADING & PREPROCESSING
% ============================================================

% Load dataset
data = readtable('lab2.xlsx');

% Extract metadata
countries = data{:,1};                 % Country names
X = data{:,2:end};                     % Feature matrix
feature_names = data.Properties.VariableNames(2:end);

% Normalize features (important for distance-based methods)
X_norm = normalize(X, 'zscore');

fprintf('\nData loaded: %d countries, %d features\n', size(X,1), size(X,2));

%% ============================================================
%  2. DIMENSIONALITY REDUCTION (PCA FOR VISUALIZATION)
% ============================================================

[~, score] = pca(X_norm);

%% ============================================================
%  3. K-MEANS CLUSTERING ANALYSIS
% ============================================================

fprintf('\n================ K-MEANS CLUSTERING ================\n');

k_values = [3 4 5];
silhouette_scores = zeros(size(k_values));
davies_bouldin_scores = zeros(size(k_values));

figure('Name','K-Means Clustering Results','NumberTitle','off');

for i = 1:length(k_values)

    k = k_values(i);

    % Run K-Means
    [cluster_idx, cluster_centers] = kmeans(X_norm, k, ...
        'Replicates', 10, ...
        'Display', 'off');

    % Evaluate clustering quality
    silhouette_scores(i) = mean(silhouette(X_norm, cluster_idx));
    eva = evalclusters(X_norm, cluster_idx, 'DaviesBouldin');
    davies_bouldin_scores(i) = eva.CriterionValues;

    fprintf('\n--- K-Means: %d clusters ---\n', k);

    % Cluster summary (compact)
    for c = 1:k
        n_members = sum(cluster_idx == c);
        fprintf('Cluster %d: %d countries\n', c, n_members);
    end

    % Visualization
    subplot(3,1,i);
    gscatter(score(:,1), score(:,2), cluster_idx);
    title(sprintf('K-Means Clustering (k = %d)', k));
    xlabel('PC1'); ylabel('PC2');
    grid on;

end

% Select best K
[~, best_k_idx] = max(silhouette_scores);
best_k = k_values(best_k_idx);

fprintf('\nBest K (K-Means): %d\n', best_k);

%% ============================================================
%  4. FUZZY C-MEANS CLUSTERING
% ============================================================

fprintf('\n================ FUZZY C-MEANS ================\n');

c_values = [3 4 5];
partition_coeff = zeros(size(c_values));
partition_entropy = zeros(size(c_values));

figure('Name','Fuzzy C-Means Results','NumberTitle','off');

for i = 1:length(c_values)

    c = c_values(i);

    % Run FCM
    [centers, U] = fcm(X_norm, c);
    [~, cluster_idx] = max(U);

    % Metrics
    partition_coeff(i) = sum(U(:).^2) / size(X_norm,1);
    partition_entropy(i) = -sum(sum(U .* log(U + eps))) / size(X_norm,1);

    fprintf('\n--- Fuzzy C-Means: %d clusters ---\n', c);

    for cl = 1:c
        n_members = sum(cluster_idx == cl);
        fprintf('Cluster %d: %d countries\n', cl, n_members);
    end

    % Visualization
    subplot(3,1,i);
    gscatter(score(:,1), score(:,2), cluster_idx);
    title(sprintf('Fuzzy C-Means (c = %d)', c));
    xlabel('PC1'); ylabel('PC2');
    grid on;

end

% Best fuzzy model
[~, best_c_idx] = max(partition_coeff);
best_c = c_values(best_c_idx);

fprintf('\nBest C (FCM): %d\n', best_c);

%% ============================================================
%  5. SELF-ORGANIZING MAP (SOM) CLUSTERING
% ============================================================

fprintf('\n================ SOM CLUSTERING ================\n');

grid_sizes = {[3 3], [4 4], [5 5]};
quantization_errors = zeros(1, length(grid_sizes));

for i = 1:length(grid_sizes)

    grid = grid_sizes{i};

    % Create SOM
    net = selforgmap(grid);
    net.trainParam.epochs = 300;

    % Train network
    net = train(net, X_norm');

    % Assign clusters
    outputs = net(X_norm');
    cluster_idx = vec2ind(outputs);

    % Quantization error
    quantization_errors(i) = mean(vecnorm(X_norm' - net.IW{1}(cluster_idx,:)', 2));

    fprintf('Grid [%dx%d] QE = %.4f\n', grid(1), grid(2), quantization_errors(i));

end

% Best SOM
[best_qe, best_idx] = min(quantization_errors);
best_grid = grid_sizes{best_idx};

fprintf('\nBest SOM grid: [%d x %d], QE = %.4f\n', ...
    best_grid(1), best_grid(2), best_qe);

%% ============================================================
%  6. FINAL COMPARISON TABLE
% ============================================================

results_table = {
    'K-Means', best_k, max(silhouette_scores), min(davies_bouldin_scores), NaN, NaN, NaN;
    'Fuzzy C-Means', best_c, NaN, NaN, max(partition_coeff), min(partition_entropy), NaN;
    'SOM', NaN, NaN, NaN, NaN, NaN, best_qe
};

T_summary = cell2table(results_table, ...
    'VariableNames', {'Method','BestClusters','Silhouette','DaviesBouldin','PC','PE','QuantError'});

disp('================ FINAL COMPARISON ================');
disp(T_summary);

%% ============================================================
%  END OF SCRIPT
% ============================================================