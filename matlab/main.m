%% Question 1
% Create dataset
[dataset, datasetClasses, trainIndices, testIndices] = createDataset(0.7);

% Calculate kmeans errors and adjusted rand index
[pPerClassError, pGlobalError, adjustedRandIndex] = kmeansErrorCalculation(dataset,datasetClasses);

%% Question 2
