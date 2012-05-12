%% Question 1
% Create dataset
[dataset, datasetClasses, trainIndices, testIndices, subclassesIndices] = createDataset(0.7);

% Calculate kmeans errors and adjusted rand index
[q1pPerClassError, q1pGlobalError, q1adjustedRandIndex] = kmeansErrorCalculation(dataset,datasetClasses);

%% Question 2.a
% Classify test dataset by estimating a posteriori probabilities through
% probability density from MLE (first class) and EM (second class)
[q21globalErrorRate, q21classErrorVector, q21posterioriProbabilityMatrix] = classifyByMleEm(dataset, datasetClasses, trainIndices, subclassesIndices);

%% Question 2.b
% Classify test dataset by estimating a posteriori probabilities through
% parzen window and kernel function
[q22globalErrorArray, q22classErrorMatrix, q22posterioriProbMatrix, q22h_nRef] = classifyByParzenWindow(dataset, datasetClasses, trainIndices);

%% Question 2.c
% Classify test dataset by estimating a posteriori probabilities through
% k-nn.

[q23globalErrorArray, q23classErrorMatrix, q23posterioriProbMatrix] = classifyByKnn(dataset, datasetClasses, trainIndices);