%% Implement generation of different traininig/testing sets with the same distribution
% TODO

%% Question 1
% Create dataset
[dataset, datasetClasses, trainIndices, testIndices, subclassesIndices] = createDataset(0.7);

% Calculate kmeans errors and adjusted rand index
[q1pPerClassError, q1pGlobalError, q1adjustedRandIndex] = kmeansErrorCalculation(dataset,datasetClasses);

%% Question 2.a
% Classify test dataset by estimating a posteriori probabilities through
% probability density from MLE (first class) and EM (second class)
[q21globalErrorRate, q21classErrorVector, q21postProbMatrix] = classifyByMleEm(dataset, datasetClasses, trainIndices, subclassesIndices);

%% Question 2.b
% Classify test dataset by estimating a posteriori probabilities through
% parzen window and kernel function
[q22globalErrorArray, q22classErrorMatrix, q22postProbMatrix, q22h_nRef] = classifyByParzenWindow(dataset, datasetClasses, trainIndices);

%% Question 2.c
% Classify test dataset by estimating a posteriori probabilities through
% k-nn.
[q23globalErrorArray, q23classErrorMatrix, q23postProbMatrix] = classifyByKnn(dataset, datasetClasses, trainIndices);

% Selecting best classifiers classifications a posteriori probabilities to
% question 2.d
classifiersPostProbMatrix = zeros(size(q21postProbMatrix,1), 2, 3);
classifiersPostProbMatrix(:,:,1) = q21postProbMatrix;
[~, bestComb] = min((sum(q22classErrorMatrix,1) + q22globalErrorArray), [], 2);
classifiersPostProbMatrix(:,:,2) = q22postProbMatrix(:,:,bestComb);
[~, bestK] = min((sum(q23classErrorMatrix,1) + q23globalErrorArray), [], 2);
classifiersPostProbMatrix(:,:,3) = q23postProbMatrix(:,:,bestK);

%% Question 2.d
% Classify test dataset by combining a posteriori probabilities of 2.a, 2.b
% and 2.c through the sum rule.
[q24globalErrorRate, q24classErrorVector, q24postProbMatrix] = classifyBySumRule(dataset, datasetClasses, trainIndices, classifiersPostProbMatrix);

%% Evaluate and compare classifiers
% TODO

%% Format results
% TODO