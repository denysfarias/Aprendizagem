%% Question 1
% Create dataset
[dataset, datasetClasses, trainIndices, testIndices, subclassesIndices] = createDataset(0.7);

% Calculate kmeans errors and adjusted rand index
[pPerClassError, pGlobalError, adjustedRandIndex] = kmeansErrorCalculation(dataset,datasetClasses);

% Dividing train/test datasets
trainDataset = dataset(trainIndices,:);
trainDatasetClasses = datasetClasses(trainIndices);
testDataset = dataset(testIndices,:);
testDatasetClasses = datasetClasses(testIndices);

%% Question 2.a
% Classify test dataset by estimating a posteriori probabilities through
% probability density from MLE (first class) and EM (second class)
globalErrorRate = classifyByMleEm(dataset, trainDataset, trainDatasetClasses, testDataset, testDatasetClasses, subclassesIndices, trainIndices);

%% Question 2.b
% Classify test dataset by estimating a posteriori probabilities through
% parzen window and kernel function