function [globalErrorArray, classErrorMatrix, postProbMatrix] = classifyByKnn(dataset, datasetClasses, trainIndices)
%CLASSIFYBYKNN Knn classification which returns global errors, class errors and a posteriori probabilities for
% each k in kArray (bult-in set).
% 
% INPUT:
% dataset (n,j): n samples with j attributes.
% datasetClasses (n,1): classes for the n samples of dataset.
% trainIndices (n,1): logical vector defining train samples.
%
% OUTPUT:
% globalErrorArray (1,k): array of k global errors.
% classErrorMatrix (c,k): error matrix for c classes and k neighbors.
% postProbMatrix (n,c,k): posteriori probabilities of n samples for c
% classes for each k.
%
% {dlf2,dvro}@cin.ufpe.br

% Create train/test dataset
[trainDataset, trainDatasetClasses, testDataset, testDatasetClasses] = divideDataset(dataset, datasetClasses, trainIndices);

% Define parameters
kArray = [1, 3, 5, 7];
nTest = size(testDataset,1);
nClassMax = max([trainDatasetClasses; testDatasetClasses]);

% Allocate output matrices
globalErrorArray = zeros(1, numel(kArray));
classErrorMatrix = zeros(nClassMax, numel(kArray));
postProbMatrix = zeros(nTest, nClassMax, numel(kArray));

% Classify samples for every k in kArray
i = 1;
for k = kArray
    neighborIndices = knnsearch(trainDataset, testDataset, 'K', k);
    neighborClassification = trainDatasetClasses(neighborIndices);
    for l = 1:nClassMax
        % Calculate a posteriori probabilities
        postProbMatrix(:,l,k) = sum(neighborClassification == l,2)/k;
    end
    [~, testClassification] = max(postProbMatrix(:,:,k),[],2);
    
    % Calculate global error for each k
    globalErrorArray(i) = sum(testClassification ~= testDatasetClasses)/size(testDatasetClasses, 1);
    for l = 1:nClassMax
        % Calculate error for each class
        logicalClassPosition = (testDatasetClasses == l);
        classErrorMatrix(l,i) = sum(testClassification(logicalClassPosition) ~= l)/sum(logicalClassPosition);
    end
    i=i+1;
end