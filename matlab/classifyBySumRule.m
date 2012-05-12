function [globalErrorRate, classErrorVector, postProbMatrix] = classifyBySumRule(dataset, datasetClasses, trainIndices, classifiersPostProbMatrix)
%CLASSIFYBYSUMRULE Classification by classifiers combination which returns
%global errors, class errors and a posteriori probabilities based on sum
%rule.
%
% INPUT:
% dataset (m,2): m samples with 2 attributes.
% datasetClasses (m,1): classes for the m samples of dataset.
% trainIndices (m,1): logical vector defining train samples.
% classifiersPostProbMatrix (m,2,s): matrix of posterior probabilities for
% m test samples and 2 classes from s classifiers.
% 
% OUTPUT:
% globalErrorRate (1,1): value of global error.
% classErrorVector (2,1): error vector for 2 classes.
% postProbMatrix (m,2): matrix of posterior probabilities
% combination through sum rule for m test samples and c classes.
%
% {dlf2,dvro}@cin.ufpe.br

% Calculate classes prior probabilities
c1PriorProb = sum(datasetClasses == 1)/numel(datasetClasses);
c2PriorProb = sum(datasetClasses == 2)/numel(datasetClasses);

% Initialize variables
m = size(dataset(~trainIndices),1);
R = size(classifiersPostProbMatrix,3);
postProbMatrix = (1-R)*[c1PriorProb*ones(m,1) c2PriorProb*ones(m,1)];
testDatasetClasses = datasetClasses(~trainIndices);

% Calculate combined a posteriori probabilities
postProbMatrix = postProbMatrix + sum(classifiersPostProbMatrix,3);

% Classify by a posteriori probabilities
[~, combinedClassificationVector] = max(postProbMatrix, [], 2);

% Calculate global errors
globalErrorRate = sum(combinedClassificationVector ~= testDatasetClasses)/size(testDatasetClasses, 1);
        
% Calculate error for each class
logicalC1Position = (testDatasetClasses == 1);
logicalC2Position = (testDatasetClasses == 2);
classErrorVector = [sum(combinedClassificationVector(logicalC1Position) ~= 1)/sum(logicalC1Position); ...
    sum(combinedClassificationVector(logicalC2Position) ~= 2)/sum(logicalC2Position)];