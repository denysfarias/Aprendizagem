function [dataset, datasetClasses, trainIndices, testIndices] = createDataset(trainFraction)

% Create distribution for class 1
muC1 = [0 0];
sigmaC1 = [2^2 1.7; 1.7 1^2];
nExamplesC1 = 150;
datasetC1 = generateDistribution(muC1, sigmaC1, nExamplesC1);

% Create distribution for class 2
muC2 = [0 3];
sigmaC2 = [0.5^2 0; 0 0.5^2];
nExamplesC2 = 100;
datasetC2 = generateDistribution(muC2, sigmaC2, nExamplesC2);

% Create distribution for class 3
muC3 = [4 3];
sigmaC3 = [2^2 -1.7; -1.7 1^2];
nExamplesC3 = 50;
datasetC3 = generateDistribution(muC3, sigmaC3, nExamplesC3);

% Create total dataset
dataset = [datasetC1; datasetC2; datasetC3];
datasetClasses = [ones(size(datasetC1,1),1); 2*ones(size(datasetC2,1),1); 3*ones(size(datasetC3,1),1)];

% Divide train/test sets
[trainIndices, testIndices] = crossvalind('HoldOut', datasetClasses, 1 - trainFraction);