function [dataset, datasetClassVector, subclassIndexVector] = createDistributions()
%%CREATEDISTRIBUTIONS Creates 3 normal distributions with 2 classes.
%
% OUTPUT:
% dataset (m,2): set of m samples derived from 3 normal distributions and 2
% classes.
% datasetClassVector (m,1): respective classes of dataset.
% subclassIndexVector (m,1): respective subclasses of dataset.
%
% {dlf2,dvro}@cin.ufpe.br

% Create distribution for class 1
muC1 = [0 0];
sigmaC1 = [2^2 1.7; 1.7 1^2];
nExamplesC1 = 150;
datasetC1 = createDistribution(muC1, sigmaC1, nExamplesC1);

% Create distribution for class 2
muC2 = [0 3];
sigmaC2 = [0.5^2 0; 0 0.5^2];
nExamplesC2 = 100;
datasetC2 = createDistribution(muC2, sigmaC2, nExamplesC2);

% Create distribution for class 3
muC3 = [4 3];
sigmaC3 = [2^2 -1.7; -1.7 1^2];
nExamplesC3 = 50;
datasetC3 = createDistribution(muC3, sigmaC3, nExamplesC3);

subclassIndexVector = [ones(size(datasetC1,1),1); 2*ones(size(datasetC2,1),1); 3*ones(size(datasetC3,1),1)];

% Class 3 as subclass of Class 2
datasetC2 = [datasetC2; datasetC3];

% Create total dataset
dataset = [datasetC1; datasetC2];
datasetClassVector = [ones(size(datasetC1,1),1); 2*ones(size(datasetC2,1),1)];

% Save dataset, classes and subclasses
[status1, msg1] = xlswrite('dataset.xls', dataset);
[status2, msg2] = xlswrite('datasetClassVector.xls', datasetClassVector);
[status3, msg3] = xlswrite('subclassIndexVector.xls', subclassIndexVector);