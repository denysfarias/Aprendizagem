function [globalErrorArray, classErrorMatrix, posterioriProbMatrix, h_nRef] = ...
    classifyByParzenWindow(dataset, datasetClasses, trainIndices)
%CLASSIFYBYPARZENWINDOW Classification based on Parzen Window (smoothing density
%estimation), which returns returns global errors, class errors and a
%posteriori probabilities for each h1 and h2 widths in h_nMatrix (bult-in
%set) used in classification.
% 
% INPUT:
% dataset (m,2): m samples with 2 attributes.
% datasetClasses (m,1): classes for the m samples of dataset.
% trainIndices (m,1): logical vector defining train samples.
%
% OUTPUT:
% globalErrorArray (1,nComb): array of nComb global errors of each
% h_nRef element.
% classErrorMatrix (c,nComb): error matrix for c classes and for each
% h_nRef element.
% posterioriProbMatrix (m,c,nComb): posteriori probabilities of m test
% samples for c classes for each h_nRef element.
% % h_nRef (nComb,2): combinations of h_1 and h_2.
%
% {dlf2,dvro}@cin.ufpe.br

% Create train/test dataset
[trainDataset, trainDatasetClasses, testDataset, testDatasetClasses] = divideDataset(dataset, datasetClasses, trainIndices);

% Divide trainDataset by class
c1TrainDataset = trainDataset(trainDatasetClasses == 1,:);
c2TrainDataset = trainDataset(trainDatasetClasses == 2,:);

% Initialize parameters
h_nMatrix = [1,5,10; 1,5,10];
m = size(testDataset,1);
nComb = size(h_nMatrix,2)^2;
globalErrorArray = zeros(1,nComb);
classErrorMatrix = zeros(2,nComb);
posterioriProbMatrix = zeros(m,2,nComb);
h_nRef = zeros(nComb,2);
inComb = 1;

for h_1 = h_nMatrix(1,:)
    for h_2 = h_nMatrix(2,:)
        h_nRef(inComb,:) = [h_1, h_2];
        
        % Initialize variables
        c1pdf = zeros(m,1);
        c2pdf = zeros(m,1);
        
        % Calculate densities per class
        for i = 1:m
            c1pdf(i) = biVariateParzenWindow(testDataset(i,:),c1TrainDataset, [h_1, h_2]);
            c2pdf(i) = biVariateParzenWindow(testDataset(i,:),c2TrainDataset, [h_1, h_2]);
        end
        
        % Classify test samples for each pair of h_n values
        pdfFinal = c1pdf >= c2pdf;
        finalClassification = 2*ones(size(pdfFinal));
        finalClassification(pdfFinal) = 1;
        globalErrorArray(1,inComb) = sum(finalClassification ~= testDatasetClasses)/size(testDatasetClasses, 1);
        
        % Calculate error for each class
        logicalC1Position = (testDatasetClasses == 1);
        logicalC2Position = (testDatasetClasses == 2);
        classErrorMatrix(:,inComb) = [sum(finalClassification(logicalC1Position) ~= 1)/sum(logicalC1Position); ...
            sum(finalClassification(logicalC2Position) ~= 2)/sum(logicalC2Position)];
        
        % Calculate a posteriori probabilities
        posterioriProbMatrix(:,:,inComb) = [c1pdf./(c1pdf + c2pdf) c2pdf./(c1pdf + c2pdf)];

        inComb = inComb+1;
    end
end