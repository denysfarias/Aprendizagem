% {dlf2,dvro}@cin.ufpe.br

clear; clc;

%% Question 1
% Create dataset (if does not exist)
if (exist('dataset.xls','file') + exist('datasetClassVector.xls','file') + exist('subclassIndexVector.xls','file') ~= 3*2)
    [dataset, datasetClassVector, subclassIndexVector] = createDistributions();
    disp('% Distributions generated.');
else
    % Load dataset (if does exist)
    dataset = xlsread('dataset.xls');
    datasetClassVector = xlsread('datasetClassVector.xls');
    subclassIndexVector = xlsread('subclassIndexVector.xls'); 
end

% Plot distribution
disp('%Plotting generated distributions.');
figure;
gscatter(dataset(:,1), dataset(:,2), subclassIndexVector, 'rbg', 'o', 5, 0);
title('Generated distributions', 'FontWeight','Bold','FontSize',14);
drawnow;

% Initialize variables
nIterations = 50;
globalErrorPerRoundMatrix = zeros(nIterations, 4);
classErrorPerRoundMatrix = zeros(nIterations, 2, 4);
kmeansGlobalErrorPerRoundVector = zeros(nIterations, 1);
kmeansClassErrorPerRoundMatrix = zeros(nIterations, 2);
kmeansAdjustedRandIndexVector = zeros(nIterations,1);
bestCombs = zeros(25,1);

for i = 1:nIterations
    fprintf('[Iteration %i]\n',i);
    % Divide dataset
    [trainIndexVector,testIndexVector] = crossValidationHoldout(datasetClassVector,0.7);
    disp('% Dataset divided.');
    
    % Calculate kmeans errors and adjusted rand index
    [q1pGlobalError, q1pPerClassError, q1adjustedRandIndex] = kmeansErrorCalculation(dataset,datasetClassVector, i == 1);
    disp('% Kmeans calculated.');
    
    %% Question 2.a
    % Classify test dataset by estimating a posteriori probabilities through
    % probability density from MLE (first class) and EM (second class)
    [q21globalErrorRate, q21classErrorVector, q21postProbMatrix] = classifyByMleEm(dataset, datasetClassVector, trainIndexVector, subclassIndexVector);
    disp('% MLE/EM calculated.');
    
    %% Question 2.b
    % Classify test dataset by estimating a posteriori probabilities through
    % parzen window and kernel function
    [q22globalErrorArray, q22classErrorMatrix, q22postProbMatrix, q22h_nRef] = classifyByParzenWindow(dataset, datasetClassVector, trainIndexVector);
    disp('% Parzen Window calculated.');
    
    %% Question 2.c
    % Classify test dataset by estimating a posteriori probabilities through
    % k-nn.
    [q23globalErrorArray, q23classErrorMatrix, q23postProbMatrix] = classifyByKnn(dataset, datasetClassVector, trainIndexVector);
    disp('% KNN calculated.');
    
    % Selecting best classifiers classifications a posteriori probabilities to
    % question 2.d
    classifiersPostProbMatrix = zeros(size(q21postProbMatrix,1), 2, 3);
    classifiersPostProbMatrix(:,:,1) = q21postProbMatrix;
    %[~, bestComb] = min((sum(q22classErrorMatrix,1) + q22globalErrorArray), [], 2);
    %bestCombs(bestComb) = bestCombs(bestComb) + 1;
    bestComb = 8; % h_1 = 10, h_2 = 5
    classifiersPostProbMatrix(:,:,2) = q22postProbMatrix(:,:,bestComb);
    %[~, bestK] = min((sum(q23classErrorMatrix,1) + q23globalErrorArray), [], 2);
    bestK = 2; % k = 3
    classifiersPostProbMatrix(:,:,3) = q23postProbMatrix(:,:,bestK);

    %% Question 2.d
    % Classify test dataset by combining a posteriori probabilities of 2.a, 2.b
    % and 2.c through the sum rule.
    [q24globalErrorRate, q24classErrorVector, q24postProbMatrix] = classifyBySumRule(dataset, datasetClassVector, trainIndexVector, classifiersPostProbMatrix);
    disp('% Classifiers combination calculated.');
    
    % Preparing global error matrix for comparison
    globalErrorPerRoundMatrix(i,1) = q21globalErrorRate;
    globalErrorPerRoundMatrix(i,2) = q22globalErrorArray(bestComb);
    globalErrorPerRoundMatrix(i,3) = q23globalErrorArray(bestK);
    globalErrorPerRoundMatrix(i,4) = q24globalErrorRate;
    
    % Collecting results to present
    classErrorPerRoundMatrix(i,:,1) = q21classErrorVector;
    classErrorPerRoundMatrix(i,:,2) = q22classErrorMatrix(:,bestComb);
    classErrorPerRoundMatrix(i,:,3) = q23classErrorMatrix(:,bestK);
    classErrorPerRoundMatrix(i,:,4) = q24classErrorVector;
    
    kmeansGlobalErrorPerRoundVector(i) = q1pGlobalError;
    kmeansClassErrorPerRoundMatrix(i,:) = q1pPerClassError;
    kmeansAdjustedRandIndexVector(i) = q1adjustedRandIndex;
end

%% Evaluate and compare classifiers
[hVector, pValueVector, combMatrix] = compareClassifiers(globalErrorPerRoundMatrix);

%% Format results
[st1, msg1] = xlswrite('kmeansGlobalErrorPerRoundVector.xls', kmeansGlobalErrorPerRoundVector);
[st2, msg2] = xlswrite('kmeansClassErrorPerRoundMatrix.xls', kmeansClassErrorPerRoundMatrix);
[st3, msg3] = xlswrite('globalErrorPerRoundMatrix.xls', globalErrorPerRoundMatrix);
[st4, msg4] = xlswrite('classErrorPerRoundMatrixClassifier1.xls', classErrorPerRoundMatrix(:,:,1));
[st5, msg5] = xlswrite('classErrorPerRoundMatrixClassifier2.xls', classErrorPerRoundMatrix(:,:,2));
[st6, msg6] = xlswrite('classErrorPerRoundMatrixClassifier3.xls', classErrorPerRoundMatrix(:,:,3));
[st7, msg7] = xlswrite('classErrorPerRoundMatrixClassifier4.xls', classErrorPerRoundMatrix(:,:,4));
[st8, msg8] = xlswrite('kmeansAdjustedRandIndexVector.xls', kmeansAdjustedRandIndexVector);
[st9, msg9] = xlswrite('hVector.xls', hVector);
[st10, msg10] = xlswrite('pValueVector.xls', pValueVector);
[st11, msg11] = xlswrite('combMatrix.xls', combMatrix);