%% Question 1
% Create dataset
[dataset, datasetClassVector, subclassIndexVector] = createDistributions();
disp('% Distributions generated.');

nIterations = 10;
globalErrorPerRoundMatrix = zeros(nIterations, 5);
for i = 1:nIterations
    itText = sprintf('% [Iteration %i]\n',i);
    disp(itText);
    % Divide dataset
    [trainIndexVector,testIndexVector] = crossValidationHoldout(datasetClassVector,0.7);
    disp('% Dataset divided.');
    
    % Calculate kmeans errors and adjusted rand index
    [q1pPerClassError, q1pGlobalError, q1adjustedRandIndex] = kmeansErrorCalculation(dataset,datasetClassVector, i == 1);
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
    [~, bestComb] = min((sum(q22classErrorMatrix,1) + q22globalErrorArray), [], 2);
    classifiersPostProbMatrix(:,:,2) = q22postProbMatrix(:,:,bestComb);
    [~, bestK] = min((sum(q23classErrorMatrix,1) + q23globalErrorArray), [], 2);
    classifiersPostProbMatrix(:,:,3) = q23postProbMatrix(:,:,bestK);

    %% Question 2.d
    % Classify test dataset by combining a posteriori probabilities of 2.a, 2.b
    % and 2.c through the sum rule.
    [q24globalErrorRate, q24classErrorVector, q24postProbMatrix] = classifyBySumRule(dataset, datasetClassVector, trainIndexVector, classifiersPostProbMatrix);
    disp('% Classifiers combination calculated.');
    
    % Preparing global error matrix for comparison
    globalErrorPerRoundMatrix(i,1) = q1pGlobalError;
    globalErrorPerRoundMatrix(i,2) = q21globalErrorRate;
    globalErrorPerRoundMatrix(i,3) = q22globalErrorArray(bestComb);
    globalErrorPerRoundMatrix(i,4) = q23globalErrorArray(bestK);
    globalErrorPerRoundMatrix(i,5) = q24globalErrorRate;
end

%% Evaluate and compare classifiers
% TODO

%% Format results
% TODO