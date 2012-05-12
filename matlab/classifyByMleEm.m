function [globalErrorRate, classErrorVector, postProbMatrix] = classifyByMleEm(dataset, datasetClasses, trainIndices, subclassesIndices)
%CLASSIFYBYMLEEM Mle (class 1) and EM (class 2) classification which
%returns global error, class errors and a posteriori probabilities.
% 
% INPUT:
% dataset (n,j): n samples with j attributes.
% datasetClasses (n,1): classes for the n samples of dataset.
% trainIndices (n,1): logical vector defining train samples.
% subclassesIndices (n,1): subclasses for the n samples of dataset.
%
% OUTPUT:
% globalErrorRate (1,1): value of global error.
% classErrorVector (2,1): error vector for 2 classes.
% postProbMatrix (n,2): posteriori probabilities of n samples for 2
% classes.
%
% {dlf2,dvro}@cin.ufpe.br

% Divide train/test dataset
[trainDataset, trainDatasetClasses, testDataset, testDatasetClasses] = divideDataset(dataset, datasetClasses, trainIndices);

% Calculate pdf of class 1
c1TrainDataset = trainDataset(trainDatasetClasses == 1,:);
c1MU = mean(c1TrainDataset);
c1SIGMA = std(c1TrainDataset);
c1pdf = mvnpdf(testDataset, c1MU, c1SIGMA);

% Calculate pdf of class 2 components
c2TrainDataset = trainDataset(trainDatasetClasses == 2,:);
obj = gmdistribution.fit(c2TrainDataset, 2);
c2MUS    = obj.mu;
c2SIGMAS = obj.Sigma;

c2pdf1 = mvnpdf(testDataset, c2MUS(1,:), c2SIGMAS(:,:,1));
c2pdf2 = mvnpdf(testDataset, c2MUS(2,:), c2SIGMAS(:,:,2));

% Associate each subclass of class 2 to respective prior probability
c21dataset = dataset((subclassesIndices == 2) & trainIndices, :);
medc21 = mean(c21dataset);
subclassMatchIdx = knnsearch(c2MUS, medc21);

wc21 = 1/3;
wc22 = 1/3;
if subclassMatchIdx == 1
    wc21 = 2/3;
else
    wc22 = 2/3;
end

% Combine class 2 components pdfs
c2pdf = wc21 * c2pdf1 + wc22 * c2pdf2;

% Once classes prior probabilities are equal and evidence is common to both
% classes, classification by a posteriori probability is reduced to
% classification by probability density
pdfFinal = c1pdf >= c2pdf;
finalClassification = 2*ones(size(pdfFinal));
finalClassification(pdfFinal) = 1;

% Calculate global error
globalErrorRate = sum(finalClassification ~= testDatasetClasses)/size(testDatasetClasses, 1);

% Calculate class error
classErrorVector = zeros(2,1);
c1Indices = (testDatasetClasses == 1);
c2Indices = (testDatasetClasses == 2);
classErrorVector(1) = sum(finalClassification(c1Indices) ~= 1)/sum(c1Indices);
classErrorVector(2) = sum(finalClassification(c2Indices) ~= 2)/sum(c2Indices);

% Calculate a posteriori probabilities
postProbMatrix = [c1pdf./(c1pdf + c2pdf) c2pdf./(c1pdf + c2pdf)];