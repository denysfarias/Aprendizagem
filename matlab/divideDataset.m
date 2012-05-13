function [trainDataset, trainDatasetClasses, testDataset, testDatasetClasses] = divideDataset(dataset, datasetClasses, trainIndices)
%DIVIDEDATASET Divides dataset according to train/test indices.
% 
% INPUT:
% dataset (n,j): n samples with j attributes.
% datasetClasses (n,1): classes for the n samples of dataset.
% trainIndices (n,1): logical vector defining train samples.
%
% OUTPUT:
% trainDataset (n-m,j): n-m train samples with j attributes.
% trainDatasetClasses (n-m,1): classes for the n-m train samples.
% testDataset (m,j): m test samples with j attributes.
% testDatasetClasses (m,1): classes for the m test samples.
%
% {dlf2,dvro}@cin.ufpe.br

trainDataset = dataset(trainIndices,:);
trainDatasetClasses = datasetClasses(trainIndices);
testDataset = dataset(~trainIndices,:);
testDatasetClasses = datasetClasses(~trainIndices);