function [trainDataset, trainDatasetClasses, testDataset, testDatasetClasses] = divideDataset(dataset, datasetClasses, trainIndices)

trainDataset = dataset(trainIndices,:);
trainDatasetClasses = datasetClasses(trainIndices);
testDataset = dataset(~trainIndices,:);
testDatasetClasses = datasetClasses(~trainIndices);