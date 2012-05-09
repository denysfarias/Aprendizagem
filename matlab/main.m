%% Question 1
% Create dataset
[dataset, datasetClasses, trainIndices, testIndices, subclassesIndices] = createDataset(0.7);

% Calculate kmeans errors and adjusted rand index
[pPerClassError, pGlobalError, adjustedRandIndex] = kmeansErrorCalculation(dataset,datasetClasses);

trainDataset = dataset(trainIndices,:);
trainDatasetClasses = datasetClasses(trainIndices);
testDataset = dataset(testIndices,:);
testDatasetClasses = datasetClasses(testIndices);

%% Question 2.a
c1TrainDataset = trainDataset(trainDatasetClasses == 1,:);
% c1TestDataset = testDataset(testDatasetClasses == 1,:);
c1MU = mean(c1TrainDataset);
c1SIGMA = std(c1TrainDataset);
c1pdf = mvnpdf(testDataset, c1MU, c1SIGMA);

c2TrainDataset = trainDataset(trainDatasetClasses == 2,:);
% c2TestDataset = testDataset(testDatasetClasses == 2,:);
obj = gmdistribution.fit(c2TrainDataset, 2);
c2MUS    = obj.mu;
c2SIGMAS = obj.Sigma;

c2pdf1 = mvnpdf(testDataset, c2MUS(1,:), c2SIGMAS(:,:,1));
c2pdf2 = mvnpdf(testDataset, c2MUS(2,:), c2SIGMAS(:,:,2));

c21dataset = dataset((subclassesIndices == 2) & trainIndices, :);
medc21 = mean(c21dataset);
subclassMatchIdx = knnsearch(c2MUS, medc21);

wc21 = 1/3;
wc22 = 1/3;
if subclassMatchIdx == 1
    wc21 = 2/3;
else
    ec22 = 2/3;
end

c2pdf = wc21 * c2pdf1 + wc22 * c2pdf2;

% now we have c1pdf and c2pdf
pdfFinal = c1pdf >= c2pdf;
finalClassification = 2*ones(size(pdfFinal));
finalClassification(pdfFinal) = 1;

globalErrorRate = (1 - sum(finalClassification == testDatasetClasses)/size(testDatasetClasses, 1));