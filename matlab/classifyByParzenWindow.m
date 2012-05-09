function globalErrors = classifyByParzenWindow(dataset, datasetClasses, trainIndices)

[trainDataset, trainDatasetClasses, testDataset, testDatasetClasses] = divideDataset(dataset, datasetClasses, trainIndices);

c1TrainDataset = trainDataset(trainDatasetClasses == 1,:);
c2TrainDataset = trainDataset(trainDatasetClasses == 2,:);

globalErrors = zeros(3*3,3);
n = size(trainDataset,1);
m = size(testDataset,1);
k = 1;
for h1 = [1, 5, 10]
    for h2 = [1, 5, 10]
        c1pdf = zeros(m,1);
        c2pdf = zeros(m,1);
        for i = 1:m
            c1pdf(i) = biVariateParzenWindow(testDataset(i,:),c1TrainDataset, [h1; h2]);
            c2pdf(i) = biVariateParzenWindow(testDataset(i,:),c2TrainDataset, [h1; h2]);
        end
        pdfFinal = c1pdf >= c2pdf;
        finalClassification = 2*ones(size(pdfFinal));
        finalClassification(pdfFinal) = 1;
        globalErrors(k,:) = [h1, h2, (1 - sum(finalClassification == testDatasetClasses)/size(testDatasetClasses, 1))];
        k = k+1;
    end
end