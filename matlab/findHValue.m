dataset = xlsread('dataset.xls');
datasetClassVector = xlsread('datasetClassVector.xls');
subclassIndexVector = xlsread('subclassIndexVector.xls');

[trainIndexVector,testIndexVector] = crossValidationHoldout(datasetClassVector,0.7);

% Create train/test dataset
[trainDataset, trainDatasetClasses, testDataset, testDatasetClasses] = divideDataset(dataset, datasetClassVector, trainIndexVector);

% Divide trainDataset by class
c1TrainDataset = trainDataset(trainDatasetClasses == 1,:);
c2TrainDataset = trainDataset(trainDatasetClasses == 2,:);

% Initialize parameters
h_1Array = 0.1:0.1:10;
h_2Array = 0.1:0.1:10;
m = size(testDataset,1);
globalErrorMatrixPerHValue = zeros(numel(h_1Array),numel(h_2Array));

ih1 = 1;
for h_1 = h_1Array
    ih2 = 1;
    for h_2 = h_2Array
        fprintf('h_1: %g - h_2: %g\n', h_1, h_2);
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
        globalErrorMatrixPerHValue(ih1,ih2) = sum(finalClassification ~= testDatasetClasses)/size(testDatasetClasses, 1);
        ih2 = ih2+1;
    end
    ih1 = ih1 + 1;
end
[minValue, minIndex] = min(globalErrorMatrixPerHValue(:));
[st, msg] = xlswrite('globalErrorPerHValue.xls', globalErrorMatrixPerHValue);
fprintf('End!');