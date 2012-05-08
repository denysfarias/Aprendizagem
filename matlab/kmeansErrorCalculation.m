% Create dataset
[dataset, datasetClasses, trainIndices, testIndices] = createDataset(0.7);

% Apply k-means 100 times for 2 clusters
[idx, centroids] = kmeans(dataset, 2, 'replicates', 100);

% Plot clusters
disp('%Plotando os clusters identificados.');
figure;
gscatter(dataset(:,1), dataset(:,2), idx, 'rbg', 'o', 5, 0);
title('Clusters', 'FontWeight','Bold','FontSize',14);
drawnow;

% Define classes centroids
classesCentroid = [mean(dataset(datasetClasses == 1, :));
    mean(dataset(datasetClasses == 2, :))]

% Define clusters classes
clustersClasses = knnsearch(classesCentroid, centroids);

% Find classification by centroids
c1IdxCentroids = knnsearch(centroids, dataset(datasetClasses == 1, :));
c1IdxCentroids(c1IdxCentroids == 1) = classesCentroid(1);
c1IdxCentroids(c1IdxCentroids == 2) = classesCentroid(2);

c2IdxCentroids = knnsearch(centroids, dataset(datasetClasses == 2, :));
c2IdxCentroids(c2IdxCentroids == 1) = classesCentroid(1);
c2IdxCentroids(c2IdxCentroids == 2) = classesCentroid(2);

% Calculate error per class
nC1CorrectlyAssigned = sum(c1IdxCentroids == 1);
nC2CorrectlyAssigned = sum(c2IdxCentroids == 2);

pC1Error     = 1 - nC1CorrectlyAssigned/size(c1IdxCentroids);
pC2Error     = 1 - nC2CorrectlyAssigned/size(c2IdxCentroids);
pGlobalError = 1- (nC1CorrectlyAssigned + nC2CorrectlyAssigned)/size(dataset,1);

% Calculate Rand

