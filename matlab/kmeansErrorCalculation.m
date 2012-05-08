function [pPerClassError, pGlobalError, adjustedRandIndex] = kmeansErrorCalculation(dataset,datasetClasses)

% Apply k-means 100 times for 2 clusters
[kmeansIndices, kmeansCentroids] = kmeans(dataset, 2, 'replicates', 100);

% Define classes centroids
classesCentroids = [mean(dataset(datasetClasses == 1, :));
    mean(dataset(datasetClasses == 2, :))];

% Define clusters classes
clustersClasses = knnsearch(classesCentroids, kmeansCentroids);

% Find classification by centroids
c1ClassificationByCentroids = knnsearch(kmeansCentroids, dataset(datasetClasses == 1, :));
c1ClassificationByCentroids(c1ClassificationByCentroids == 1) = clustersClasses(1);
c1ClassificationByCentroids(c1ClassificationByCentroids == 2) = clustersClasses(2);

c2ClassificationByCentroids = knnsearch(kmeansCentroids, dataset(datasetClasses == 2, :));
c2ClassificationByCentroids(c2ClassificationByCentroids == 1) = clustersClasses(1);
c2ClassificationByCentroids(c2ClassificationByCentroids == 2) = clustersClasses(2);

classificationByCentroids = kmeansIndices;
classificationByCentroids(classificationByCentroids == 1) = clustersClasses(1);
classificationByCentroids(classificationByCentroids == 2) = clustersClasses(2);

% Calculate error per class
nC1CorrectlyAssigned = sum(c1ClassificationByCentroids == 1);
nC2CorrectlyAssigned = sum(c2ClassificationByCentroids == 2);

pC1Error     = 1 - nC1CorrectlyAssigned/size(c1ClassificationByCentroids,1);
pC2Error     = 1 - nC2CorrectlyAssigned/size(c2ClassificationByCentroids,1);
pPerClassError = [pC1Error; pC2Error];
pGlobalError = 1- (nC1CorrectlyAssigned + nC2CorrectlyAssigned)/size(dataset,1);

% Calculate Rand
adjustedRandIndex = RandIndex(kmeansIndices, datasetClasses);

% Plot clusters
disp('%Plotting identified clusters.');
figure;
gscatter(dataset(:,1), dataset(:,2), classificationByCentroids, 'rbg', 'o', 5, 0);
title('Clusters', 'FontWeight','Bold','FontSize',14);
drawnow;