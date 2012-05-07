function dataset = createDistribution(mu, SIGMA, nExamples)

% Initialization...
nColumns = size(mu,2);
if (nColumns ~= size(SIGMA,2))
    error('The matrices mu and SIGMA should have the same number of columns.')
end

% Create multivariate gaussian distribution
dataset = mvnrnd(mu, SIGMA, nExamples);

% Plot distribution
% plot(dataset(:,1), dataset(:,2), '+');