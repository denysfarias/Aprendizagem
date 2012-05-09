function result = multiGaussianKernel(testVector, trainVector, hVector)

result = (1/prod(hVector));
for j = 1:numel(trainVector)
    result = result * uniGaussianKernel((testVector(j) - trainVector(j))/hVector(j));
end