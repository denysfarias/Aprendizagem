function pdf = biVariateParzenWindow(testVector, classTrainDataset, h_nVector)

n = size(classTrainDataset,1);
hVector = (1/sqrt(n)).*h_nVector;
temp = 0;
for i = 1:n
    temp = temp + multiGaussianKernel(testVector, classTrainDataset(i,:), hVector);
end
pdf = temp/n;