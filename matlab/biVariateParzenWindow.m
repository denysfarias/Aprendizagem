function pdf = biVariateParzenWindow(testArray, classTrainDataset, h_nArray)
%BIVARIATEPARZENWINDOW Calculate the density estimation for bivariate
%samples of testArray based on classTrainDataset and h_nVector.
% 
% INPUT:
% testArray (1,j): a test sample with j attributes.
% classTrainDataset (n,j): n train samples with j atrributes of specific class.
% h_nArray (1,j): array of h_n for each attribute.
%
% OUTPUT:
% pdf (1,1): the density estimation for the sample testArray based on
% classTrainDataset and h_nVector.
%
% {dlf2,dvro}@cin.ufpe.br

n = size(classTrainDataset,1);
hArray = (1/sqrt(n)).*h_nArray;
temp = 0;
for i = 1:n
    temp = temp + multiGaussianKernel(testArray, classTrainDataset(i,:), hArray);
end
pdf = temp/n;