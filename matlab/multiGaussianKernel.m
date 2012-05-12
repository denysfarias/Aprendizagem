function result = multiGaussianKernel(testArray, trainArray, hArray)
%MULTIGAUSSIANKERNEL Calculate the multivariate gaussian kernel.
% 
% INPUT:
% testArray (1,j): a test sample with j attributes.
% trainArray (1,j): a train sample with j attributes.
% hArray (1,j): array of h for each attribute.
%
% OUTPUT:
% result (1,1): multivariate gaussian kernel value.
%
% {dlf2,dvro}@cin.ufpe.br

result = (1/prod(hArray));
for j = 1:numel(trainArray)
    result = result * uniGaussianKernel((testArray(j) - trainArray(j))/hArray(j));
end