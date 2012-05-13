function [hVector, pValueVector, combMatrix] = compareClassifiers(globalErrorPerRoundMatrix)
%COMPARECLASSIFIERS Classifiers comparison through paired t-student test.
%
% INPUT:
% globalErrorPerRoundMatrix (r,c): global error
% values from c classifiers for r rounds.
%
% OUTPUT:
% hVector (nComb,1): h value for each nComb combination.
% pValueVector (nComb,1): p value for each nComb combination.
% combMatrix (nComb,2): nComb combinations of classifiers.
%
% {dlf2,dvro}@cin.ufpe.br

% Create combination matrix
combMatrix = combnk(1:size(globalErrorPerRoundMatrix,2),2);

% Initialize variables
hVector = zeros(size(combMatrix,1),1);
pValueVector = zeros(size(combMatrix,1),1);

for i = 1:size(combMatrix,1)
    [hVector(i), pValueVector(i)] = ttest(globalErrorPerRoundMatrix(:,combMatrix(i,1))-globalErrorPerRoundMatrix(:,combMatrix(i,2)));
end