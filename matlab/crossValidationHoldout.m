function [trainIndexVector, testIndexVector] = crossValidationHoldout(datasetClasses, trainFraction)
%CROSSVALIDATIONHOLDOUT Divide dataset in train/test sets using stratified
%Holdout Cross Validation.
%
% INPUT:
% datasetClassVector (m,1): vector of classes for m samples.
% trainFraction (1,1): train set proportion to datasetClassVector.
% 
% OUTPUT:
% trainIndexVector (m,1): train indices for the m samples.
% testIndexVector (m,1): test indices for the m samples.
%
% {dlf2,dvro}@cin.ufpe.br

% Divide train/test sets
[trainIndexVector, testIndexVector] = crossvalind('HoldOut', datasetClasses, 1.0 - trainFraction);