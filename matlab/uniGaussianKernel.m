function result = uniGaussianKernel(u)
%UNIGAUSSIANKERNEL Calculate the univariate gaussian kernel centered on u with variance zero.
% 
% INPUT:
% u (1,1): gaussian center value.
%
% OUTPUT:
% result (1,1): univariate gaussian kernel value centered on u with variance zero.
%
% {dlf2,dvro}@cin.ufpe.br

result = (1/sqrt(2*pi))*exp(-0.5*u^2);