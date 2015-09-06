function [ yHat ] = predict( logW,X,K,theta )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

[U,S,V] = svd(logW);
W = U*diag(exp(diag(S)))*V';

if trace(W*X)>=(K+1)/2/K/theta
    yHat = 1;
else
    yHat = -1;
end

end

