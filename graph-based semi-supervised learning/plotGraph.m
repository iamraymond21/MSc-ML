function [ A ] = plotGraph( data,A )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% [U,S,V] = svd(cov(data(:,2:end)));
% reconstructed = data(:,2:end)*U(:,1:2);
L = diag(sum(A))-A;
[v,d] = eig(full(L));

figure;
hold on;
plot(v(data(:,1)==1,2),v(data(:,1)==1,3),'*r');
plot(v(data(:,1)==-1,2),v(data(:,1)==-1,3),'*b');
%legend('digit 1','digit 2');
legend('odd digits','even digits');
title('odd vs even')
gplot(A,v(:,2:3),'k');

end

