function [ theta, omega] = getTheta(yData,HLabel,G,H,K,sqrtG,sqrtH,srG,srH, groups)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

omega = zeros(length(G),K);
mu = zeros(length(H),K);
z = zeros(length(G)+length(H),K);
theta = 0;
% groups = zeros(K,5);
% groups = [1,3,5,7,9;1,2,3,4,5;1,2,4,7,0];
for iK = 1:K
%     class = randperm(10);
%     class(class == 10) = 0;
%     groups(iK,:) = class(1:5);
    class = groups(iK,:);
    tmp = (yData==class(1)) + (yData==class(2)) + (yData==class(3)) + (yData==class(4)) + (yData==class(5));
    omega(tmp==true,iK) = 1;
    omega(tmp==false,iK) = -1;
    mu(HLabel==iK,iK) = 1;
    z(:,iK) = [srG*inv(sqrtG)*omega(:,iK);srH*inv(sqrtH)*mu(:,iK)];
    theta = theta + norm(z(:,iK))^2;
end

end

