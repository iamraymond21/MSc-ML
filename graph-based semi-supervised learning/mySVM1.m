function [ acc ] = mySVM1(data,K,nLabeled)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    yLabeled = data(1:nLabeled,1);
    
    % learning a classifier with SVM
    [U,S,V] = svd(K);
    X = U*sqrt(S);
    svmModel = fitcsvm(X(1:nLabeled,:),yLabeled,'BoxConstraint',inf);

    % predict unlabeled data and calculate accuracy
    yUnlabeledPredict = predict(svmModel,X(nLabeled+1:end,:));
    yUnlabeled = data(nLabeled+1:end,1);
    acc = sum(yUnlabeled==yUnlabeledPredict)/length(yUnlabeled);
    
end