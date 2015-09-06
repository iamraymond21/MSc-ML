function [ acc ] = myPerceptron1(data,K,nLabeled)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

    % get labels of training data
    yLabeled = data(1:nLabeled,1);

    % online learning a classifier
    mistake = nLabeled;
    alpha = zeros(nLabeled,1);
    while mistake~=0
        mistake = 0;
        for t = 1:nLabeled
            yHat = mysign(alpha'*K(1:nLabeled,t));
            if yHat~=yLabeled(t)
                alpha(t) = alpha(t)+yLabeled(t);
                mistake = mistake+1;
            end
        end
    end

    % predict unlabeled data and calculate accuracy
    yUnlabeledPredict = mysign(K(1:nLabeled,nLabeled+1:end)'*alpha);
    yUnlabeled = data(nLabeled+1:end,1);
    acc = sum(yUnlabeled==yUnlabeledPredict)/length(yUnlabeled);

end

