function [ acc ] = myHarmonic1(data,L,nLabeled)

    % get the true labels of data
    yLabeled = data(1:nLabeled,1);
    yUnlabeled = data(nLabeled+1:end,1);

    % predict and calculate accuracy
    yUnlabeledPredict = -inv(L(nLabeled+1:end,nLabeled+1:end))*L(nLabeled+1:end,1:nLabeled)*yLabeled;
    yUnlabeledPredict(yUnlabeledPredict>0) = 1;
    yUnlabeledPredict(yUnlabeledPredict<0) = -1;
    acc = sum(yUnlabeled==yUnlabeledPredict)/length(yUnlabeled);
    
end