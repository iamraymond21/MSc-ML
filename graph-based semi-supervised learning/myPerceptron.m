function [ meanAcc, stdAcc ] = myPerceptron(trainingSet,nLabeled,numsOfData,nTrial,k,b)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

meanAcc = zeros(1,length(numsOfData));
stdAcc = zeros(1,length(numsOfData));

for i = 1:length(numsOfData)
    nData = numsOfData(i);
    acc = zeros(1,nTrial);
    for j = 1:nTrial
        % choose data
        index = nLabeled+randperm(size(trainingSet,1)-nLabeled,nData-nLabeled);
        data = [trainingSet(1:nLabeled,:); trainingSet(index,:)];
        
        % training
        % get labeled graph matrix
        graph = buildGraph(data,k);

        % get Laplacian matrix L
        L = diag(sum(graph))-graph;
        L = full(L);

        % get kernel matrix K
        Lplus = pinv(L);
        if b == -1
            b = max(diag(Lplus));
        end
        K = Lplus+ b;

        % get labels of training data
        yLabeled = trainingSet(1:nLabeled,1);

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
        acc(j) = sum(yUnlabeled==yUnlabeledPredict)/length(yUnlabeled);
    end
    meanAcc(i) = mean(acc);
    stdAcc(i) = std(acc);
end
end

