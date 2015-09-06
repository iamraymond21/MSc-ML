function [ meanAcc, stdAcc ] = myHarmonic(trainingSet,nLabeled,numsOfData,nTrial,k)

meanAcc = zeros(1,length(numsOfData));
stdAcc = zeros(1,length(numsOfData));
for i = 1:length(numsOfData)
    nData = numsOfData(i);
    acc = zeros(1,nTrial);
    for j = 1:nTrial
        % choose data
        index = nLabeled+randperm(size(trainingSet,1)-nLabeled,nData-nLabeled);
        data = [trainingSet(1:nLabeled,:); trainingSet(index,:)];

        % get graph matrix
        graph = buildGraph(data,k);

        % get Laplacian matrix L
        L = diag(sum(graph))-graph;

        % get the true labels of data
        yLabeled = data(1:nLabeled,1);
        yUnlabeled = data(nLabeled+1:end,1);

        % predict and calculate accuracy
        yUnlabeledPredict = -inv(L(nLabeled+1:end,nLabeled+1:end))*L(nLabeled+1:end,1:nLabeled)*yLabeled;
        yUnlabeledPredict(yUnlabeledPredict>0) = 1;
        yUnlabeledPredict(yUnlabeledPredict<0) = -1;
        acc(j) = sum(yUnlabeled==yUnlabeledPredict)/length(yUnlabeled);
    end
    meanAcc(i) = mean(acc);
    stdAcc(i) = std(acc);
end

end