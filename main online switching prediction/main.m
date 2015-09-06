% load USPS data
% data is represented as a nData*(1+Dimension) matrix
load('originTrainData.mat');

% pre-process the data
% get a large data set
% dataSet = originTrainData([7290,7275,7291,7286,7256,7244,7289,7288,7260,7254,7272,7270,7266,7235,7245,7207,7257,7252,7276,7248,1:4076],:);
dataSet = originTrainData([7290,7291,7256,7289,7260,7272,7266,7245,7257,7276,7275,7286,7244,7288,7254,7270,7235,7207,7252,7248,1:4076],:);
% so that dataSet contains 4096 points, and the first 20 points are
% labeled and the others are regarded as unlabeled.

%% Switching
% initial set
nData = 50;
nLabeled = 20;
r = 300;  % number of trials in one period
nPeriod = 10; % 1 + number of switches
k = 2; % using k-NN to build graph

% choose data
% index = nLabeled+randperm(size(dataSet,1)-nLabeled,nData-nLabeled);
% data = [dataSet(1:nLabeled,:); dataSet(index,:)];
load('gooddata.mat');

etaSet = [0.3,0.4,0.3,0.4];
thetaSet = [0.1,0.1,inf,inf];

for ii = 1:4
    if ii~=2
        groups = [1,3,5,7,9;1,2,3,4,5;1,2,4,7,0];
    else
        groups = [1,3,5,7,9;1,2,3,4,5;6,7,8,9,0];
    end
    if ii==3
        labelings = [3,1,2,2,1,1,3,3,2,3];
    elseif ii==4
        labelings = [1,2,1,1,2,2,3,2,3,3];
    else
        labelings = [1,1,2,2,1,3,3,2,2,3];
    end
    
    % get G
    G = buildGraph(data,k);
    L = diag(sum(G))-G;
    L = full(L);
    Lplus = pinv(L);
    R = max(diag(Lplus));
    G = Lplus+ R;

    % get H
    H = diag(ones(300-1,1),1)+diag(ones(300-1,1),-1);
    p = length(H);
    LH = diag(sum(H))-H;
    LH = full(LH);
    LplusH = pinv(LH);
    R = max(diag(LplusH));
    H = LplusH + R;

    % switching algorithm
    K = 3;
    % eta = 1/2*log((K+3)/(K+1));
    eta = etaSet(ii);
    % labelings = randi(K,nPeriod,1);
    HLabel = kron(labelings,ones(1,30));
    % HLabel = randi(K,length(H),1);
    sqrtG = sqrtm(G);
    sqrtH = sqrtm(H);
    srG = sqrt(max(diag(G)));
    srH = sqrt(max(diag(H)));
    [theta, omega] = getTheta(data(:,1),HLabel,G,H,K,sqrtG,sqrtH,srG,srH,groups);
    theta = thetaSet(ii);

    tic;
    % initialization
    W = eye(nData+p)/K/(nData+p);
    [U,S,V] = svd(W);
    logW = U*diag(log(diag(S)))*V';

    % learning
    cumMistake = zeros(1,r*nPeriod);
    jt = 1;
    iTrial = 1;
    acc = zeros(nPeriod,1);
    for iPeriod = 1:nPeriod
        ir = 0;
        while 1
            for it = 1:nLabeled
                x = [sqrtG(:,it)/srG;sqrtH(:,jt)/srH];
                X = x*x'/2;
                yHat = predict(logW,X,K,theta);
                y = omega(it,labelings(iPeriod));
                if yHat~=y
                    logW = logW + eta*(y-yHat)*X;
                    cumMistake(iTrial:end) = cumMistake(iTrial)+1;
                    jt = jt+1;
                end
                iTrial = iTrial+1;
                ir = ir+1;
                if ir>=r
                    break;
                end
            end
            if ir>=r
                break;
            end
        end

        error = 0;
        for it = (nLabeled+1):nData
            x = [sqrtG(:,it)/srG;sqrtH(:,jt)/srH];
            X = x*x'/2;
            yHat = predict(logW,X,K,theta);
            y = omega(it,labelings(iPeriod));
            if yHat~=y
                error = error+1;
            end
        end
        acc(iPeriod) = 1-error/(nData-nLabeled);
    end
    toc;

    % show cummulative mistakes
    figure;
    plot(cumMistake,'b');
    title('predict a switching sequence of graph labelings');
    xlabel('trial');
    ylabel('cummulative mistakes');
    hold on;

    %% Switching tree

    % get H
    H = zeros(1023,1023);
    for iNode = 1:511
        H(iNode,2*iNode) = 1;
        H(2*iNode,iNode) = 1;
        H(iNode,2*iNode+1) = 1;
        H(2*iNode+1,iNode) = 1;
    end
    H = H(1:950,1:950);
    p = length(H);
    LH = diag(sum(H))-H;
    LH = full(LH);
    LplusH = pinv(LH);
    R = max(diag(LplusH));
    H = LplusH + R;

    % switching algorithm
    K = 3;
    % eta = 1/2*log((K+3)/(K+1));
    eta = etaSet(ii);
    % HLabel = kron(labelings,ones(1,r/5));
    HLabel = zeros(1023,1);
    HLabel(512:1023) = [kron(labelings,ones(1,51)),labelings(end),labelings(end)];
    for iNode = 511:-1:1
        HLabel(iNode) = HLabel(2*iNode);
    end
    HLabel = HLabel(1:950);
    sqrtH = sqrtm(H);
    srH = sqrt(max(diag(H)));
    [theta, omega] = getTheta(data(:,1),HLabel,G,H,K,sqrtG,sqrtH,srG,srH,groups);
    theta = thetaSet(ii);

    tic;
    % initialization
    W = eye(nData+p)/K/(nData+p);
    [U,S,V] = svd(W);
    logW = U*diag(log(diag(S)))*V';

    % learning
    cumMistake2 = zeros(1,r*nPeriod);
    jt = 512;
    iTrial = 1;
    acc2 = zeros(nPeriod,1);
    for iPeriod = 1:nPeriod
        ir = 0;
        while 1
            for it = 1:nLabeled
                x = [sqrtG(:,it)/srG;sqrtH(:,jt)/srH];
                X = x*x'/2;
                yHat = predict(logW,X,K,theta);
                y = omega(it,labelings(iPeriod));
                if yHat~=y
                    logW = logW + eta*(y-yHat)*X;
                    cumMistake2(iTrial:end) = cumMistake2(iTrial)+1;
                    jt = jt+1;
                end
                iTrial = iTrial+1;
                ir = ir+1;
                if ir>=r
                    break;
                end
            end
            if ir>=r
                break;
            end
        end

        error = 0;
        for it = (nLabeled+1):nData
            x = [sqrtG(:,it)/srG;sqrtH(:,jt)/srH];
            X = x*x'/2;
            yHat = predict(logW,X,K,theta);
            y = omega(it,labelings(iPeriod));
            if yHat~=y
                error = error+1;
            end
        end
        acc2(iPeriod) = 1-error/(nData-nLabeled);

    end
    toc;

    plot(cumMistake2,'g');

    %% perceptron
    % b = -1;
    cumMistake3 = zeros(1,r*nPeriod);
    % % compute graph
    % graph = buildGraph(data,k);
    % 
    % % get Laplacian matrix L
    % L = diag(sum(graph))-graph;
    % L = full(L);
    % 
    % % get kernel matrix K
    % Lplus = pinv(L);
    % if b == -1
    %     b = max(diag(Lplus));
    % end
    % K = Lplus+ b;
    K = G;

    tic;
    % initialize model
    alpha = zeros(nLabeled,1);  

    acc3 = zeros(nPeriod,1);
    for iPeriod = 1:nPeriod
        % choose classes
        tmp = (data(:,1)==groups(labelings(iPeriod),1)) + (data(:,1)==groups(labelings(iPeriod),2)) + (data(:,1)==groups(labelings(iPeriod),3)) + (data(:,1)==groups(labelings(iPeriod),4)) + (data(:,1)==groups(labelings(iPeriod),5));
        yData = zeros(nData,1);
        yData(tmp==true,1) = 1;
        yData(tmp==false,1) = -1;

    %         % initialize model
    %         alpha = zeros(nLabeled,1); 

        % online learning a classifier
        ir = 1;
        while 1
            for t = 1:nLabeled
                yHat = mysign(alpha'*K(1:nLabeled,t));
                if yHat~=yData(t)
                    alpha(t) = alpha(t)+yData(t);
                    cumMistake3((r*(iPeriod-1)+ir):end) = cumMistake3((r*(iPeriod-1)+ir))+1;
                end
                if ir>=r
                    break;
                end    
                ir = ir+1;
            end
            if ir>=r
                break;
            end
        end

        yPredict = mysign(alpha'*K(1:nLabeled,(nLabeled+1):nData));
        acc3(iPeriod) = sum(yPredict'==yData((nLabeled+1):nData))/(nData-nLabeled);
    end
    toc;

    plot(cumMistake3,'r');

    %% perceptron with projection
    e = 10;
    dataPro = data;
    for i = 1:nData
        if norm(data(i,2:end))>e
            dataPro(i,2:end) = data(i,2:end)/norm(data(i,2:end))*e;
        end
    end
    b = -1;
    cumMistake4 = zeros(1,r*nPeriod);
    % compute graph
    graph = buildGraph(dataPro,k);

    % get Laplacian matrix L
    L = diag(sum(graph))-graph;
    L = full(L);

    % get kernel matrix K
    Lplus = pinv(L);
    if b == -1
        b = max(diag(Lplus));
    end
    K = Lplus+ b;

    tic;
    % initialize model
    alpha = zeros(nLabeled,1);

    acc4 = zeros(nPeriod,1);
    for iPeriod = 1:nPeriod
        % choose classes
        tmp = (data(:,1)==groups(labelings(iPeriod),1)) + (data(:,1)==groups(labelings(iPeriod),2)) + (data(:,1)==groups(labelings(iPeriod),3)) + (data(:,1)==groups(labelings(iPeriod),4)) + (data(:,1)==groups(labelings(iPeriod),5));
        yData = zeros(nData,1);
        yData(tmp==true,1) = 1;
        yData(tmp==false,1) = -1;

    %         % initialize model
    %         alpha = zeros(nLabeled,1); 

        % online learning a classifier
        ir = 1;
        while 1
            for t = 1:nLabeled
                yHat = mysign(alpha'*K(1:nLabeled,t));
                if yHat~=yData(t)
                    alpha(t) = alpha(t)+yData(t);
    %                 if norm(alpha)>e
    %                     alpha = e*alpha/norm(alpha);
    %                 end
                    cumMistake4((r*(iPeriod-1)+ir):end) = cumMistake4((r*(iPeriod-1)+ir))+1;
                end
                if ir>=r
                    break;
                end    
                ir = ir+1;
            end
            if ir>=r
                break;
            end
        end

        yPredict = mysign(alpha'*K(1:nLabeled,(nLabeled+1):nData));
        acc4(iPeriod) = sum(yPredict'==yData((nLabeled+1):nData))/(nData-nLabeled);
    end
    toc;

    plot(cumMistake4,'c');
    legend('switching with latent graph (line chart)','switching with latent graph (binary tree)','kernel perceptron without projection','kernel perceptron with projection');
    set(gca,'XTick',0:300:3000);
    set(gca,'XGrid','on');
    hold off;

    %%
    mean([acc,acc2,acc3,acc4])
end  
    
% figure;
% hold on;
% plot(acc,'b');
% plot(acc2,'g');
% plot(acc3,'r');
% plot(acc4,'c');
% title('accuracy of switching');
% xlabel('period');
% ylabel('accuracy');



    