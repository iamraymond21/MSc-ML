load('gooddata.mat');

% initial set
nData = 50;
nLabeled = 20;
r = 300;  % number of trials in one period
nPeriod = 10; % 1 + number of switches
k = 2; % using k-NN to build graph

% get G
G = buildGraph(data,k);
L = diag(sum(G))-G;
L = full(L);
Lplus = pinv(L);
R = max(diag(Lplus));
G = Lplus+ R;

% get H
H = zeros(1023,1023);
for iNode = 1:511
    H(iNode,2*iNode) = 1;
    H(2*iNode,iNode) = 1;
    H(iNode,2*iNode+1) = 1;
    H(2*iNode+1,iNode) = 1;
end
H = H(1:860,1:860);
p = length(H);
LH = diag(sum(H))-H;
LH = full(LH);
LplusH = pinv(LH);
R = max(diag(LplusH));
H = LplusH + R;

% switching algorithm
K = 3;
eta = 1/2*log((K+3)/(K+1));
etai = 0.5;
labelings = [1,1,2,2,1,3,3,2,2,3];
% HLabel = kron(labelings,ones(1,r/5));
HLabel = zeros(1023,1);
HLabel(512:1023) = [kron(labelings,ones(1,51)),labelings(end),labelings(end)];
for iNode = 511:-1:1
    HLabel(iNode) = HLabel(2*iNode);
end
HLabel = HLabel(1:860);
sqrtG = sqrtm(G);
sqrtH = sqrtm(H);
srG = sqrt(max(diag(G)));
srH = sqrt(max(diag(H)));
groups = [1,3,5,7,9;1,2,3,4,5;1,2,4,7,0];
% groups = [1,3,5,7,9;1,2,3,4,5;6,7,8,9,0];
[theta, omega] = getTheta(data(:,1),HLabel,G,H,K,sqrtG,sqrtH,srG,srH,groups);
thetai = 0.1;

iter = 1;
cumMistake = zeros(4,r*nPeriod);
acc = zeros(4,nPeriod);
% for etai = [0.1, eta, 0.5, 0.9]
%     thetai = theta;
% for thetai = [theta/2, theta, theta*1.5, theta*2]
%    etai = eta;
    tic;
    % initialization
    W = eye(nData+p)/K/(nData+p);
    [U,S,V] = svd(W);
    logW = U*diag(log(diag(S)))*V';

    % learning
    jt = 1;
    iTrial = 1;
    for iPeriod = 1:nPeriod
        iPeriod
        ir = 0;
        while 1
            for it = 1:nLabeled
                x = [sqrtG(:,it)/srG;sqrtH(:,jt)/srH];
                X = x*x'/2;
                yHat = predict(logW,X,K,thetai);
                y = omega(it,labelings(iPeriod));
                if yHat~=y
                    logW = logW + etai*(y-yHat)*X;
                    cumMistake(iter,iTrial:end) = cumMistake(iter,iTrial)+1;
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
            yHat = predict(logW,X,K,thetai);
            y = omega(it,labelings(iPeriod));
            if yHat~=y
                error = error+1;
            end
        end
        acc(iter,iPeriod) = 1-error/(nData-nLabeled);
    end
    iter = iter+1;
    toc;
% end

figure;
hold on;
plot(cumMistake(1,:),'b');
% plot(cumMistake(2,:),'g');
% plot(cumMistake(3,:),'r');
% plot(cumMistake(4,:),'c');
title('switching with latent binary tree (vary eta)');
% legend('eta=1/2log((K+3)/(K+1))','eta=0.1', 'eta=0.5', 'eta=0.9');
% title('switching with line graph (vary theta)');
xlabel('trial');
ylabel('cummulative mistakes');
set(gca,'XTick',0:300:3000);
set(gca,'XGrid','on');

% figure;
% hold on;
% plot(acc(1,:),'b');
% plot(acc(2,:),'g');
% plot(acc(3,:),'r');
% plot(acc(4,:),'c');
% title('switching with latent binary tree (vary eta)');
% % title('switching with line graph (vary theta)');
% legend('eta=1/2log((K+3)/(K+1))','eta=0.1', 'eta=0.5', 'eta=0.9');
% xlabel('trial');
% ylabel('cummulative mistakes');

%%
iter = 1;
cumMistake0 = zeros(4,r*nPeriod);
acc0 = zeros(4,nPeriod);
for thetai = [0.1, 1, theta, inf]
    etai = eta;
    tic;
    % initialization
    W = eye(nData+p)/K/(nData+p);
    [U,S,V] = svd(W);
    logW = U*diag(log(diag(S)))*V';

    % learning
    jt = 1;
    iTrial = 1;
    for iPeriod = 1:nPeriod
        iPeriod
        ir = 0;
        while 1
            for it = 1:nLabeled
                x = [sqrtG(:,it)/srG;sqrtH(:,jt)/srH];
                X = x*x'/2;
                yHat = predict(logW,X,K,thetai);
                y = omega(it,labelings(iPeriod));
                if yHat~=y
                    logW = logW + etai*(y-yHat)*X;
                    cumMistake0(iter,iTrial:end) = cumMistake0(iter,iTrial)+1;
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
            yHat = predict(logW,X,K,thetai);
            y = omega(it,labelings(iPeriod));
            if yHat~=y
                error = error+1;
            end
        end
        acc0(iter,iPeriod) = 1-error/(nData-nLabeled);
    end
    iter = iter+1;
    toc;
end

% show cummulative mistakes
figure;
hold on;
plot(cumMistake0(1,:),'b');
plot(cumMistake0(2,:),'g');
plot(cumMistake0(3,:),'r');
plot(cumMistake0(4,:),'c');
% title('switching with line graph (vary eta)');
title('switching with latent binary tree (vary theta)');
legend('theta=0.1','theta=1','theta=default','theta=inf');
xlabel('trial');
ylabel('cummulative mistakes');
set(gca,'XTick',0:300:3000);
set(gca,'XGrid','on');

% figure;
% hold on;
% plot(acc(1,:),'b');
% plot(acc(2,:),'g');
% plot(acc(3,:),'r');
% plot(acc(4,:),'c');
% % title('switching with line graph (vary eta)');
% title('switching with line graph (vary theta)');
% xlabel('trial');
% ylabel('cummulative mistakes');

%% nData
% initial set
load('originTrainData.mat');
dataSet = originTrainData([7290,7291,7256,7289,7260,7272,7266,7245,7257,7276,7275,7286,7244,7288,7254,7270,7235,7207,7252,7248,1:4076],:);

nLabeled = 20;
r = 300;  % number of trials in one period
nPeriod = 10; % 1 + number of switches
k = 2; % using k-NN to build graph

% get H
H = zeros(1023,1023);
for iNode = 1:511
    H(iNode,2*iNode) = 1;
    H(2*iNode,iNode) = 1;
    H(iNode,2*iNode+1) = 1;
    H(2*iNode+1,iNode) = 1;
end
p = length(H);
LH = diag(sum(H))-H;
LH = full(LH);
LplusH = pinv(LH);
R = max(diag(LplusH));
H = LplusH + R;
    
iter = 1;
cumMistake3 = zeros(4,r*nPeriod);
acc3 = zeros(4,nPeriod);
for nData = [25,50,75,100];
    index = nLabeled+randperm(size(dataSet,1)-nLabeled,nData-nLabeled);
    data = [dataSet(1:nLabeled,:); dataSet(index,:)];
    % get G
    G = buildGraph(data,k);
    L = diag(sum(G))-G;
    L = full(L);
    Lplus = pinv(L);
    R = max(diag(Lplus));
    G = Lplus+ R;

    % switching algorithm
    K = 3;
    eta = 1/2*log((K+3)/(K+1));
    labelings = [1,1,2,2,1,3,3,2,2,3];
    % HLabel = kron(labelings,ones(1,r/5));
    HLabel = zeros(1023,1);
    HLabel(512:1023) = [kron(labelings,ones(1,51)),labelings(end),labelings(end)];
    for iNode = 511:-1:1
        HLabel(iNode) = HLabel(2*iNode);
    end
    sqrtG = sqrtm(G);
    sqrtH = sqrtm(H);
    srG = sqrt(max(diag(G)));
    srH = sqrt(max(diag(H)));
    groups = [1,3,5,7,9;1,2,3,4,5;1,2,4,7,0];
    % groups = [1,3,5,7,9;1,2,3,4,5;6,7,8,9,0];
    [theta, omega] = getTheta(data(:,1),HLabel,G,H,K,sqrtG,sqrtH,srG,srH,groups);


%     for etai = [0.1, eta, 0.5, 0.9]
    thetai = theta;
    % for thetai = [theta/2, theta, theta*1.5, theta*2]
    etai = eta;
    tic;
    % initialization
    W = eye(nData+p)/K/(nData+p);
    [U,S,V] = svd(W);
    logW = U*diag(log(diag(S)))*V';

    % learning
    jt = 1;
    iTrial = 1;
    for iPeriod = 1:nPeriod
        iPeriod
        ir = 0;
        while 1
            for it = 1:nLabeled
                x = [sqrtG(:,it)/srG;sqrtH(:,jt)/srH];
                X = x*x'/2;
                yHat = predict(logW,X,K,thetai);
                y = omega(it,labelings(iPeriod));
                if yHat~=y
                    logW = logW + etai*(y-yHat)*X;
                    cumMistake3(iter,iTrial:end) = cumMistake3(iter,iTrial)+1;
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
            yHat = predict(logW,X,K,thetai);
            y = omega(it,labelings(iPeriod));
            if yHat~=y
                error = error+1;
            end
        end
        acc3(iter,iPeriod) = 1-error/(nData-nLabeled);
    end
    iter = iter+1;
    toc;
end

figure;
hold on;
plot(cumMistake3(1,:),'b');
plot(cumMistake3(2,:),'g');
plot(cumMistake3(3,:),'r');
plot(cumMistake3(4,:),'c');
% title('switching with latent binary tree (vary eta)');
% legend('eta=1/2log((K+3)/(K+1))','eta=0.1', 'eta=0.5', 'eta=0.9');
% title('switching with line graph (vary theta)');
xlabel('trial');
ylabel('cummulative mistakes');
set(gca,'XTick',0:300:3000);
set(gca,'XGrid','on');
title('switching with latent binary tree (vary number of data)');
legend('n=25','n=50', 'n=75', 'n=100');

