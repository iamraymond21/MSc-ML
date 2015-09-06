% perceptron
% load('gooddata.mat');
load('originTrainData.mat');
dataSet = originTrainData([7290,7291,7256,7289,7260,7272,7266,7245,7257,7276,7275,7286,7244,7288,7254,7270,7235,7207,7252,7248,1:4076],:);

nData = 50;
nLabeled = 20;
r = 300;  % number of trials in one period
nPeriod = 10; % 1 + number of switches
k = 2; % using k-NN to build graph

% acc = zeros(1,4);
% for ii=1:100
 
iter = 1;
cumMistakeP = zeros(4,r*nPeriod);
accP = zeros(4,nPeriod);

% e = 10;
% dataPro = data;
% for i = 1:nData
%     if norm(data(i,2:end))>e
%         dataPro(i,2:end) = data(i,2:end)/norm(data(i,2:end))*e;
%     end
% end

for nData = [50,100,500,1000]
    % choose data
    index = nLabeled+randperm(size(dataSet,1)-nLabeled,nData-nLabeled);
    data = [dataSet(1:nLabeled,:); dataSet(index,:)];
    b = -1;
    % compute graph
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
    
    labelings = [1,1,2,2,1,3,3,2,2,3];
    groups = [1,3,5,7,9;1,2,3,4,5;1,2,4,7,0];

% for etai = [0.1, 0.4, 0.7, 1];
    etai = 1;
    tic;
    % initialize model
    alpha = zeros(nLabeled,1);  

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
                    alpha(t) = alpha(t)+etai*yData(t);
                    cumMistakeP(iter,(r*(iPeriod-1)+ir):end) = cumMistakeP(iter,(r*(iPeriod-1)+ir))+1;
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
        accP(iter,iPeriod) = sum(yPredict'==yData((nLabeled+1):nData))/(nData-nLabeled);
    end
    iter = iter+1;
    toc;
end

% acc = acc+mean(accP');
% end

% show cummulative mistakes
figure;
hold on;
plot(cumMistakeP(1,:),'b');
plot(cumMistakeP(2,:),'g');
plot(cumMistakeP(3,:),'r');
plot(cumMistakeP(4,:),'c');
title('graph-based kernel perceptron (vary n)');
legend('n=50','n=100', 'n=500', 'n=1000');
% title('graph-based kernel perceptron (vary eta)');
% legend('eta=0.1','eta=0.4', 'eta=0.7', 'eta=1');
% title('switching with latent line graph (vary theta)');
% legend('theta=0.1','theta=1','theta=default','theta=inf');
xlabel('trial');
ylabel('cummulative mistakes');
set(gca,'XTick',0:300:3000);
set(gca,'XGrid','on');

% disp(acc/100);