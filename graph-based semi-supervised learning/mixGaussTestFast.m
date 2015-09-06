% Generate data
nLabeled = 20;
X = zeros(4096,2);
Y = zeros(4096,1);
for s = [0,2,4,8];
    s
    figure;
    hold on;
    for iData = 1:4096
        tmp = rand(1);
        if tmp<0.5
            X(iData,:) = randn(1,2);
            Y(iData) = 1;
            plot(X(iData,1),X(iData,2),'.r');
        else
            X(iData,:) = randn(1,2)+[s,0];
            Y(iData) = -1;
            plot(X(iData,1),X(iData,2),'.b');
        end
    end
    
    title(['Mixture of Gaussians Data (s=' num2str(s) ')']);
    axis equal;
    trainingSet = [Y,X];
    hold off;

    
    numsOfData = [32,64,128,256,512,1024,2048];
    nTrial = 100;
    color = ['b','g','r','c','m','y'];
    
    meanAcc = zeros(length(numsOfData),8);
    stdAcc = zeros(length(numsOfData),8);
    
    for iData = 1:length(numsOfData)

        nData = numsOfData(iData)
        acc = zeros(nTrial,8);
        for iTrial = 1:nTrial
            % choose data
            index = nLabeled+randperm(size(trainingSet,1)-nLabeled,nData-nLabeled);
            data = [trainingSet(1:nLabeled,:); trainingSet(index,:)];

            for k = [2,5]
                % training
                % get labeled graph matrix
                graph = buildGraph(data,k);

                % get Laplacian matrix L
                L = diag(sum(graph))-graph;
                L = full(L);

                % get kernel matrix K
                Lplus = pinv(L);

                for b = [0,-1]
                    if b == -1
                        K = Lplus+ max(diag(Lplus));
                    else
                        K = Lplus+ b;
                    end

                    if k==2 && b==0
                    acc(iTrial,1) = myHarmonic1(data,L,nLabeled);
                    end
                    if k==5 && b==0
                    acc(iTrial,2) = myHarmonic1(data,L,nLabeled);
                    end
                    if (b==-1) && (k==2)
                        acc(iTrial,3) = mySVM1(data,K,nLabeled);
                    end
                    if b==-1 && k==5
                        acc(iTrial,4) = mySVM1(data,K,nLabeled);
                    end
                    if b==0 && k==2
                        acc(iTrial,5) = mySVM1(data,K,nLabeled);
                    end
                    if b==-1 && k==2
                        acc(iTrial,6) = myPerceptron1(data,K,nLabeled);
                    end
                    if b==-1 && k==5
                        acc(iTrial,7) = myPerceptron1(data,K,nLabeled);
                    end
                    if b==0 && k==2
                        acc(iTrial,8) = myPerceptron1(data,K,nLabeled);
                    end
                end
            end
        end
        meanAcc(iData,:) = mean(acc);
        stdAcc(iData,:) = std(acc);
    end

    figure;
    hold on;
    errorbar(log2(numsOfData),meanAcc(:,1),stdAcc(:,1),color(1));
    errorbar(log2(numsOfData),meanAcc(:,2),stdAcc(:,2),color(2));
    legend('k=2','k=5');
    xlabel('log2 of total number of data');
    ylabel('accuracy');
    title('Harmonic');
    hold off;
    
    figure;
    hold on;
    errorbar(log2(numsOfData),meanAcc(:,3),stdAcc(:,3),color(1));
    errorbar(log2(numsOfData),meanAcc(:,4),stdAcc(:,4),color(2));
    errorbar(log2(numsOfData),meanAcc(:,5),stdAcc(:,5),color(3));
    legend('k=2, b=max diagonal element of L+','k=5, b=max diagonal element of L+','k=2, b=0');
    xlabel('log2 of total number of data');
    ylabel('accuracy');
    title('SVM');
    hold off;
    
    figure;
    hold on;
    errorbar(log2(numsOfData),meanAcc(:,6),stdAcc(:,6),color(1));
    errorbar(log2(numsOfData),meanAcc(:,7),stdAcc(:,7),color(2));
    errorbar(log2(numsOfData),meanAcc(:,8),stdAcc(:,8),color(3));
    legend('k=2, b=max diagonal element of L+','k=5, b=max diagonal element of L+','k=2, b=0');
    xlabel('log2 of total number of data');
    ylabel('accuracy');
    title('Kernel Perceptron');
    hold off;
    
end