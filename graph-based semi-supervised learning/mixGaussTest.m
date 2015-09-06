% Generate data
nLabeled = 20;
X = zeros(4096,2);
Y = zeros(4096,1);
for s = [0,2,4,8];
    s
    figure;
    hold on;
    for i = 1:4096
        tmp = rand(1);
        if tmp<0.5
            X(i,:) = randn(1,2);
            Y(i) = 1;
            plot(X(i,1),X(i,2),'.r');
        else
            X(i,:) = randn(1,2)+[s,0];
            Y(i) = -1;
            plot(X(i,1),X(i,2),'.b');
        end
    end
    
    title(['Mixture of Gaussians Data (s=' num2str(s) ')']);
    axis equal;
    trainingSet = [Y,X];
    hold off;

    % build graph and predict
    % Method 1: harmonic 
    numsOfData = [32,64,128,256,512,1024,2048];
    nTrial = 100;
    figure;
    color = ['b','g','r','c','m','y'];
    i = 1;
    for k = [2,5]
        k
        [ meanAcc, stdAcc ] = myHarmonic(trainingSet,nLabeled,numsOfData,nTrial,k);
        errorbar(log2(numsOfData),meanAcc,stdAcc,color(i));
        hold on;
        i = i+1;
    end
    legend('k=2','k=5');
    xlabel('log2 of total number of data');
    ylabel('accuracy');
    title('Harmonic');
    hold off;


    % Method 2a: SVM
    numsOfData = [32,64,128,256,512,1024,2048];
    nTrial = 100;
    figure;
    color = ['b','g','r','c','m','y'];

    % fix b
    i = 1;
    b = -1;
    for k = [2,5]
        k
        [ meanAcc, stdAcc ] = mySVM(trainingSet,nLabeled,numsOfData,nTrial,k,b);
        errorbar(log2(numsOfData),meanAcc,stdAcc,color(i));
        hold on;
        i = i+1;
    end
    legend('k=2','k=5');
    xlabel('log2 of total number of data');
    ylabel('accuracy');
    title('SVM (fixed b=max diagonal element of L+)');
    hold off;

    % fix k
    i = 1;
    k = 2;
    for b = [0,-1];    
        b
        [ meanAcc, stdAcc ] = mySVM(trainingSet,nLabeled,numsOfData,nTrial,k,b);
        errorbar(log2(numsOfData),meanAcc,stdAcc,color(i));
        hold on;
        i = i+1;
    end
    legend('b=0','b=max diagonal element of L+');
    xlabel('log2 of total number of data');
    ylabel('accuracy');
    title('SVM (fixed k=2)');
    hold off;

    % Method 2b: perceptron
    numsOfData = [32,64,128,256,512,1024,2048];
    nTrial = 100;
    figure;
    color = ['b','g','r','c','m','y'];

    % fix b
    i = 1;
    b = -1;
    for k = [2,5]
        k
        [ meanAcc, stdAcc ] = myPerceptron(trainingSet,nLabeled,numsOfData,nTrial,k,b);
        errorbar(log2(numsOfData),meanAcc,stdAcc,color(i));
        hold on;
        i = i+1;
    end
    legend('k=2','k=5');
    xlabel('log2 of total number of data');
    ylabel('accuracy');
    title('Perceptron (fixed b=max diagonal element of L+)');
    hold off;

    % fix k
    i = 1;
    k = 2;
    for b = [0,-1]; 
        b
        [ meanAcc, stdAcc ] = myPerceptron(trainingSet,nLabeled,numsOfData,nTrial,k,b);
        errorbar(log2(numsOfData),meanAcc,stdAcc,color(i));
        hold on;
        i = i+1;
    end
    legend('b=0','b=max diagonal element of L+');
    xlabel('log2 of total number of data');
    ylabel('accuracy');
    title('Perceptron (fixed k=2)');
    hold off;
end