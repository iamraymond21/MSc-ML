%% load USPS data
% data is represented as a nData*(1+Dimension) matrix
load('originTrainData.mat');
% or
oritinTrainData = load('ziptrain.dat');

% For instance, show the first handwritten digits and its label
figure;
imshow(reshape(originTrainData(1,2:end),16,16)');
disp(['The label of example graph is ' num2str(originTrainData(1,1))]);

% pre-process the data
% The first nLabeled data points are labeled, and others are unlabeled.
% 1 VS 2

nLabeled = 20;

% get a large training set
% tmpSet = sortrows(originTrainData,1);
% dataSet = tmpSet([1195:1204,2921:2930,1205:2920],:);
% tmp = (dataSet(:,1)==1);
% dataSet(tmp==true,1) = 1;
% dataSet(tmp==false,1) = -1;
% so that trainingSet contains 4096 points, and the first 20 points are
% labeled and the others are regarded as unlabeled.

% odd vs even
dataSet = originTrainData([7290,7275,7291,7286,7256,7244,7289,7288,7260,7254,7272,7270,7266,7235,7245,7207,7257,7252,7276,7248,1:4076],:);
tmp = (dataSet(:,1)==1) + (dataSet(:,1)==3) + (dataSet(:,1)==5) + (dataSet(:,1)==7) + (dataSet(:,1)==9);
dataSet(tmp==true,1) = 1;
dataSet(tmp==false,1) = -1;


%% build graph and predict
%% Method 1: harmonic 
numsOfData = [32,64,128,256,512,1024];
nTrial = 100;
figure;
color = ['b','g','r','c','m','k'];
i = 1;
for k = [2,5,10,20]
    k
    [ meanAcc, stdAcc ] = myHarmonic(dataSet,nLabeled,numsOfData,nTrial,k);
    errorbar(log2(numsOfData),meanAcc,stdAcc,color(i));
    hold on;
    i = i+1;
end
legend('k=2','k=5','k=10','k=20');
xlabel('log of total number of data');
ylabel('accuracy');
title('Harmonic');
hold off;


%% Method 2a: SVM
numsOfData = [32,64,128,256,512,1024];
nTrial = 100;
figure;
color = ['b','g','r','c','m','k'];

% fix b
i = 1;
b = -1;
for k = [2,5,10,20]
    k
    [ meanAcc, stdAcc ] = mySVM(dataSet,nLabeled,numsOfData,nTrial,k,b);
    errorbar(log2(numsOfData),meanAcc,stdAcc,color(i));
    hold on;
    i = i+1;
end
legend('k=2','k=5','k=10','k=20');
xlabel('log of total number of data');
ylabel('accuracy');
title('SVM (fixing b=max diagonal element of L+)');
hold off;

% fix k
figure;
i = 1;
k = 2;
for b = [0,0.5,1,-1];    
    b
    [ meanAcc, stdAcc ] = mySVM(dataSet,nLabeled,numsOfData,nTrial,k,b);
    errorbar(log2(numsOfData),meanAcc,stdAcc,color(i));
    hold on;
    i = i+1;
end
legend('b=0','b=0.5','b=1','b=max diagonal element of L+');
xlabel('log of total number of data');
ylabel('accuracy');
title('SVM (fixing k=2)');
hold off;

%% Method 2b: perceptron
numsOfData = [32,64,128,256,512,1024];
nTrial = 100;
figure;
color = ['b','g','r','c','m','k'];

% fix b
i = 1;
b = -1;
for k = [2,5,10,20]
    k
    [ meanAcc, stdAcc ] = myPerceptron(dataSet,nLabeled,numsOfData,nTrial,k,b);
    errorbar(log2(numsOfData),meanAcc,stdAcc,color(i));
    hold on;
    i = i+1;
end
legend('k=2','k=5','k=10','k=20');
xlabel('log of total number of data');
ylabel('accuracy');
title('Perceptron (fixing b=max diagonal element of L+)');
hold off;

% fix k
figure;
i = 1;
k = 2;
for b = [0,0.5,1,-1]; 
    b
    [ meanAcc, stdAcc ] = myPerceptron(dataSet,nLabeled,numsOfData,nTrial,k,b);
    errorbar(log2(numsOfData),meanAcc,stdAcc,color(i));
    hold on;
    i = i+1;
end
legend('b=0','b=0.5','b=1','b=max diagonal element of L+');
xlabel('log of total number of data');
ylabel('accuracy');
title('Perceptron (fixing k=2)');
hold off;