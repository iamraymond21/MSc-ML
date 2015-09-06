% Generate data
nLabeled = 5;
X = zeros(4096,2);
Y = zeros(4096,1);
s = 8;
for i = 1:4096
    tmp = rand(1);
    if tmp<0.5
        X(i,:) = randn(1,2);
        Y(i) = 1;
    else
        X(i,:) = randn(1,2)+[s,0];
        Y(i) = -1;
    end
end
trainingSet = [Y,X];

% build graph and predict
% Method 1: harmonic 
disp('harmonic');
nData = 10;
k = 2;
% choose data
index = nLabeled+randperm(size(trainingSet,1)-nLabeled,nData-nLabeled);
data = [trainingSet(1:nLabeled,:); trainingSet(index,:)];
figure;
plot(data(:,2),data(:,3),'.');
legend(['s=' num2str(s)]);
xlim([-5,15]);
axis equal;

% get graph matrix
graph = buildGraphTest(data,k,nLabeled);

% get Laplacian matrix L
L = diag(sum(graph))-graph;

% get the true labels of data
yLabeled = data(1:nLabeled,1);
yUnlabeled = data(nLabeled+1:end,1);

% predict and calculate accuracy
yUnlabeledPredictScore = -inv(L(nLabeled+1:end,nLabeled+1:end))*L(nLabeled+1:end,1:nLabeled)*yLabeled
yUnlabeledPredict = yUnlabeledPredictScore;
yUnlabeledPredict(yUnlabeledPredictScore>0) = 1;
yUnlabeledPredict(yUnlabeledPredictScore<0) = -1;
acc = sum(yUnlabeled==yUnlabeledPredict)/length(yUnlabeled)


% Method 2a: SVM
% fix k
disp('SVM');
i = 1;
k = 2;
b = 0;
nData = 10;

% training
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

% learning a classifier with SVM
[U,S,V] = svd(K);
X = U*sqrt(S);
svmModel = fitcsvm(X(1:nLabeled,:),yLabeled,'BoxConstraint',inf);

% predict unlabeled data and calculate accuracy
yUnlabeledPredictScore = X(nLabeled+1:end,:)*svmModel.Beta+svmModel.Bias
yUnlabeledPredict = predict(svmModel,X(nLabeled+1:end,:));
yUnlabeled = data(nLabeled+1:end,1);
acc = sum(yUnlabeled==yUnlabeledPredict)/length(yUnlabeled)
