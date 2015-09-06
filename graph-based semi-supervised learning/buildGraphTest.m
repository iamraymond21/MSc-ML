function [ graph ] = buildGraphTest( data, k, nLabeled )
% This function is to build graph based on data, using 2-NN.
% data is input, which is a nData*(1+Dimension) matrix, and the first column contains the labels
% weightGraph is output, which is a nData*nData matrix

nData = size(data,1);
xData = data(:,2:end);

% get two nearest neighbours for each point
[~,I] = pdist2(xData,xData,'euclidean','Smallest',k+1);
kNN = I(2:end,:);

% build k-NN graph
kNNgraph = zeros(nData,nData);
for i =1:nData
    for j = 1:k
        kNNgraph(i,kNN(j,i))= 1;
        kNNgraph(kNN(j,i),i)= 1;
    end
end
kNNgraph = sparse(kNNgraph);

% display k-NN graph
% kNNgraphShow = biograph(tril(kNNgraph),[],'ShowArrows','off');
% view(kNNgraphShow);

% get minimum spanning tree
D = pdist2(xData,xData);
weight = D.^2;
weight = sparse(weight);
tree = graphminspantree(weight);

% display the tree
% treeShow = biograph(tree,[],'ShowArrows','off');
% view(treeShow);

% get the final graph
graphTmp = tree + tril(kNNgraph);
graphTmp = graphTmp>0;
graph = graphTmp + graphTmp';

% display the final graph
A = plotGraph(data,full(graph));
% graphShow = biograph(graphTmp,[],'ShowArrows','off');
% for i = 1:nLabeled
%      set(graphShow.nodes(i),'Label',num2str(data(i,1)));
% end
% for i = nLabeled+1:nData
%     set(graphShow.nodes(i),'Label','unlabeled');
% end
% view(graphShow);
% graphShow2 = biograph(graphTmp,[],'ShowArrows','off');
% for i = 1:nData
%      set(graphShow2.nodes(i),'Label',num2str(data(i,1)));
% end
% view(graphShow2);

% get the weighted graph
% weightGraph = weight;
% weightGraph(graph == 0) = 0;
end

