clear 
data = 'cora';  % 'cora' or 'citeseer' or 'PubMed'
%% Parameters
k=40;  % dimensions of matrix W and H
mu_ = 0.04;
lambda = 0.2;
lambda1 = 0.2;   % regularization parameter
lambda2 = 0.00002;
textRank = 200; % dimension of text feature, used when preprocessing
%% Input
load([data,'/graph.txt']);
load([data,'/group.txt']);
if strcmp(data,'citeseer')
    numOfGroup = 6;
elseif strcmp(data,'cora')
    numOfGroup = 7;
else
    numOfGroup = 3;
end
group(:,1) = group(:,1) + 1;
if strcmp(data,'cora')==1||strcmp(data,'citeseer')==1
    group(:,2) = group(:,2) + 1;
end
max_group_num = 0;
max_group_id = -1;
for i1 = 1:numOfGroup
   num = nnz(group(:,2)==i1);
   if num>max_group_num
       max_group_num = num;
       max_group_id = i1;
   end
end
numOfNode = size(group,1);
labels = zeros(numOfNode,1);
for i1 = 1:size(group,1)
    if group(i1,2) == max_group_id
        labels(group(i1,1),1) = 1;
    else
        labels(group(i1,1),1) = -1;
    end
end
%% Reduce the dimension of text feature from TFIDF matrix
display ('Preprocessing text feature...');
if strcmp(data,'cora')||strcmp(data,'citeseer')   % compute TFIDF matrix for cora and citeseer datasets
   load([data,'/feature.txt']);
   numOfNode = size(feature,1);
   for i=1:size(feature,2)
       if (nnz(feature(:,i)) > 0)
           feature(:,i) = feature(:,i)*log(numOfNode/nnz(feature(:,i)));
       end
   end
   [U,S,V] = svds(feature, textRank);
   text_feature = U * S;
   clear U S V
else
   load([data,'/tfidf.txt']);
   tfidf(:,1) = tfidf(:,1) + 1;
   tfidf(:,2) = tfidf(:,2) + 1;
   tfidf = sparse(tfidf(:,1),tfidf(:,2),tfidf(:,3),max(tfidf(:,1)),max(tfidf(:,2)));
   [U,S,V] = svds(tfidf, textRank);
   text_feature = U * S;
   clear U S V
end
%% Build matrix M=(A+A*A)/2
display ('Computing matrix M...');
graph(:,1) = graph(:,1) + ones(size(graph(:,1)));
graph(:,2) = graph(:,2) + ones(size(graph(:,2)));
graph = [graph;graph(:,2) graph(:,1)];
graph = sparse(graph(:,1),graph(:,2),ones(size(graph(:,1))),numOfNode,numOfNode);
[rows,cols,vals] = find(graph);
graph = sparse(rows,cols,ones(length(rows),1),numOfNode,numOfNode);
[rows,cols,vals] = find(diag(graph));
if ~isempty(rows)
    for i1 = 1:length(rows)
        graph(rows(i1),rows(i1)) = 0;
    end
end
ColFeatures = text_feature;
for i=1:size(graph,1)
    if (norm(graph(i,:))>0)
        graph(i,:) = graph(i,:)/nnz(graph(i,:));
    end
end
g2 = graph * graph;
graph = graph + g2;
graph = graph ./ 2;

for i=1:size(ColFeatures,2)
    if (norm(ColFeatures(:,i))>0)
        ColFeatures(:,i) = ColFeatures(:,i)/norm(ColFeatures(:,i));
    end
end
%% Learning Parameters
display ('Learning Parameters...');
Features = speye(numOfNode);
[W1, H1, time] = IMC(sparse(graph), sparse(Features), sparse(ColFeatures), 80, lambda, 10); % learn TADW representations
F1_Matrix = zeros(10,10); % F1 values for TADW+SVM
F2_Matrix = zeros(10,10); % F1 values for DMF
for i1 = 1:10
    train_ratio = 0.02*i1; % for PubMed, train_ratio varies from 0.002 to 0.02
    for i3 = 1:10
        rp = randperm(numOfNode);
        
        testId = sort(rp(1:floor(numOfNode*(1-train_ratio))));
        trainId = 1:numOfNode;
        trainId(:,testId) = [];

        X1 = [W1',ColFeatures*H1'];
        model = svmtrain(labels(trainId,1), X1(trainId,:), '-t 0 -c 10');
        [predict_label, accuracy, dec_values] = svmpredict(labels(testId), X1(testId,:), model);
        TP1 = nnz((predict_label==1).*(labels(testId,1)==1));
        FP1 = nnz((predict_label==1).*(labels(testId,1)==-1));
        FN1 = nnz((predict_label==-1).*(labels(testId,1)==1));
        F1 = 2*TP1/(2*TP1+FP1+FN1);
        F1_Matrix(i1,i3) = F1;
        
        [ W2, H2, eta, iter ] = DMF_Solver( sparse(graph), sparse(ColFeatures'), k, labels, trainId, mu_, lambda1, lambda2 );
        X2 = [W2',ones(numOfNode,1)];
        output = X2(testId,:)*eta;
        TP2 = nnz((output>0).*(labels(testId,1)==1));
        FP2 = nnz((output>0).*(labels(testId,1)==-1));
        FN2 = nnz((output<=0).*(labels(testId,1)==1));
        F2 = 2*TP2/(2*TP2+FP2+FN2);
        F2_Matrix(i1,i3) = F2;
    end
end
F1_Matrix
F2_Matrix
