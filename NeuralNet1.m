function [outputArg1,outputArg2] = NeuralNet1(inputArg1,inputArg2)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
outputArg1 = inputArg1;
outputArg2 = inputArg2;

%table = readtable('../kddcupdata_10percent_duplicatesremoved_small.xlsx');
%table = readtable('../kddcupdata_10percent_duplicatesremoved_small_cols8_20_21_deleted.xlsx');
%table = readtable('../kddcupdata_10percent_duplicatesremoved_allTargets_Col7_8_9_15_20_21_deleted_multiclass.xlsx','Range','A1:AO145578');
%table = readtable('../kddcupdata_10percent_duplicatesremoved_allTargets_Col7_8_9_15_20_21_deleted_multiclass.xlsx','Range','A1:AO1000000');
%table = readtable('../kddcupdata_10percent_duplicatesremoved_allTargets_Col21_deleted_mulitclass.xlsx','Range','A1:AT100000');
table = readtable('../kddcupdata_10percent_duplicatesremoved_allTargets_Col20_21_deleted_mulitclass.xlsx','Range','A1:AS145587');

%X = table(:,[1:end-3]); %small file
X = table(:,1:39); %big file
A = table2array(X);

%T = table(:,[end-1:end]); %small file
T = table(:,41:end); %big file
targets = table2array(T);

%normalise data
normA = normalize(A);

%Remove NaNs
% Create an anonymous function to detect NaNs
f = @(x) any(isnan(x));
% Detect where there are NaNs
out = num2cell(normA,1);
B = cellfun(f,out);
% Now detect columns that contain ONLY NaNs
C = all(B);
% Remove these columns
normA(:,C) = [];

% Principal Component Analysis
[coeff,score,latent]  = pca(normA);

%use Kaiser criteria rather than constant???????????????? drop every
%component under 1.0 -> 11 for 50k sample
%                    -> 13 for 100k sample
%                     -> 10 for total dataset
k = 10;
%k = cumsum(latent)./sum(latent);
inputs = score(:,[1:k]);

% Detect where there are NaNs
out = num2cell(inputs,1);
B = cellfun(f,out);
% Now detect columns that contain ONLY NaNs
C = all(B);
% Remove these columns
inputs(:,C) = [];


% Create a Pattern Recognition Network
%hiddenLayerSize = 2;
%net = patternnet(hiddenLayerSize);
%
%inputs = transpose(A);
inputs = transpose(inputs);
targets = transpose(targets);

% K-fold cross validation
k = 10;
CVO = cvpartition(length(inputs),'KFold',k); %%% low perforance k = 3;


for i = 1:k     %# for each fold
    trainIdx = CVO.training(i);
    testIdx = CVO.test(i); %# get indices training instances
    trInd = find(trainIdx);
    tstInd = find(testIdx);
    % Create a Pattern Recognition Network
    %hiddenLayerSize = 10;
    %net = patternnet(hiddenLayerSize);
     if i == 1
        hiddenLayerSize = 10;
        net = patternnet(hiddenLayerSize);
        net.trainFcn = 'trainbr';
        %net.trainFcn = 'trainbr';
        net.trainParam.epochs = 50;
     end   
     if i == 2   
        hiddenLayerSize = 10;
        net = patternnet(hiddenLayerSize);
        net.trainFcn = 'trainbr';
        net.trainParam.epochs = 100;
     end   
     if i == 3  
        hiddenLayerSize = 10;
        net = patternnet(hiddenLayerSize);
        net.trainFcn = 'traingd';
%        net.trainFcn = 'traingd';
        net.trainParam.epochs = 50;
     end   
     if i == 4
        hiddenLayerSize = 10;
        net = patternnet(hiddenLayerSize);
        net.trainFcn = 'traingd';
        net.trainParam.epochs = 100;
     end   
     if i == 5
        hiddenLayerSize = 10;
        net = patternnet(hiddenLayerSize);
        %net.trainFcn = 'trainbr';
        net.trainFcn = 'traingda';
        net.trainParam.epochs = 50;
     end   
     if i == 6
        hiddenLayerSize = 10;
        net = patternnet(hiddenLayerSize);

        net.trainFcn = 'traingda';
        net.trainParam.epochs = 100;
     end   
     if i == 7
        hiddenLayerSize = 10;
        net = patternnet(hiddenLayerSize);

        net.trainFcn = 'trainrp';
        net.trainParam.epochs = 50;
     end   
     if i == 8
        hiddenLayerSize = 10;
        net = patternnet(hiddenLayerSize);

        net.trainFcn = 'trainrp';
        net.trainParam.epochs = 100;
     end  
     if i == 9
        hiddenLayerSize = 10;
        net = patternnet(hiddenLayerSize);
        net.trainFcn = 'trainb';
        net.trainParam.epochs = 50;
      end 
      if i == 10
        hiddenLayerSize = 10;
        net = patternnet(hiddenLayerSize);
        net.trainFcn = 'trainb';
        net.trainParam.epochs = 100;
     end 
     
    %net.trainParam.lr = 0.5; %Learning rate
    %net.trainFcn = 'trainbr';  
    %net.trainFcn = 'traingd';  
    %net.trainParam.epochs = 100;
    %net.trainParam.epochs = 200;
    net.divideFcn = 'divideind'; 
    net.divideParam.trainInd = trInd;
    net.divideParam.testInd = tstInd;

    % Choose a Performance Function
    net.performFcn = 'mse';  % Mean squared error

    % Train the Network
    [net,tr] = train(net,inputs,targets);

     %# test using test instances
    outputs = net(inputs);
    errors = gsubtract(targets,outputs);
    performance = perform(net,targets,outputs);
    trainTargets = targets .* tr.trainMask{1};
    testTargets = targets  .* tr.testMask{1};
    trainPerformance = perform(net,trainTargets,outputs);
    testPerformance = perform(net,testTargets,outputs);
    test(i)=testPerformance;
    %save net;
    figure, plotconfusion(targets,outputs);
    
end
accuracy=mean(test);
% View the Network
view(net);

end

