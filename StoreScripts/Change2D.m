Data_x = [];

for idx = 1 : length(training)
    
    Data_x = cat(1, Data_x, training{idx});
    
end

save('Train_Data.mat', 'Data_x', 'Prediction');

Downsample = 2;
Data_x = Data_x(:, 1:Downsample:end, :);
Data_x = reshape(Data_x, [5440, 200*57]);

train_x = Data_x(1:5000, :);
train_y = Prediction(1:5000);
test_x = Data_x(5001:end, :);
test_y = Prediction(5001:end);

test_y = [test_y, 1-test_y];
train_y = [train_y, 1-train_y];

kk = randperm(5440);
train_x = X(kk(1:5000), :);
train_y = Prediction(kk(1:5000), :);
test_x = X(kk(5001:end), :);
test_y = Prediction(kk(5001:end), :);

test_y = [test_y, 1-test_y];
train_y = [train_y, 1-train_y];

%% select raw data channel
train_x_all = cell(16, 1);
train_y_all = cell(16, 1);
load('Training.mat');
for idx = 1 : 16
    subject_train_x = [];
    subject_train_y = [];
    for sec = 1:5
        subject_train_x = cat(1, subject_train_x, training{(idx-1)*5 + sec});
        subject_train_y = cat(1, subject_train_y, ALLLBL{(idx-1)*5 + sec});
    end
    train_x_all{idx} = subject_train_x;
    train_y_all{idx} = subject_train_y;
end

clear('training');

%% select testing
load('Testing.mat')
test_x_all = cell(10, 1);
for idx = 1 : 10
    subject_test_x = [];
    for sec = 1:5
        subject_test_x = cat(1, subject_test_x, testing{(idx-1)*5 + sec});
    end
    test_x_all{idx} = subject_test_x;
end

test = [];
test_y = Prediction>0.5;
for idx = 1:10
    test = cat(1, test, test_x_all{idx});
end
train = [];
train_y = [];
for idx = 1:16
    train = cat(1, train, train_x_all{idx});
    train_y = cat(1, train_y, train_y_all{idx});
end
clear('testing');

%% Select channels
load('RawData.mat');
test_y = [test_y, 1-test_y];
train_y = [train_y, 1-train_y];
% 3840, 48 (0.73), 38 (300: 0.72), 46 (250: 0.729), 5255
% 4648, 4838
% PCA: 48
channel = [48];
train_x = train(:, 1:260, channel);
test_x = test(:, 1:260, channel);
[epoch, time, chan] = size(train_x);
train_x = reshape(train_x, [epoch, time*chan]);
[epoch, time, chan] = size(test_x);
test_x = reshape(test_x, [epoch, chan*time]);
% remove the base line
a = mean(train_x, 2);
a = repmat(a, [1, time*chan]);
train_x = train_x-a;
a = mean(test_x, 2);
a = repmat(a, [1, time*chan]);
test_x = test_x-a;
% PCA
all_x = [train_x; test_x];
[coeff_x, score_x] = pca(all_x);
train_x = score_x(1:5440, 1:100);
test_x = score_x(5441:end, 1:100);
% Test
[Az_DNN_48_PCA, Pred_DNN_48_PCA, ~, ~] = test_example_DBN(train_x, ...
test_x, train_y, test_y, [1000], 0.0003, 1000, 80);

%% Training and testing
all_chan = [29   31    38    40    45    46    48    52    55];
channel = [48];
load('RawData.mat')
train_x = [];
test_x = [];
test_y = [];

for idx = 1:16
    train_y = cat(1, train_y, train_y_all{idx});
end
test_y = Prediction>0.5;
test_y = [test_y, 1-test_y];
train_y = [train_y, 1-train_y];

Az_DNN = cell(1, 9);
parfor idx = 1:9

    channel = all_chan(idx);
    
train_x = train(:, 1:260, channel);
test_x = test(:, 1:260, channel);

[epoch, time, chan] = size(train_x);
train_x = reshape(train_x, [epoch, time*chan]);
[epoch, time, chan] = size(test_x);
test_x = reshape(test_x, [epoch, chan*time]);

% remove the base line
a = mean(train_x, 2);
a = repmat(a, [1, time*chan]);
train_x = train_x-a;
a = mean(test_x, 2);
a = repmat(a, [1, time*chan]);
test_x = test_x-a;

[Az_DNN_, Pred_DNN, ~, ~] = test_example_DBN(train_x, ...
        test_x, train_y, test_y, [2000], 0.0003, 5000, 80);
    
end

%% training your data
Az_16_500_2000_9_001_01_5_5_sigm_4chan = zeros(16, 2000);

parfor i = 1:16
    
    [Az_DNN, ~] = TrainingDNN2( train_x_all, train_y_all, i, A, 2000, 0.001, 2000 );
    Az_16_500_2000_9_001_01_5_5_sigm_4chan(i, :) = Az_DNN;
    
end

%% run testing

[~, ~, pred_LDA_single] = classify(test_x, train_x, train_y(:, 1), 'diaglinear');
[Az_LDA_single, ~, ~, ~] = rocarea(pred_LDA_single, test_y(:, 2));

SVMModel = fitcsvm(train_x,train_y(:, 1),'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');
[~,scores] = predict(SVMModel,test_x);

[Az_DSN, pred_DSN, ~, ~] = test_example_DSN_keepW(train_x, ...
   test_x, train_y, test_y, 100, 0.0003, 4, 100);

[Az_DNN_6, pred_DNN, ~, ~] = test_example_DBN(train_x, ...
   test_x, train_y, test_y, 1000, 0.0003, 500, 100);

%% filtering

srate = 200;
locutoff = 1;
filtorder = 3*fix(srate/locutoff);
hicutoff = 20;
filtwts = fir1(filtorder, [locutoff, hicutoff]./(srate/2));

i = 12;
a = find(subject ~= train_idx(i));
train_x = current_data(a, 7:end);
train_y = Prediction(a);
a = find(subject == train_idx(i));
test_x = current_data(a, 7:end);
test_y = Prediction(a);
test_y = [test_y, 1-test_y];
train_y = [train_y, 1-train_y];

% FIR filter
train_x = filter(filtwts,1,train_x);
test_x = filter(filtwts,1,test_x);

% remove baseline
train = current_data(:, 7:end);
a = mean(train, 2);
a = repmat(a, [1, 261]);
train = train-a;

a = find(subject ~= train_idx(i));
train_x = train(a, :);
train_y = Prediction(a);
a = find(subject == train_idx(i));
test_x = train(a, :);
test_y = Prediction(a);

test_y = [test_y, 1-test_y];
train_y = [train_y, 1-train_y];


% run test
[Az_DSN_Small, pred_DSN, ~, ~] = test_example_DBN(train_x, ...
        test_x, train_y, test_y, 2000, 0.00001, 2000, 100);


test = test_data(:, 6:end);
a = mean(test, 2);
a = repmat(a, [1, 261]);
test = test-a;

Prediction1 = Prediction1>0.5;
Prediction1 = [Prediction1 1-Prediction1];
Prediction = [Prediction 1-Prediction];

csvwrite('test_ref.csv',test);
csvwrite('train_ref.csv',train);


%% run cross subject
train_idx = [2,6,7,11,12,13,14,16,17,18,20,21,22,23,24,26];

% a = find(subject == 14);
subject = current_data(:, 3);
% current_data(a, :) = [];

%% run cross subject
Az_16_2000_001_01_9_5_sigm_1000 = zeros(16, 1000);
parfor i = 1:16
    
%     a = find(subject ~= train_idx(i));
%     train_x = current_data(a, 7:end);
%     train_y = Prediction(a);
%     a = find(subject == train_idx(i));
%     test_x = current_data(a, 7:end);
%     test_y = Prediction(a);
    
    a = find(subject ~= train_idx(i));
    train_x = train(a, :);
    train_y = Prediction(a);
    a = find(subject == train_idx(i));
    test_x = train(a, :);
    test_y = Prediction(a);

    test_y = [test_y, 1-test_y];
    train_y = [train_y, 1-train_y];
    
    [Az_DSN, pred_DSN, ~, ~] = test_example_DBN(train_x, ...
        test_x, train_y, test_y, 2000, 0.0001, 1000, 100);
    Az_16_2000_001_01_9_5_sigm_1000(i, :) = Az_DSN;
    
end

%% run PCA
current_data = csvread('train_cz.csv', 1, 0);
test_data = csvread('test_cz.csv', 1, 0);
load('Label.mat');

train = current_data(:, 7:end);
test = test_data(:, 6:end);
[coeff, score] = princomp(train);
AAA = bsxfun(@minus, test, mean(test))*coeff;

csvwrite('test_pca.csv',AAA(:, 1:100));
csvwrite('train_pca.csv',score(:, 1:100));

%% bag-of-words
% current_data = csvread('train_cz.csv', 1, 0);
subject = current_data(:, 3);
train_idx = [2,6,7,11,12,13,14,16,17,18,20,21,22,23,24,26];
data = current_data(:, 7:end);
data_kmeans =reshape(data, [5440, 9, 29]);
data_k = reshape(data_kmeans, [5440*9, 29]);
idx = kmeans(data_k, 30, 'MaxIter',1000);
idx_k = reshape(idx, [5440, 9]);

parfor i = 1:16
    a = find(subject ~= train_idx(i));
    train_x = idx_k(a,:);
    train_y = Prediction(a);
    a = find(subject == train_idx(i));
    test_x = idx_k(a,:);
    test_y = Prediction(a);

    NBModel2 = fitNaiveBayes(train_x,train_y);
    pred = posterior(NBModel2,test_x);

    [~, ~, ~, Az(i) ] = perfcurve(test_y(:, 1), pred(:, 1), 1);
end
