function RunTraining
%RUNTRAINING Summary of this function goes here
clc
addpath(genpath('/home/zijing.mao/Kaggle'));

ALLEEG = [];
ALLLBL = [];
load('TrainingAfterRef');

%% load testing data set
load('TestingAfterRef');
test_x_all = cell(10, 1);
for idx = 1 : 10
    subject_test_x = [];
    for sec = 1:5
        subject_test_x = cat(3, subject_test_x, ALLEEG{(idx-1)*5 + sec});
    end
    test_x_all{idx} = subject_test_x;
end

test = [];
for idx = 1:10
    test = cat(3, test, test_x_all{idx});
end
train = [];
train_y = [];
for idx = 1:16
    train = cat(3, train, train_x_all{idx});
    train_y = cat(1, train_y, train_y_all{idx});
end

B = [5     6    11    18    22    33    41    43];
Prediction = Prediction>0.5;
% A = [ 5 6 33 41 ];
Az_DNN = cell(1, 8);
parfor idx = 1:8
    A = B(idx);
train_x = train(A, :, :);
[chan, time, epoch] = size(train_x);
train_x = reshape(train_x, [chan*time, epoch])';

test_x = test(A, :, :);
[chan, time, epoch] = size(test_x);
test_x = reshape(test_x, [chan*time, epoch])';

a = mean(train_x, 2);
a = repmat(a, [1, time*chan]);
train_x = train_x-a;
a = mean(test_x, 2);
a = repmat(a, [1, time*chan]);
test_x = test_x-a;

[Az_DNN{idx}, ~, ~, ~] = test_example_DBN(train_x, ...
        test_x, [train_y 1-train_y], [Prediction 1-Prediction], 2000, 0.0003, 2000, 80);
end
    
%% running just single channel
train_x_all = cell(16, 1);
train_y_all = cell(16, 1);
for idx = 1 : 16
    subject_train_x = [];
    subject_train_y = [];
    for sec = 1:5
        subject_train_x = cat(3, subject_train_x, ALLEEG{(idx-1)*5 + sec});
        subject_train_y = cat(1, subject_train_y, ALLLBL{(idx-1)*5 + sec});
    end
    train_x_all{idx} = subject_train_x;
    train_y_all{idx} = subject_train_y;
end

clear('ALLEEG', 'ALLLBL');

%% training your data
Az_16_500_001_9_01_5_5_sigm_4chan = zeros(16, 2000);
parfor i = 1:16
    
    [Az_DNN, ~] = TrainingDNN_Channel( train_x_all, train_y_all, i, [ 5 6 11 18 22 33 41 43], [2000], 2000 );
    Az_16_500_001_9_01_5_5_sigm_4chan(i, :) = Az_DNN;
    
end

save('matlab1.mat', 'Az_16_500_500_500_001_9_01_5_5_sigm_4chan')

end

