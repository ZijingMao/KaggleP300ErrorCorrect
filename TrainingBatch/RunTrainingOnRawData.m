function RunTrainingOnRawData(Group, Channel)
%RUNTRAINING Summary of this function goes here
clc
addpath(genpath('/home/zijing.mao/Kaggle'));

training = [];
load('Training.mat');

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

Channel = str2num(Channel);
Group = str2num(Group);

%% training your data
Az_16_500_1000_001_9_01_5_5_sigm = cell(1, Channel);
iters = 1000;
hidden = 1000;

for j = 1:Channel

    Az = zeros(16, iters);
    parfor i = 1:16

        [Az_DNN, ~] = TrainingDNN_Channel2( train_x_all, train_y_all, i, (Group-1)*j+j, hidden, iters );
        Az(i, :) = Az_DNN;

    end
    Az_16_500_1000_001_9_01_5_5_sigm{j} = Az;
    
end

save(['RawDataChannel' num2str((Group-1)*Channel+1) 'to' ...
    num2str(Group*Channel) '.mat'], 'Az_16_500_1000_001_9_01_5_5_sigm')

end

