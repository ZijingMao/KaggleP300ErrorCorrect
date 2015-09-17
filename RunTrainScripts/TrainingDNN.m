function [ Az_DNN, Pred_DNN ] = TrainingDNN( train_x_all, train_y_all, idx )
%TRAININGDNN Summary of this function goes here

channel = 29;

train_x = [];
train_y = [];
test_x = [];
test_y = [];
for i = 1:length(train_x_all)
    if idx == i
        test_x = train_x_all{idx};
        test_y = train_y_all{idx};
    else
        train_x = cat(3, train_x, train_x_all{i});
        train_y = cat(1, train_y, train_y_all{i});
    end
end

test_y = [test_y, 1-test_y];
train_y = [train_y, 1-train_y];

train_x = squeeze(train_x(channel, :, :))';
test_x = squeeze(test_x(channel, :, :))';

[Az_DNN, Pred_DNN, ~, ~] = test_example_DBN(train_x, ...
        test_x, train_y, test_y, 2000, 0.0006, 1000, 100);

end

