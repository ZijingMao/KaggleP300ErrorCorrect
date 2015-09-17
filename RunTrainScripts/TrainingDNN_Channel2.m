function [ Az_DNN, Pred_DNN ] = TrainingDNN_Channel2( train_x_all, train_y_all, idx, channel, hidden, iters )
%TRAININGDNN Summary of this function goes here

train_x = [];
train_y = [];
test_x = [];
test_y = [];
for i = 1:length(train_x_all)
    if idx == i
        test_x = train_x_all{idx};
        test_y = train_y_all{idx};
    else
        train_x = cat(1, train_x, train_x_all{i});
        train_y = cat(1, train_y, train_y_all{i});
    end
end

test_y = [test_y, 1-test_y];
train_y = [train_y, 1-train_y];

train_x = train_x(:, 1:260, channel);
test_x = test_x(:, 1:260, channel);

[epoc, time, chan] = size(train_x);
train_x = reshape(train_x, [epoc, time*chan]);
[epoc, time, chan] = size(test_x);
test_x = reshape(test_x, [epoc, time*chan]);

% remove the base line
a = mean(train_x, 2);
a = repmat(a, [1, time*chan]);
train_x = train_x-a;
a = mean(test_x, 2);
a = repmat(a, [1, time*chan]);
test_x = test_x-a;

[Az_DNN, Pred_DNN, ~, ~] = test_example_DBN(train_x, ...
        test_x, train_y, test_y, hidden, 0.001, iters, 100);

end

