function Store2Mat(duration)
%STORE2MAT Store everything into mat file

clc
addpath(genpath('/home/zijing.mao/Kaggle'));

%% Start training part
cd(['/home/zijing.mao/Kaggle/train']);
filelist2cell = @(str) extractfield( (dir(str)), 'name' );
eeg_dataset_list = filelist2cell(pwd)';
eeg_dataset_list(1:2) = [];

disp(['Duration is 1s, start epoching...']);

training = cell(1, length(eeg_dataset_list));
% Load files
for idx = 1 : length(eeg_dataset_list)
    current_data = csvread(eeg_dataset_list{idx}, 1, 0);
    event = current_data(:, end);
    data = current_data(:, 2:end-1);
    event_location = find(event == 1);
    data_epoch = zeros(length(event_location), duration, size(data, 2));
    for epoch = 1 : length(event_location)
        data_epoch(epoch, :, :) = data(event_location(epoch):event_location(epoch)+duration-1, :);
    end
    % Store into a cell file
    training{idx} = data_epoch;
    disp(['Finishing part ' num2str(idx)]);
end
save('Training.mat', 'eeg_dataset_list', 'training');
disp(['Stored file.']);
cd ..

clear('training', 'eeg_dataset_list');

%% Start testing part
cd(['/home/zijing.mao/Kaggle/test']);
filelist2cell = @(str) extractfield( (dir(str)), 'name' );
eeg_dataset_list = filelist2cell(pwd)';
eeg_dataset_list(1:2) = [];

disp(['Duration is 1s, start epoching...']);

testing = cell(1, length(eeg_dataset_list));
% Load files
for idx = 1 : length(eeg_dataset_list)
    current_data = csvread(eeg_dataset_list{idx}, 1, 0);
    event = current_data(:, end);
    data = current_data(:, 2:end-1);
    event_location = find(event == 1);
    data_epoch = zeros(length(event_location), duration, size(data, 2));
    for epoch = 1 : length(event_location)
        data_epoch(epoch, :, :) = data(event_location(epoch):event_location(epoch)+duration-1, :);
    end
    % Store into a cell file
    testing{idx} = data_epoch;
    disp(['Finishing part ' num2str(idx)]);
end
save('Testing.mat', 'eeg_dataset_list', 'testing');
disp(['Stored file.']);
cd ..

end

