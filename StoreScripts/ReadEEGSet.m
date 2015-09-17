function ReadEEGSet
%READEEGSET Summary of this function goes here
%   Detailed explanation goes here
clc
addpath(genpath('/home/zijing.mao/Kaggle'));

%% Start training part
cd(['/home/zijing.mao/Kaggle/KaggleBCIStandardLevel2/train']);
filelist2cell = @(str) extractfield( (dir(str)), 'name' );
eeg_dataset_list = filelist2cell(pwd)';
eeg_dataset_list(1:2) = [];
cd .. 

disp(['Duration is 1s, start epoching...']);

ALLEEG = pop_loadset('filename', eeg_dataset_list, 'filepath', ['/home/zijing.mao/Kaggle/KaggleBCIStandardLevel2/train/']);

for i = 1 : length(ALLEEG)
    ALLEEG(i).nbchan = 56;
    ALLEEG(i).data(56, :) = [];
    ALLEEG(i).chanlocs(56) = [];
end

% extract epoch
A = [0 
    1];
A = mat2cell(A, [1 1]);
for i = 1 : length(ALLEEG)
    eeg_struct(i) = pop_epoch(ALLEEG(i), A, [0 1.3]);
    eeg_struct(i) = pop_rmbase(eeg_struct(i), []);
end
clear('ALLEEG');

% extract data
ALLEEG = cell(length(eeg_struct), 1);
ALLLBL = cell(length(eeg_struct), 1);
for j = 1:length(eeg_struct)
    EEG = eeg_struct(j);
    len = EEG.trials;
    lbl = zeros(len, 1);
    label = cell(len, 1);
    for i = 1 : len
        label{i} = EEG.epoch(1, i).eventtype(1);
        if strcmp(label(i), '1')
            lbl(i) = 1;
        else
            lbl(i) = 0;
        end    
    end
    data = EEG.data;
    ALLEEG{j} = data;
    ALLLBL{j} = lbl;
end
clear('eeg_struct');

save('TrainingAfterRef.mat', 'ALLEEG', 'ALLLBL');


%% Start training part
cd(['/home/zijing.mao/Kaggle/KaggleBCIStandardLevel2/test']);
filelist2cell = @(str) extractfield( (dir(str)), 'name' );
eeg_dataset_list = filelist2cell(pwd)';
eeg_dataset_list(1:2) = [];
cd .. 

disp(['Duration is 1s, start epoching...']);

ALLEEG = pop_loadset('filename', eeg_dataset_list, 'filepath', ['/home/zijing.mao/Kaggle/KaggleBCIStandardLevel2/test/']);

for i = 1 : length(ALLEEG)
    ALLEEG(i).nbchan = 56;
    ALLEEG(i).data(56, :) = [];
    ALLEEG(i).chanlocs(56) = [];
end

% extract epoch
A = ['FB'];
A = mat2cell(A, [1]);
for i = 1 : length(ALLEEG)
    eeg_struct(i) = pop_epoch(ALLEEG(i), A, [0 1.3]);
    eeg_struct(i) = pop_rmbase(eeg_struct(i), []);
end
clear('ALLEEG');

% extract data
ALLEEG = cell(length(eeg_struct), 1);
for j = 1:length(eeg_struct)
    EEG = eeg_struct(j);
    data = EEG.data;
    ALLEEG{j} = data;
end
clear('eeg_struct');

save('TestingAfterRef.mat', 'ALLEEG');

end

