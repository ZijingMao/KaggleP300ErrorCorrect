subplot(2, 2, 1)
imagesc(squeeze(EEG.data(1, :, 1:56))')
subplot(2, 2, 2)
imagesc(squeeze(EEG.data(2, :, 1:56))')
subplot(2, 2, 3)
imagesc(squeeze(EEG.data(3, :, 1:56))')
subplot(2, 2, 4)
imagesc(squeeze(EEG.data(4, :, 1:56))')

%% get the best performance of preprocessed subject
Channel = 8;
for Group = 1 : 7
    name = ['AfterPreprocessChannel' num2str((Group-1)*Channel+1) 'to' ...
        num2str(Group*Channel) '.mat'];
    load(name);
    for idx = 1 : Channel
        tmp = Az_16_500_1000_001_9_01_5_5_sigm{idx};
        tmp = mean(tmp);
        AfterPreprocess((Group-1)*8+idx) = tmp(end);
         subplot(7, 8, (Group-1)*8+idx)
         plot(tmp);
    end
end

%% get the best performance of raw subject
Channel = 8;
for Group = 1 : 7
    name = ['RawDataChannel' num2str((Group-1)*Channel+1) 'to' ...
        num2str(Group*Channel) '.mat'];
    load(name);
    for idx = 1 : Channel
        tmp = Az_16_500_1000_001_9_01_5_5_sigm{idx};
        tmp = mean(tmp);
        RawData((Group-1)*8+idx) = tmp(end);
         subplot(7, 8, (Group-1)*8+idx)
         plot(tmp);
    end
end

%% plot 4 * 4
for i = 1:3
    for j = 1:3
        subplot(3, 3, (i-1)*3+j)
        plot(Az_DNN{(i-1)*3+j});
    end
end

%% plot 4 * 4
for i = 1:4
    for j = 1:4
        subplot(4, 4, (i-1)*4+j)
        plot(Az_16_500_001_9_01_5_5_sigm_4chan((i-1)*4+j, :));
    end
end

%% plot 8 * 8
data_x = reshape(X, [5440, 100, 56]);
target = find(Y == 1);
nontarget = find(Y == 0);
a = find(idx == 2);
for i = 1:8
    for j = 1:8
        subplot(8, 8, (i-1)*8+j)
        %plot(test_x(a((i-1)*8+j), :))
        if (i-1)*8+j < 28
            %imagesc(squeeze(data_x(target((i-1)*8+j), :, :))');
            plot(n2((i-1)*8+j, :)');
        else
            %imagesc(squeeze(data_x(nontarget((i-1)*8+j-32), :, :))');
            plot(t2((i-1)*8+j-27, :)');
        end
    end
end

%% plot single signal
n1 = train_x(train_y(:, 1)==0, :);
t1 = train_x(train_y(:, 1)==1, :);
n2 = test_x(test_y(:, 1)==0, :);
t2 = test_x(test_y(:, 1)==1, :);

subplot(2, 1, 1); plot(n2(8, :)');
subplot(2, 1, 2); plot(t2(8, :)');


subplot(2, 1, 1);plot(mean(n1), 'Color', 'b');hold on
plot(mean(t1), 'Color', 'r')
subplot(2, 1, 2);plot(mean(n2), 'Color', 'b');hold on
plot(mean(t2), 'Color', 'r')