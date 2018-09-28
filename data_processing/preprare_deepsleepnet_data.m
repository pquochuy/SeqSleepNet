% As the end-to-end DeepSleepNet receive raw signals as input, its data is
% handled separately
clear all
close all
clc

raw_data_path = './raw_data/';
mat_path = './mat/';
if(~exist(mat_path, 'dir'))
    mkdir(mat_path);
end

% list all subjects
listing = dir([raw_data_path, 'SS*']);

for i = 1 : numel(listing)
	disp(listing(i).name)
    
    load([raw_data_path, listing(i).name]);
    [~, filename, ~] = fileparts(listing(i).name);

    % label and one-hot encoding
    y = double(labels);
    label = zeros(size(y,1),1);
    for k = 1 : size(y,1)
        [~, label(k)] = find(y(k,:));
    end
    clear labels
    
    %% EEG
    X = single(data(:,:,1));
    y = single(y);
    label=single(label);
    save([mat_path, filename,'_deepsleepnet_eeg.mat'], 'X', 'label', 'y', '-v7.3');
    clear X
    
    %% EOG
    X = single(data(:,:,2));
    y = single(y);
    label=single(label);
    save([mat_path, filename,'_deepsleepnet_eog.mat'], 'X', 'label', 'y', '-v7.3');
    clear X
    
    %% EMG
    X = single(data(:,:,3));
    y = single(y);
    label=single(label);
    save([mat_path, filename,'_deepsleepnet_emg.mat'], 'X', 'label', 'y', '-v7.3');
    clear X y label
end