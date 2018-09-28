clear all
close all
clc

raw_data_path = './raw_data/';
mat_path = './mat/';
if(~exist(mat_path, 'dir'))
    mkdir(mat_path);
end

fs = 100; % sampling frequency
win_size  = 2;
overlap = 1;
nfft = 2^nextpow2(win_size*fs);

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
    N = size(data, 1);
    X = zeros(N, 29, nfft/2+1);
    eeg_epochs = squeeze(data(:,:,1)); % eeg channel
    for k = 1 : size(eeg_epochs, 1)
        if(mod(k,100) == 0)
            disp([num2str(k),'/',num2str(size(eeg_epochs, 1))]);
        end
        [Xk,~,~] = spectrogram(eeg_epochs(k,:), hamming(win_size*fs), overlap*fs, nfft);
        % log magnitude spectrum
        Xk = 20*log10(abs(Xk));
        Xk = Xk';
        X(k,:,:) = Xk;
    end
    X = single(X);
    y = single(y);
    label=single(label);
    save([mat_path, filename,'_seqsleepnet_eeg.mat'], 'X', 'label', 'y', '-v7.3');
    clear X
    
    %% EOG
    X= zeros(N, 29, nfft/2+1);
    eeg_epochs = squeeze(data(:,:,2)); % eog channel
    for k = 1 : size(eeg_epochs, 1)
        if(mod(k,100) == 0)
            disp([num2str(k),'/',num2str(size(eeg_epochs, 1))]);
        end
        [Xk,~,~] = spectrogram(eeg_epochs(k,:), hamming(win_size*fs), overlap*fs, nfft);
        % log magnitude spectrum
        Xk = 20*log10(abs(Xk));
        Xk = Xk';
        X(k,:,:) = Xk;
    end
    X = single(X);
    y = single(y);
    label=single(label);
    save([mat_path, filename,'_seqsleepnet_eog.mat'], 'X', 'label', 'y', '-v7.3');
    clear X
    
    %% EMG
    X= zeros(N, 29, nfft/2+1);
    eeg_epochs = squeeze(data(:,:,3)); % emg channel
    for k = 1 : size(eeg_epochs, 1)
        if(mod(k,100) == 0)
            disp([num2str(k),'/',num2str(size(eeg_epochs, 1))]);
        end
        [Xk,~,~] = spectrogram(eeg_epochs(k,:), hamming(win_size*fs), overlap*fs, nfft);
        % log magnitude spectrum
        Xk = 20*log10(abs(Xk));
        Xk = Xk';
        X(k,:,:) = Xk;
    end
    X = single(X);
    y = single(y);
    label=single(label);
    save([mat_path, filename,'_seqsleepnet_emg.mat'], 'X', 'label', 'y', '-v7.3');
    clear X y label
end