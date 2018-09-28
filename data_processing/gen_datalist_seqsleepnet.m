% This script list of training and testing data files which will be
% processed by SeqSleepNet, and two baselines E2E-ARNN and Multitask E2E-ARNN in tensorflow (for efficiency)

clear all
close all
clc

load('./data_split_eval.mat');

mat_path = './mat/';
Nfold = 20;

%% EEG
listing = dir([mat_path, '*_seqsleepnet_eeg.mat']);
tf_path = './tf_data/seqsleepnet_eval_eeg/';
if(~exist(tf_path, 'dir'))
    mkdir(tf_path);
end

for s = 1 : Nfold

    disp(['Fold: ', num2str(s),'/',num2str(Nfold)]);
    
	train_s = train_sub{s};
    eval_s = eval_sub{s};
    test_s = test_sub{s};
    
    train_filename = [tf_path, 'train_list_n', num2str(s),'.txt'];
    fid = fopen(train_filename,'wt');
    for i = 1 : numel(train_s)
        sname = listing(train_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../data_processing/mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    test_filename = [tf_path, 'test_list_n', num2str(s),'.txt'];
    fid = fopen(test_filename,'wt');
    for i = 1 : numel(test_s)
        sname = listing(test_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../data_processing/mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    eval_filename = [tf_path, 'eval_list_n', num2str(s),'.txt'];
    fid = fopen(eval_filename,'wt');
    for i = 1 : numel(eval_s)
        sname = listing(eval_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../data_processing/mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
end


%% EOG
listing = dir([mat_path, '*_seqsleepnet_eog.mat']);
tf_path = './tf_data/seqsleepnet_eval_eog/';
if(~exist(tf_path, 'dir'))
    mkdir(tf_path);
end

for s = 1 : Nfold

    disp(['Fold: ', num2str(s),'/',num2str(Nfold)]);
    
	train_s = train_sub{s};
    eval_s = eval_sub{s};
    test_s = test_sub{s};
    
    train_filename = [tf_path, 'train_list_n', num2str(s),'.txt'];
    fid = fopen(train_filename,'wt');
    for i = 1 : numel(train_s)
        sname = listing(train_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../data_processing/mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    test_filename = [tf_path, 'test_list_n', num2str(s),'.txt'];
    fid = fopen(test_filename,'wt');
    for i = 1 : numel(test_s)
        sname = listing(test_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../data_processing/mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    eval_filename = [tf_path, 'eval_list_n', num2str(s),'.txt'];
    fid = fopen(eval_filename,'wt');
    for i = 1 : numel(eval_s)
        sname = listing(eval_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../data_processing/mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
end


%% EMG
listing = dir([mat_path, '*_seqsleepnet_emg.mat']);
tf_path = './tf_data/seqsleepnet_eval_emg/';
if(~exist(tf_path, 'dir'))
    mkdir(tf_path);
end

for s = 1 : Nfold

    disp(['Fold: ', num2str(s),'/',num2str(Nfold)]);
    
	train_s = train_sub{s};
    eval_s = eval_sub{s};
    test_s = test_sub{s};
    
    train_filename = [tf_path, 'train_list_n', num2str(s),'.txt'];
    fid = fopen(train_filename,'wt');
    for i = 1 : numel(train_s)
        sname = listing(train_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../data_processing/mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    test_filename = [tf_path, 'test_list_n', num2str(s),'.txt'];
    fid = fopen(test_filename,'wt');
    for i = 1 : numel(test_s)
        sname = listing(test_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../data_processing/mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    eval_filename = [tf_path, 'eval_list_n', num2str(s),'.txt'];
    fid = fopen(eval_filename,'wt');
    for i = 1 : numel(eval_s)
        sname = listing(eval_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['../data_processing/mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
end




