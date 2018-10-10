function [acc, kappa, f1, sens, spec, classwise_sens, classwise_sel, C] = eval_seqsleepnet(seq_len)

    addpath('./evaluation/');
    
    if(nargin == 0)
        seq_len = 20;
    end
    
    Ncat = 5;
    
    Nfold = 20;
    yh = cell(Nfold,1);
    yt = cell(Nfold,1);
    mat_path = './data_processing/mat/';
    listing = dir([mat_path, '*_seqsleepnet_eeg.mat']);
    load('./data_processing/data_split_eval.mat');

    for fold = 1 : Nfold
        fold
        
        % ground truth
        test_s = test_sub{fold};
        sample_size = zeros(numel(test_s), 1);
        for i = 1 : numel(test_s)
            i
            sname = listing(test_s(i)).name;
            load([mat_path,sname], 'label');
            % handle the different here
            sample_size(i) = numel(label) -  (seq_len - 1); 
            yt{fold} = [yt{fold}; double(label)];
        end
        
        % load network output and perform probabilistic aggregation
        load(['./tensorflow_net/SeqSleepNet/seqsleepnet_sleep_nfilter32_seq',num2str(seq_len),'_dropout0.75_nhidden64_att64_3chan/n',num2str(fold),'/test_ret.mat']);
        score_ = cell(1,seq_len);
        for n = 1 : seq_len
            score_{n} = softmax(squeeze(score(n,:,:)));
        end
        score = score_;
        clear score_;

        for i = 1 : numel(test_s)
            start_pos = sum(sample_size(1:i-1)) + 1;
            end_pos = sum(sample_size(1:i-1)) + sample_size(i);
            score_i = cell(1,seq_len);
            for n = 1 : seq_len
                score_i{n} = score{n}(start_pos:end_pos, :);
                N = size(score_i{n},1);
                
                score_i{n} = [ones(seq_len-1,Ncat); score{n}(start_pos:end_pos, :)];
                score_i{n} = circshift(score_i{n}, -(seq_len - n), 1);
            end

            fused_score = log(score_i{1});
            for n = 2 : seq_len
                fused_score = fused_score + log(score_i{n});
            end

            yhat = zeros(1,size(fused_score,1));
            for k = 1 : size(fused_score,1)
                [~, yhat(k)] = max(fused_score(k,:));
            end

            yh{fold} = [yh{fold}; double(yhat')];
        end
    end
    yh = cell2mat(yh);
    yt = cell2mat(yt);
    
    [acc, kappa, f1, sens, spec] = calculate_overall_metrics(yt, yh);
    [classwise_sens, classwise_sel]  = calculate_classwise_sens_sel(yt, yh);
    C = confusionmat(yt, yh);
end
