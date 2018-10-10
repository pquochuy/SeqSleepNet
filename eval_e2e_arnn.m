function [acc, kappa, f1, sens, spec, classwise_sens, classwise_sel, C] = eval_e2e_arnn()

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
            sname = listing(test_s(i)).name;
            load([mat_path,sname], 'label');
            yt{fold} = [yt{fold}; double(label)];
        end
        
        load(['./tensorflow_net/E2E-ARNN/arnn_sleep_nfilter32_dropout0.75_nhidden64_att32_3chan/n',num2str(fold),'/test_ret.mat']);
        yh{fold} = yhat';
    end
    yh = cell2mat(yh);
    yt = cell2mat(yt);
    
    [acc, kappa, f1, sens, spec] = calculate_overall_metrics(yt, yh);
    [classwise_sens, classwise_sel]  = calculate_classwise_sens_sel(yt, yh);
    C = confusionmat(yt, yh);
end
