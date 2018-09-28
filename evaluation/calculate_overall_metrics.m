% This fucntion calculate overall performance metrics
function [acc, kappa, f1, sens, spec] = calculate_overall_metrics(y, yhat)
    
    Ncat = numel(unique(y));
    
    acc = sum(y == yhat)/numel(yhat);
    kappa = kappaindex(yhat,y,Ncat);
    
    f1 = zeros(Ncat,1);
    sens = zeros(Ncat,1);
    spec = zeros(Ncat,1);
    for cl = 1 : Ncat
        [f1(cl), sens(cl), spec(cl)]  = classwise_metrics(y,yhat,cl);
    end
    f1 = mean(f1);
    sens = mean(sens);
    spec = mean(spec);
end


function [f1, sens, spec] = classwise_metrics(y,yhat,class)
    ind = (y == class);
    y(~ind) = 0;
    y(ind) = 1;
    
    ind = (yhat == class);
    yhat(~ind) = 0;
    yhat(ind) = 1;

    bin_metrics = binary_metrics(y,yhat);
    f1 = bin_metrics(6);
    sens = bin_metrics(2);
    spec = bin_metrics(3);
end


function bin_metrics = binary_metrics(y,yhat)
    idx = (y()==1);

    p = length(y(idx));
    n = length(y(~idx));
    N = p+n;

    tp = sum(y(idx)==yhat(idx));
    tn = sum(y(~idx)==yhat(~idx));
    fp = n-tn;
    fn = p-tp;

    tp_rate = tp/p;
    tn_rate = tn/n;

    accuracy = (tp+tn)/N;
    sensitivity = tp_rate;
    specificity = tn_rate;
    precision = tp/(tp+fp);
    recall = sensitivity;
    f_measure = 2*((precision*recall)/(precision + recall));
    gmean = sqrt(tp_rate*tn_rate);

    bin_metrics = [accuracy sensitivity specificity precision recall f_measure gmean];
end
