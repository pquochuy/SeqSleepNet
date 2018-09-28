function [sens, sel]  = calculate_classwise_sens_sel(y, yhat)
    Ncat = numel(unique(y));
    sens = zeros(Ncat,1);
    sel = zeros(Ncat,1);
    for cl = 1 : Ncat
        ind = (yhat == cl);
        true_det = sum(y(ind) == cl);
        num_ref = sum(y == cl);
        num_det = sum(ind);
        sens(cl) = true_det/num_ref;
        sel(cl) = true_det/num_det;
    end
end