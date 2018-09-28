function prob = softmax(score)
    score_ = score;    
    score_ = exp(score_);
    prob = zeros(size(score_));
    for i = 1 : size(score_,1)
        prob(i,:) = score_(i,:)/sum(score_(i,:));
    end
end