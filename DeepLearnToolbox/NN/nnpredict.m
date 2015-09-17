function [labels, a] = nnpredict(nn, x, stack)
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)), stack);
    nn.testing = 0;
    
    [~, i] = max(nn.a{end},[],2);
    labels = i;
    a = nn.a{end};
end
