function [er, bad, a, b] = nntest(nn, x, y, stack)
    [labels, a] = nnpredict(nn, x, stack);
    [~, expected] = max(y,[],2);
    bad = find(labels ~= expected);    
    er = numel(bad) / size(x, 1);
    b = expected;
end
