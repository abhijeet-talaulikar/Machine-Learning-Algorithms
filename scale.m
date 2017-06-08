function data = scale(m)
    minmat = repmat(min(m), size(m,1), 1);
    maxmat = repmat(max(m), size(m,1), 1);
    data = (m - minmat) ./ (maxmat - minmat);
end