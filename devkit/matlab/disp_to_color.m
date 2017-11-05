function I = disp_to_color (D,max_disp)
% computes color representation of disparity map
% code adapted from Oliver Woodford's sc.m
% max_disp optionally specifies the scaling factor

D = double(D);

if nargin==1
  max_disp = max(D(:));
end

I = disp_map(min(D(:)/max_disp,1));
I = reshape(I, [size(D,1) size(D,2) 3]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function I = disp_map(I)

map = [0 0 0 114; 0 0 1 185; 1 0 0 114; 1 0 1 174; ...
       0 1 0 114; 0 1 1 185; 1 1 0 114; 1 1 1 0];

bins  = map(1:end-1,4);
cbins = cumsum(bins);
bins  = bins./cbins(end);
cbins = cbins(1:end-1) ./ cbins(end);
ind   = min(sum(repmat(I(:)', [6 1]) > repmat(cbins(:), [1 numel(I)])),6) + 1;
bins  = 1 ./ bins;
cbins = [0; cbins];

I = (I-cbins(ind)) .* bins(ind);
I = min(max(map(ind,1:3) .* repmat(1-I, [1 3]) + map(ind+1,1:3) .* repmat(I, [1 3]),0),1);

