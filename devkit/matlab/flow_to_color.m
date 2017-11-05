function I = flow_to_color (F,max_flow)
% computes color representation of optical flow field
% code adapted from Oliver Woodford's sc.m
% max_flow optionally specifies the scaling factor

F = double(F);

F_du  = shiftdim(F(:,:,1));
F_dv  = shiftdim(F(:,:,2));
F_val = shiftdim(F(:,:,3));

if nargin==1
  max_flow = max([abs(F_du(F_val==1)); abs(F_dv(F_val==1))]);
else
  max_flow = max(max_flow,1);
end

F_mag = sqrt(F_du.*F_du+F_dv.*F_dv);
F_dir = atan2(F_dv,F_du);

I = flow_map(F_mag(:),F_dir(:),F_val(:),max_flow,8);
I = reshape(I, [size(F_du,1) size(F_du,2) 3]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function I = flow_map(F_mag,F_dir,F_val,max_flow,n)

I(:,1) = mod(F_dir/(2*pi),1);
I(:,2) = F_mag * n / max_flow;
I(:,3) = n - I(:,2);
I(:,[2 3]) = min(max(I(:,[2 3]),0),1);
I = hsv2rgb(reshape(I,[],1,3));

I(:,:,1) = I(:,:,1) .* F_val;
I(:,:,2) = I(:,:,2) .* F_val;
I(:,:,3) = I(:,:,3) .* F_val;
