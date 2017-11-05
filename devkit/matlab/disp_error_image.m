function D_err = disp_error_image (D_gt,D_est,tau,dilate_radius)

if nargin==3
  dilate_radius = 1;
end

[E,D_val] = disp_error_map (D_gt,D_est);
E = min(E/tau(1),(E./abs(D_gt))/tau(2));

cols = error_colormap();

D_err = zeros([size(D_gt) 3]);
for i=1:size(cols,1)
  [v,u] = find(D_val > 0 & E >= cols(i,1) & E <= cols(i,2));
  D_err(sub2ind(size(D_err),v,u,1*ones(length(v),1))) = cols(i,3);
  D_err(sub2ind(size(D_err),v,u,2*ones(length(v),1))) = cols(i,4);
  D_err(sub2ind(size(D_err),v,u,3*ones(length(v),1))) = cols(i,5);
end

D_err = imdilate(D_err,strel('disk',dilate_radius));
