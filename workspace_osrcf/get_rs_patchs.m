function IMG_R = get_rs_patchs(image, center, origion_size, target_sz, ras, method)

scalar = origion_size./target_sz;
% use ba_interp2
w = target_sz(1);
h = target_sz(2);
[Dx, Dy] = meshgrid(-floor(w/2):floor((w-1)/2), -floor(h/2):floor((h-1)/2));

num = numel(ras);
RDx = zeros(h, w*num);
RDy = zeros(h, w*num);
for i = 1:num
    rotate_angle = ras(i);
    %     R(2*i-1:2i, :) = [scalar(1) * cos(rotate_angle) scalar(1) * sin(rotate_angle); -scalar(2) * sin(rotate_angle) scalar(2) * cos(rotate_angle)];
    cosr = cos(rotate_angle); sinr = sin(rotate_angle);
    Ri = [scalar(1) * cosr scalar(1) * sinr; -scalar(2) * sinr scalar(2) * cosr];
    RDi = Ri * [Dx(:)'; Dy(:)'];
    RDx(:, w*(i-1)+1 : w*i) = reshape(RDi(1,:), size(Dx))+center(1);
    RDy(:, w*(i-1)+1 : w*i) = reshape(RDi(2,:), size(Dy))+center(2);
end

% nearest, linear, or cubic.
IMG_R = ba_interp2(single(image), single(RDx), single(RDy), method);
%
IMG_R(IMG_R<0) = 0;
IMG_R(IMG_R>1) = 1;
end