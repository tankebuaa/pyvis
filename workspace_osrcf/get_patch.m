function IMG_R = get_patch(image, center, origion_size, target_sz, rotate_angle, method)

if 1-cos(rotate_angle)<eps
%     disp('use mexresize');
    % use mexResize
    w = origion_size(1);
    h = origion_size(2);
    % extract indexes
    xs = floor(center(1)) + (1:w) - floor(w/2);
    ys = floor(center(2)) + (1:h) - floor(h/2);
    % indexes outside of image: use border pixels
    xs(xs < 1) = 1;
    ys(ys < 1) = 1;
    xs(xs > size(image,2)) = size(image,2);
    ys(ys > size(image,1)) = size(image,1);
    % extract from image
    if origion_size == target_sz
        IMG_R = image(ys, xs, :);
    else
        im = image(ys, xs, :);
        % resize to reference size
        IMG_R = mexResize(im, target_sz);
    end
else
    scalar = origion_size./target_sz;
    % use ba_interp2
    w = target_sz(1);
    h = target_sz(2);
    [Dx, Dy] = meshgrid(-floor(w/2):floor((w-1)/2), -floor(h/2):floor((h-1)/2));
    R = [scalar(1) * cos(rotate_angle) scalar(1) * sin(rotate_angle); -scalar(2) * sin(rotate_angle) scalar(2) * cos(rotate_angle)];%shun
    RD = R * [Dx(:)'; Dy(:)'];
    RDx = reshape(RD(1,:), size(Dx))+center(1);
    RDy = reshape(RD(2,:), size(Dy))+center(2);
    
    % nearest, linear, or cubic.
    IMG_R = ba_interp2(image, RDx, RDy, method);
    %
    IMG_R(IMG_R<0) = 0;
    IMG_R(IMG_R>1) = 1;
end

end