function [out, feat_sz]= extract_feature(image, params)
%% HOG feature channel 1:27
feat_hog = double(fhog(single(image), params.hog_cell_size, params.hog_orientations));
%% gray feature channel 28
h = size(feat_hog, 1);
w = size(feat_hog, 2);
feat_sz = [h w];
if params.hog_cell_size > 1
    im_patch = mexResize(image, [h, w] ,'auto');
end

% add cn
if params.is_color_image
%     im_gray = rgb2gray(im_patch);
    feat_hog(:,:,end) = [];%single(im_gray) - 0.5;
    % add CN feature
    feat_cn = im2c(single(im_patch*255), params.w2c, -2);
    % output
    out = cat(3,feat_hog,feat_cn);
else
%     im_gray = im_patch(:,:,1);
%     feat_hog(:,:,end) = single(im_gray) - 0.5;
    feat_hog(:,:,end) = [];
    out = feat_hog;
end



end