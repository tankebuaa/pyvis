function norm_search_window = get_sub_window(im, pos, search_window_sz,template_size)
%GET_SUBWINDOW Obtain sub-window from image, with replication-padding.
%   Returns sub-window of image IM centered at POS ([x, y] coordinates),
%   with size SZ ([height, width]). If any pixels are outside of the image,
%   they will replicate the values at the borders.


[h, w, ~] = size(im);

xs = floor(pos(2)) + (1:search_window_sz(2)) - floor(search_window_sz(2)/2);
ys = floor(pos(1)) + (1:search_window_sz(1)) - floor(search_window_sz(1)/2);

%check for out-of-bounds coordinates, and set them to the values at
%the borders
xs(xs < 1) = 1;
ys(ys < 1) = 1;
xs(xs > h) = h;
ys(ys > w) = w;

%extract image
search_im = im(xs, ys, :);
% resize to reference size
norm_search_window = mexResize(search_im, template_size);
end