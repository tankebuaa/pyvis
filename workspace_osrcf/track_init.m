function [state, region] = track_init(image, region)
% region and rotation
if numel(region) == 8%poly
    x1 = min(region(1:2:end));
    x2 = max(region(1:2:end));
    y1 = min(region(2:2:end));
    y2 = max(region(2:2:end));
    pos = [(x1+x2)/2 (y1+y2)/2];
    width = round(0.5*(norm(region(1:2)-region(3:4), 2)+norm(region(5:6)-region(7:8),2)));
    height = round(0.5*(norm(region(3:4)-region(5:6), 2)+norm(region(7:8)-region(1:2),2)));
    target_sz = [width height];
    % we set the center edge point as the angle pointer
    ra = atan2(pos(2)-(region(2)+region(4))/2, (region(1)+region(3))/2 - pos(1)) - pi/2;
    cx = mean(region(1:2:end));
    cy = mean(region(2:2:end));
    A1 = norm(region(1:2) - region(3:4)) * norm(region(3:4) - region(5:6));
    A2 = (x2 - x1) * (y2 - y1);
    s = sqrt(A1/A2);
    w = s * (x2 - x1) + 1;
    h = s * (y2 - y1) + 1;
%     rect = round([cx-w/2 cy-h/2 w h]);
    rect = double([x1 y1 x2-x1 y2-y1]);
    state.bb_rect = rect;
    params.use_orientation = true;
    params.use_initra = false;
    params.ra_space = [-pi/6,-pi/12, 0, pi/12, pi/6];
elseif numel(region) == 4% rect
    rect = region;
    state.bb_rect = rect;

    pos = region(1:2) + floor(region(3:4)/2);
    target_sz = region(3:4);
    % poly
    LT = pos - target_sz/2;
    RT = pos + [target_sz(1) -target_sz(2)]/2;
    RB = pos + target_sz/2;
    LB = pos + [-target_sz(1) target_sz(2)]/2;
    region = [LT RT RB LB];
    state.bb_poly=region;
    ra = 0;
    params.use_initra = true;
    params.max_peak = 0;
    params.use_orientation = true;
    params.ra_space = [-pi/9, 0, pi/9];
end

% debug
params.debug = 0;

% poor shape
if max(target_sz) /min(target_sz)>4
   params.crop_factor = 0.55-4/2*min(target_sz)/max(target_sz);%(new-min)/2/new=0.5-0.5*min/new%0.55
   target_sz(target_sz == min(target_sz)) = max(target_sz)/4;
else
    params.crop_factor = 0.05;%0.05
end

% hog params
features_params.hog_cell_size = 4;
features_params.hog_orientations = 9;

% cn params
temp = load('w2crs');
features_params.w2c = temp.w2crs;
        
% Image sample parameters
params.search_area_scale = 5;
params.max_search_area = 300^2;
params.min_search_area = 200^2;
params.interp_method = 'cubic'; % 'nearest'  'linear' 'cubic'


% Learning parameters
params.output_sigma_factor = 1/16;		% Label function sigma
params.lambda = 1e-2;                   % Regularization
params.learning_rate = 0.05;	 	 	% Learning rate0.05

% channal weight
params.use_channel_weights = 0;
params.channels_weight_lr = params.learning_rate;

% scale adaptation parameters (from DSST)
params.currentScaleFactor = 1.0;
params.n_scales = 33;
params.scale_model_factor = 1.0;
params.scale_sigma_factor = 1/4;
params.scale_step = 1.02;
params.scale_model_max_area = 32*16;
params.scale_sigma = sqrt(params.n_scales) * params.scale_sigma_factor;
params.scale_lr = 0.025;

% Check if color image
img = im2double(image);
if size(img,3) == 3
    features_params.is_color_image = true;
else
    features_params.is_color_image = false;
end
params.features_params = features_params;
%% template params
% serach window size
search_window_sz = repmat(sqrt(prod(target_sz)) * params.search_area_scale, 1, 2); % square area, ignores the target aspect ratio
% Calculate search area and initial scale factor
search_area = prod(search_window_sz);
if search_area > params.max_search_area
    ScaleFactor = sqrt(search_area / params.max_search_area);
elseif search_area < params.min_search_area
    ScaleFactor = sqrt(search_area / params.min_search_area);
else
    ScaleFactor = 1.0;
end
% template size is fixed later
template_size = floor(search_window_sz / ScaleFactor);
% use orientation or not
if params.use_orientation
    norm_search_window = get_patch(img, pos, search_window_sz, template_size, ra, params.interp_method);
else
    norm_search_window = get_sub_window(img, pos, search_window_sz, template_size);
end

%% build mask
% target size in template
norm_target_sz = round(target_sz./search_window_sz.*template_size);

% target mask, use target region for approximation (crop)
mask = zeros(template_size([2,1]));
crop_factor = 1 - params.crop_factor;
mask(floor((template_size(2)-crop_factor*norm_target_sz(2))/2):end-floor((template_size(2)-crop_factor*norm_target_sz(2))/2),...
    floor((template_size(1)-crop_factor*norm_target_sz(1))/2):end-floor((template_size(1)-crop_factor*norm_target_sz(1))/2)) = 1;

%% get feature and regression target
[feat, feat_sz]= extract_feature(norm_search_window, params.features_params);

% gaussian-shaped desired response, centred in (1,1)
% bandwidth proportional to target size
output_sigma = sqrt(prod(norm_target_sz)) * params.output_sigma_factor / features_params.hog_cell_size;
% output_sigma = 1.0;
y = gaussian_shaped_labels(output_sigma, feat_sz);
yf = fft2(y);
% Hann (cosine) window
cos_window = single(hann(size(yf,1)) * hann(size(yf,2))');
%% regularized filter
% apply Hann window
x = bsxfun(@times, cos_window, feat);

% filter mask
filter_mask = imresize(mask,feat_sz,'nearest');

% build RRCF model
hf = build_cf(x, yf, filter_mask, zeros(size(feat)));

%=====================per-channel feature weights================
if params.use_channel_weights
    response = real(ifft2(fft2(x).*conj(hf)));
    chann_w = max(reshape(response, [size(response,1)*size(response,2), size(response,3)]), [], 1);
    % normalize: sum = 1
    params.chann_w = chann_w / sum(chann_w);
end

%======================scales filters setup=======================
% label function for the scales
ss = (1:params.n_scales) - ceil(params.n_scales/2);
% create gaussian shaped labels
ys = exp(-0.5 * (ss.^2) / params.scale_sigma^2);
ysf = single(fft(ys));

% scales cosine window
if mod(params.n_scales,2) == 0
    scale_cos_win = single(hann(params.n_scales+1));
    scale_cos_win = scale_cos_win(2:end);
else
    scale_cos_win = single(hann(params.n_scales));
end

% scales factors list
ss = 1:params.n_scales;
scaleFactors = params.scale_step.^(ceil(params.n_scales/2) - ss);% 

% scale model factor, rescaled to a fixed area
if params.scale_model_factor^2 * prod(template_size) > params.scale_model_max_area
    scale_model_factor = sqrt(params.scale_model_max_area/prod(template_size));
end
scale_model_sz = floor(template_size * scale_model_factor);

scaleSizeFactors = scaleFactors;
params.min_scale_factor = params.scale_step ^ ceil(log(max(5 ./ template_size)) / log(params.scale_step));
params.max_scale_factor = params.scale_step ^ floor(log(min([size(image,1) size(image,2)] ./ target_sz)) / log(params.scale_step));

% scale search model 
xs = get_scale_subwindow(norm_search_window, template_size/2, norm_target_sz([2,1]), scaleSizeFactors, scale_cos_win, scale_model_sz([2,1]));
% fft over the scale dim
xsf = fft(xs,[],2);
sf_num = bsxfun(@times, ysf, conj(xsf));
sf_den = sum(xsf .* conj(xsf), 1);
%% init status

state.pos = pos;
state.target_sz = target_sz;
state.ra = ra;
state.bb_poly = region;

% -----------------params--------------------
params.template_szie = template_size;
params.cos_window = cos_window;

params.filter_mask = filter_mask;
params.yf = yf;
params.model_hf = hf;
params.init_hf = hf;

params.ysf = ysf;
params.sf_num = sf_num;
params.sf_den = sf_den;

params.scale_cos_win = scale_cos_win;
params.scaleFactors = scaleFactors;
params.scale_model_sz = scale_model_sz;
params.scaleSizeFactors = scaleSizeFactors;
% -------------------------------------------
state.params = params;
end