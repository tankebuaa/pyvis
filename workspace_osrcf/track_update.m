function [state, region] = track_update(image, state)
params = state.params;

% target state
pos = state.pos;
target_sz = state.target_sz;
ra = state.ra;

% RRCF filter
template_size = params.template_szie;
hf = params.model_hf;

% scale filter
sf_den = params.sf_den;
sf_num = params.sf_num;

img = im2double(image);
%% TRANSLATION SPACE DETECT
% get search patch
search_window_sz = repmat(floor(sqrt(prod(target_sz)) * params.search_area_scale), 1, 2);

% get testing patch -- use orientation or not
if params.use_orientation
    norm_search_window = get_patch(img, double(pos), search_window_sz, template_size, ra, params.interp_method);
else
    norm_search_window = get_sub_window(img, pos, search_window_sz, template_size);
end

% extract feature
[feat, feat_sz]= extract_feature(norm_search_window, params.features_params);

% apply Hann window
z = bsxfun(@times, params.cos_window, feat);

% equation for fast detection  
if ~params.use_channel_weights
    response = real(ifft2(sum(fft2(z).*conj(hf), 3)));
else
    response_chann = real(ifft2(fft2(z).*conj(hf)));
    response = sum(bsxfun(@times, response_chann, reshape(params.chann_w, 1, 1, size(response_chann,3))), 3);
end

% find max
peak = max(response(:));
[row, col] = find(response == peak, 1);

% subpixel accuracy: response map is smaller than image patch  -  due to HoG histogram (cell_size > 1)
v_neighbors = response(mod(row + [-1, 0, 1] - 1, size(response,1)) + 1, col);
h_neighbors = response(row, mod(col + [-1, 0, 1] - 1, size(response,2)) + 1);

row = row + subpixel_peak(v_neighbors);
col = col + subpixel_peak(h_neighbors);

% wrap around
if row > size(response,1) / 2,
    row = row - size(response,1);
end
if col > size(response,2) / 2,
    col = col - size(response,2);
end

% backproject to the real img_axes pos
if params.use_orientation
    trans_pos = pos + ([col - 1, row - 1].*search_window_sz./feat_sz) * [cos(ra) -sin(ra);sin(ra) cos(ra)];
else
    trans_pos = pos + ([col - 1, row - 1].*search_window_sz./feat_sz);
end

% calculate detection-based weights
if params.use_channel_weights
    channel_discr = ones(1, size(response_chann, 3));
    for i = 1:size(response_chann, 3)
        norm_response = normalize_img(response_chann(:, :, i));
        local_maxs_sorted = localmax_nonmaxsup2d(squeeze(norm_response(:, :)));
        
        if local_maxs_sorted(1) == 0, continue; end;
        channel_discr(i) = 1 - (local_maxs_sorted(2) / local_maxs_sorted(1));
        
        % sanity checks
        if channel_discr(i) < 0.5, channel_discr(i) = 0.5; end;
    end
end
% visual
if params.debug
    % ========================test========================
    figure(9);surf(fftshift(response)),shading flat;disp(['trans peak:' num2str(peak)]);
    figure(3);subplot(221);imshow(norm_search_window);hold on;handle = imagesc(mexResize(fftshift(response), template_size));colorbar;alpha(handle, 0.8);title('trans response');
end

%% ROTATION SPACE DETECT
if params.use_orientation
    % num of orientations
    ras = ra + params.ra_space;
    if params.use_initra
        ras = [ras 0];
    end
    num_ra_space = numel(ras);
    ra_space_response = zeros([feat_sz num_ra_space],'single');
    peaks = zeros(1, num_ra_space, 'single');
%     ra_space_chann = zeros([size(feat,1) size(feat,2) size(feat,3) num_ra_space]);
    % iteration on orientation space
    for i = 1:num_ra_space
        norm_search_window = get_patch(img, double(trans_pos), search_window_sz, template_size, ras(i), params.interp_method);
        [feat, ~]= extract_feature(norm_search_window, params.features_params);
        % apply Hann window
        z = bsxfun(@times, params.cos_window, feat);
        % equation for fast detection
        %         if ~params.use_channel_weights
        %             response_cf = real(ifft2(sum(fft2(z).*conj(hf), 3)));
        %         else
        %             response_chann = real(ifft2(fft2(z).*conj(hf)));
        %             ra_space_chann(:,:,:,i) = response_chann;
        %             response_cf = sum(bsxfun(@times, response_chann, reshape(params.chann_w, 1, 1, size(response_chann,3))), 3);
        %         end
        response_cf = real(ifft2(sum(fft2(z).*conj(hf), 3)));
        ra_space_response(:,:,i) = response_cf;
        peaks(i) = max(response_cf(:));
        % visual
        if params.debug
            figure(8);imshow(norm_search_window);
            figure(9);surf(fftshift(response_cf)),shading flat;disp(['trans peak:' num2str(max(response_cf(:)))]);axis off;
%           figure(3);subplot(222);imshow(norm_search_window);hold on;handle =imagesc(mexResize(fftshift(response_cf), template_size));colorbar;alpha(handle, 0.5);
        end
        
    end
    peak_ra = max(peaks);
%     disp(peak_ra);
    if params.use_initra
        params.max_peak = max(params.max_peak, peak_ra);
        if peak_ra > 1.05*peaks(2)
            rotate_id = find(peaks == peak_ra, 1);
        else
            if peaks(4) == peak_ra
                rotate_id = 4;
            else
                rotate_id = 2;
            end
        end
        params.max_peak = max(params.max_peak, peak_ra);
    else
        rotate_id = find(peaks == peak_ra, 1);
    end
    response = ra_space_response(:,:,rotate_id);
    [row, col] = find(response == peaks(rotate_id), 1);
%     if params.use_channel_weights
%         response_chann = ra_space_chann(:,:,:,rotate_id);
%     end
    % subpixel accuracy: response map is smaller than image patch - due to HoG histogram (cell_size > 1)
    v_neighbors = response(mod(row + [-1, 0, 1] - 1, size(response,1)) + 1, col);
    h_neighbors = response(row, mod(col + [-1, 0, 1] - 1, size(response,2)) + 1);
    row = row + subpixel_peak(v_neighbors);
    col = col + subpixel_peak(h_neighbors);
    % wrap around
    if row > size(response,1) / 2,
        row = row - size(response,1);
    end
    if col > size(response,2) / 2,
        col = col - size(response,2);
    end
    
    % get the rotation of the object
    rotate_ra = ras(rotate_id);
    cosra = cos(rotate_ra);
    sinra = sin(rotate_ra);
    rotate_pos = trans_pos + ([col - 1, row - 1].*search_window_sz./feat_sz) * [cosra -sinra;sinra cosra];
    if params.use_initra && peak_ra < 0.25 * params.max_peak
        rotate_pos = pos;
        rotate_ra = 0;
%         disp(params.max_peak);
    end
else
    rotate_pos = trans_pos;
end


%% SCALE SPACE DETECT
% get current norm search patch
if params.use_orientation
    norm_search_window = get_patch(img, double(rotate_pos), search_window_sz, template_size, rotate_ra, params.interp_method);
else
    norm_search_window = get_sub_window(img, rotate_pos, search_window_sz, template_size);
end

% extract scale spacve feature
xs = get_scale_subwindow(norm_search_window, template_size/2, target_sz([2,1])./search_window_sz.*template_size, params.scaleSizeFactors, params.scale_cos_win, params.scale_model_sz([2,1]));
xsf = fft(xs,[],2);

% scale correlation response
scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + 1e-2) ));
recovered_scale = ind2sub(size(scale_response),find(scale_response == max(scale_response(:)), 1));

%set the scale
currentScaleFactor = params.scaleFactors(recovered_scale);


%% get the final results
% set center position
pos = double(rotate_pos);

% set object size --check for min/max scale
if currentScaleFactor < params.min_scale_factor
    currentScaleFactor = params.min_scale_factor;
elseif currentScaleFactor > params.max_scale_factor
    currentScaleFactor = params.max_scale_factor;
end
target_sz = currentScaleFactor*target_sz;

if params.use_orientation
    % orientation -- norm
    %     ra = 0.25 * ra + 0.75 * rotate_ra;%337
    ra=double(rotate_ra);%3312
    % poly
    cosra = cos(ra);
    sinra = sin(ra);
    LT = pos - target_sz/2*[cosra -sinra;sinra cosra];
    RT = pos + [target_sz(1) -target_sz(2)]/2*[cosra -sinra;sinra cosra];
    RB = pos + target_sz/2*[cosra -sinra;sinra cosra];
    LB = pos + [-target_sz(1) target_sz(2)]/2*[cosra -sinra;sinra cosra];
    region = double([LT RT RB LB]);
    % rect
    cx = mean(region(1:2:end));
    cy = mean(region(2:2:end));
    x1 = min(region(1:2:end));
    x2 = max(region(1:2:end));
    y1 = min(region(2:2:end));
    y2 = max(region(2:2:end));
    A1 = norm(region(1:2) - region(3:4)) * norm(region(3:4) - region(5:6));
    A2 = (x2 - x1) * (y2 - y1);
    s = sqrt(A1/A2);
    w = s * (x2 - x1) + 1;
    h = s * (y2 - y1) + 1;
    rect = double(round([cx-w/2 cy-h/2 w h]));
%     rect = double([x1 y1 x2-x1 y2-y1]);
else
    % poly
    LT = pos - target_sz/2;
    RT = pos + [target_sz(1) -target_sz(2)]/2;
    RB = pos + target_sz/2;
    LB = pos + [-target_sz(1) target_sz(2)]/2;
    region = double([LT RT RB LB]);
    % rect
    rect = double(round([pos-target_sz/2 target_sz]));
end

% visual
% if params.debug
%     figure(2);imshow(image);hold on;
%     rectangle('Position',round(rect), 'EdgeColor','g');
%     draw_region(region, [0 1 0]);
%     % truth
%     truth = state.ground_truth(state.num_frame + 1,:);
%     draw_region(truth, [0 0 1]);hold off;
% end

%% Update Model
% -----------------------update RRCF model-----------------------
% serach window size
search_window_sz = repmat(floor(sqrt(prod(target_sz)) * params.search_area_scale), 1, 2); % square area, ignores the target aspect ratio

% get current norm train patch
if params.use_orientation
    norm_search_window = get_patch(img, double(pos), search_window_sz, template_size, ra, params.interp_method);
else
    norm_search_window = get_sub_window(img, pos, search_window_sz, template_size);
end

if params.debug
    figure(3);
    subplot(224);imshow(norm_search_window);hold on;handle=imagesc(imresize(params.filter_mask, template_size, 'nearest'));alpha(handle,0.5);
end
% if max(norm_search_window(:)) >1 || min(norm_search_window(:)) < 0
%    disp('error'); 
% end
% extract features
[feat, ~]= extract_feature(norm_search_window, params.features_params);

% apply Hann window
x = bsxfun(@times, params.cos_window, feat);

% solve RRCF model
new_hf = build_cf(x, params.yf, params.filter_mask, hf);

% update RRCF model
params.model_hf = (1-params.learning_rate) * hf + params.learning_rate * new_hf;

% -----------------------update channal weights model-----------------------
if params.use_channel_weights
    response = real(ifft2(fft2(x).*conj(new_hf)));
    % calculate per-channel feature weights
    new_chann_w = max(reshape(response, [size(response,1)*size(response,2), size(response,3)]), [], 1) .* channel_discr;
    new_chann_w = new_chann_w / sum(new_chann_w);
    chann_w = (1-params.channels_weight_lr)*params.chann_w + params.channels_weight_lr*new_chann_w;
    params.chann_w = chann_w / sum(chann_w);
end


%-----------------------update scale space model-----------------------
% get scale space model
xs = get_scale_subwindow(norm_search_window,template_size/2, target_sz([2,1])./search_window_sz.*template_size, params.scaleSizeFactors, params.scale_cos_win, params.scale_model_sz([2,1]));

% fft over the scale dim
xsf = fft(xs,[],2);
new_sf_num = bsxfun(@times, params.ysf, conj(xsf));
new_sf_den = sum(xsf .* conj(xsf), 1);

% auto-regressive scale filters update
params.sf_den = (1 - params.scale_lr) * sf_den + params.scale_lr * new_sf_den;
params.sf_num = (1 - params.scale_lr) * sf_num + params.scale_lr * new_sf_num;


%% update state
state.pos = pos;
state.target_sz = target_sz;
state.ra = ra;
state.bb_poly = region;
state.bb_rect = rect;
state.params = params;
end