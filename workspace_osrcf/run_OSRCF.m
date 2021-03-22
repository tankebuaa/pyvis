function results = run_OSRCF(seq, res_path, bSaveImage)
if isfield(seq, 'ground_truth')
    ground_truth = seq.ground_truth;
end
s_frames = seq.s_frames;
start_frame = 1;
rect_position = zeros(numel(s_frames), 4);
duration = 0;
for frame = start_frame : seq.len
    
    % read image
    im = imread(s_frames{frame});
    tic
    
    if frame == start_frame
        % initialize
        region = seq.init_rect;
        [state, region] = track_init(im, region);
        
        % load ground truth
        if isfield(seq, 'ground_truth')
            state.ground_truth = ground_truth;
        end
    else
        % update
        [state, region] = track_update(im, state);
    end
    state.num_frame = frame;
    
    duration = duration + toc;
    rect_position(frame,:) = state.bb_rect;
    if bSaveImage
        rectangle('Position', rect_position(frame,:), 'Linewidth', 4, 'EdgeColor', 'r');
        text(10, 15, ['#' num2str(frame)], 'Color','y', 'FontWeight','bold', 'FontSize',24);
        drawnow;

%         saveas(gcf, [res_path num2str(i) '.jpg']);
        imwrite(frame2im(getframe(gcf)),[res_path num2str(frame) '.jpg']);
    end
    figure(1);
    imshow(im);hold on;
    rectangle('Position',rect_position(frame,:), 'Linewidth', 2, 'EdgeColor', 'r');
    if isfield(seq, 'ground_truth')
        draw_region(seq.ground_truth(frame, :), [0 0 1]);
        draw_region(region, [1 0 1]);
    end
    text(10, 15, ['#' num2str(frame)], 'Color','y', 'FontWeight','bold', 'FontSize',24);
    drawnow;
    hold off;
end

results.type = 'rect';
results.res=rect_position;
results.fps=(seq.len-1)/duration;
disp(['fps: ' num2str(results.fps)])

end