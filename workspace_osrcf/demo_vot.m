% this is the matlab code for CCOSA
% refer to <<Learning an Orientation and Scale Adaptive Tracker with Correlation Filter and Color Model>>
clear;
close all;

addpath(pwd);

sequences_base_path = 'D:\STUDY\CODE\Matlab\benchmark\vot-toolkit-master\workspace_vot2018\sequences\';
sequence_path = choose_sequence(sequences_base_path);
if isempty(sequence_path), return, end  %user cancelled
[img_files, ground_truth] = load_sequence(sequence_path);

% initialize bounding box - [x,y,width, height]
use_reinitialization = 1;
gt = read_vot_regions(fullfile(sequence_path, 'groundtruth.txt'));
n_failures = 0;

% loop
time = zeros(1,numel(img_files));
start_frame = 1;
frame = 1;
while frame <= numel(img_files)
    
    % read image
    im = imread(img_files{frame});
    tic();
    
    if frame == start_frame
        % initialize
        region = ground_truth(start_frame,:);
        [state, region] = track_init(im, region);
        
        % load ground truth
        state.ground_truth = ground_truth;
    else
        % update
        [state, region] = track_update(im, state);
    end
    
    state.num_frame = frame;
    time(frame) = toc();
   
    if use_reinitialization  % detect failures and reinit
        area = rectint(state.bb_rect, gt(frame,:));
        if area < eps && use_reinitialization
            disp(['Failure detected. Reinitializing tracker...' num2str(frame)]);
            frame = frame + 4;  % skip 5 frames at reinit (like VOT)
            start_frame = frame + 1;
            n_failures = n_failures + 1;
        end
    end
    
    figure(1);
    imshow(im, 'Border','tight');%, 'InitialMag', 100 + 100 * (length(im) < 500));
    hold on;
%   rectangle('Position',ground_truth(frame, :), 'EdgeColor','y');
%   rectangle('Position',rect, 'EdgeColor','r');
    draw_region(state.bb_poly, [0 1 0]);
    draw_region(state.bb_rect, [1 0 0]);
    draw_region(ground_truth(frame, :), [0 0 1]);
    hold off;
    drawnow
    
    frame = frame + 1;
end

disp(frame/sum(time));
