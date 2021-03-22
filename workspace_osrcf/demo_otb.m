%--------------------
clear;

% configure
sequences_base_path =  'D:\STUDY\CODE\Matlab\benchmark\tracker_benchmark\data_seq\OTB100';
res_path = './res/';
bSaveImage = 0;

% 
ground_path = choose_sequence(sequences_base_path);
seq.ground_truth = dlmread([ground_path 'groundtruth_rect.txt']);
seq.len = size(seq.ground_truth, 1);
seq.init_rect = seq.ground_truth(1,:);
seq.ext = 'jpg';
seq.s_frames = {};
seq.path = [ground_path 'img/'];
for i=1:seq.len
        id = sprintf('%04d',i);
        seq.s_frames{i} = strcat(seq.path,id,'.',seq.ext);
end

% run tracker
run_OSRCF(seq, res_path, bSaveImage)