function [img_files, ground_truth] = load_sequence(sequence_path)

content = dir([sequence_path '0*.*']);
img_files = {};
for k = 1:numel(content),
    name = content(k).name;
    if exist([sequence_path name], 'file')
        img_files{end+1} = [sequence_path name];  %#ok
    else
        error('No image files to load.')
    end
end

text_files = dir([sequence_path 'groundtruth.txt']);
assert(~isempty(text_files), 'No initial position and ground truth (*_gt.txt) to load.')

ground_truth = load([sequence_path text_files(1).name]);

end

