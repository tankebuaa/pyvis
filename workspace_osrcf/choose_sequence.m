function sequence_path = choose_sequence(base_path)

% sequence_path = choose_sequence(base_path)

%process path to make sure it's uniform
if ispc(), base_path = strrep(base_path, '\', '/'); end
if base_path(end) ~= '/', base_path(end+1) = '/'; end

%list all sub-folders
contents = dir(base_path);
names = {};
for k = 1:numel(contents),
    name = contents(k).name;
    if isdir([base_path name]) && ~strcmp(name, '.') && ~strcmp(name, '..')&&~strcmp(name, 'list.txt')
        names{end+1} = name;  %#ok
    end
end

%no sub-folders found
if isempty(names), sequence_path = []; return; end

%choice GUI
choice = listdlg('ListString',names, 'Name','Choose sequence', 'SelectionMode','single');

if isempty(choice),  %user cancelled
    sequence_path = [];
else
    sequence_path = [base_path names{choice} '/'];
end

end

