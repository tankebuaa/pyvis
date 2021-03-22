
function out = normalize_img(img)
    min_val = min(img(:));
    max_val = max(img(:));
    if (max_val - min_val) > 0
        out = (img - min_val)/(max_val - min_val);
    else
        out = zeros(size(img));
    end
end  % endfunction