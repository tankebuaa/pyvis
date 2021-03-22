
% This file compiles the c++ function  into a Matlab mex file.

% Compile ba_interp2
mex -O ba_interp2.cpp
mex -lopencv_core242 -lopencv_imgproc242 -L./ -I./ mexResize.cpp MxArray.cpp
mex -O gradientMex.cpp