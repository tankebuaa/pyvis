function H = build_cf(x, yf, M, pH)
% bulid correlation filter using ADMM
% create filter with Augmented Lagrangian iterative optimization method
% input parameters:
% img: image patch (already normalized)
% Y: gaussian shaped labels (note that the peak must be at the top-left corner)
% M: padding mask (zeros around, 1 around the center), shape: box
% pH: previous regularized filter
%2,2,2,2,0.01,15
%5,3,20,4,0.01,7
%%%%%
%    mu|
%gama  | 0   |   1   |   2   |   3   |   4   |   5   |   6  
%  0    X       X        X      X       X       X       X
%  4    X       X        X      X       X       X       X
%  8    X       X        X      X       X       X       X
%  12-   X      X       X        Y      Ysa       Y       Y
%  16    X      X       X      X       X        X       X                                X
%  20    X       X        X        X       X      X      X
%  24    X       X        X        X       X      X      X
%%%%%
%% vot
mu =5;
mu_step =  3;
mu_max = 15;
max_iters = 2;
lambda = 1e-2;
gama = 10;

% %% otb
% mu =5;%5
% mu_step =  3;%2,3
% mu_max = 16;%1,16
% max_iters = 3;%2,3
% lambda = 1e-2;%0.01

% % gama = binit * 15;
% gama = 17;%>14,17

%11%13
xf = fft2(x);

Sxy = bsxfun(@times, xf, conj(yf));
Sxx = xf.*conj(xf);

% mask filter
H = fft2(bsxfun(@times, ifft2(bsxfun(@rdivide, Sxy, (Sxx + lambda))), M));

% initialize lagrangian multiplier
L = zeros(size(H));

iter = 1;
while true
    G = (Sxy + mu*H - L) ./ (Sxx + mu);
    H = fft2(real((1/(lambda + mu + gama)) * bsxfun(@times, M, ifft2(mu*G + L + gama * pH))));
    % stop optimization after fixed number of steps
    if iter >= max_iters
        break;
    end
    % update variables for next iteration
    L = L + mu*(G - H);
    mu = min(mu_max, mu_step*mu);
    iter = iter + 1;
end

end  % endfunction
