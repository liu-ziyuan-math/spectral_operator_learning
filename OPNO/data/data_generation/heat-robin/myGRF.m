% sum{ c_k \phi_k e_k } with \phi_k = T_k - \frac{k-1}{k+3} T_{k+2}
% 
function u = myGRF(N, gamma, sigma)
p = ((0:N-2).^2 +1)' ./ ((2:N).^2 +1)';
norm_factor = (1+p.^2)*pi/2;
norm_factor(1) = (2+p(1).^2)*pi/2;

coef = sigma.* gamma.^(-(1:N-1)') .* randn(N-1, 1) ./ norm_factor;


Tcoef = zeros(N+1, 1);
Tcoef(1:2) = coef(1:2);
Tcoef(3:N-1) = coef(3:N-1) - p(1:N-3) .* coef(1:N-3);
Tcoef(N:N+1) = -p(N-2:N-1).*coef(N-2:N-1);

u = chebfun(Tcoef, 'coeffs');