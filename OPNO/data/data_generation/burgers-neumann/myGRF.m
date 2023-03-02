%Radom function from N(m, C) on [0 1] where
%C = sigma^2(-Delta + tau^2 I)^(-gamma)
%with periodic, zero dirichlet, and zero neumann boundary.
%Dirichlet only supports m = 0.
%N is the # of Fourier modes, usually, grid size / 2.
function u = myGRF(N, gamma, tau, sigma, bc)
N = N/2;
my_eigs = (abs(sigma).*((pi.*(1:N)').^2 + tau^2).^(-gamma/2));
% my_eigs(M+1:N) = 0;

if bc.type == "dirichlet"
    alpha = zeros(N,1);
else
    xi_alpha = randn(N,1);
    alpha = my_eigs.*xi_alpha;
end

if bc.type == "neumann"
    beta = zeros(N,1);
else
    xi_beta = randn(N,1);
    beta = my_eigs.*xi_beta;
end

a = alpha/2;
b = -beta/2;

c = [flipud(a) - flipud(b).*1i;0*1i;a + b.*1i];

trig = chebfun(c, [-2*pi 2*pi], 'trig', 'coeffs');

if bc.type == "dirichlet"
    %u = chebfun(trig(pi*chebpts(2*N+1))) + (bc.l+bc.r)/2 + (bc.r-bc.l)/2*chebfun('x');
    u = chebfun(@(t) trig(pi*(t+1)) + (bc.l+bc.r)/2 + (bc.r-bc.l)/2*t , [-1 1]);
end


if bc.type == "neumann"
    %u = chebfun(trig(pi*chebpts(2*N+1))) + (bc.l+bc.r)/2 + (bc.r-bc.l)/2*chebfun('x');
    u = chebfun(@(t) trig(pi*(t+1)) , [-1 1]);
end

end