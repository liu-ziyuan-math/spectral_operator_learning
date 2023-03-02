clear
clc

% number of realizations to generate
num = 1;

% parameters for the Gaussian random field
gamma = 2;
tau = 5;
sigma = 25;
Eps = 1e-10;

% viscosity
visc = 0.1/(pi);

% grid size
N = 2^10;
steps = 2;
% bc.type = 'dirichlet';
bc.type = 'neumann';
bc.l = 0;
bc.r = 0;

u0 = chebfun(zeros(N+1, num));
u1 = chebfun(zeros(N+1, num));
tspan = linspace(0,1,steps+1);
x_cgl = chebpts(N+1);
x_even = linspace(-1,1,N+1)';

for j=1:num
    u0(:, j) = myGRF(N/2, gamma, tau, sigma, bc);
    plot(u0(:, j))
    
    tic
    u1(:, j)  = burgers_cgl(u0(:, j) , tspan, N, visc, Eps, bc);
    toc
    
    hold on 
    plot(u1(:, j))
    hold off 
    disp(j);
    
end


u0_cgl = u0(x_cgl, :)';
u0_even = u0(x_even, :)';
u1_cgl = u1(x_cgl, :)';
u1_even = u1(x_even, :)';
save('burgers_neumann.mat', 'u0_cgl', 'u1_cgl', 'u0_even', 'u1_even', 'N', 'gamma', 'tau', 'sigma', 'Eps', 'j')