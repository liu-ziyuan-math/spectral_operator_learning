% number of realizations to generate
num = 1;
save_num = 100;

% parameters for the Gaussian random field
gamma = 2;
tau = 4;
sigma = 4;%0.3;
Eps = 1e-12;

gamma =  sqrt(2);
sigma = 0.3; %0.3
gamma =  1.1;
sigma = 4; %0.3

% viscosity
k = 0.02;
% visc = 0.001;

% grid size
N = 2^8;
M = N;
steps = 2;
%bc.type = 'dirichlet';
% bc.type = 'neumann';
bc.type = 'robin';

u0 = chebfun(zeros(N+1, save_num));
u1 = chebfun(zeros(N+1, save_num));

% u0_cgl = zeros(num, N+1);
% u1_cgl = zeros(num, N+1);
% u0_even = zeros(num, N+1);
% u1_even = zeros(num, N+1);


tspan = linspace(0,1,steps+1);
x_cgl = chebpts(N+1);
x_even = linspace(-1,1,N+1)';

for j=1:num
    u0(:, mod(j,save_num)+1) = myGRF(N, gamma, sigma);
    disp(j)
    disp(datestr(now))                                                                                                                                                                                     
    
    plot(u0(:, mod(j,save_num)+1), '.')
    hold on 
    tic
    u1(:, mod(j,save_num)+1)  = heat_robin(u0(:, mod(j,save_num)+1) , tspan, k, Eps);
    toc

    plot(u1(:, mod(j,save_num)+1))
    pause(0.1)
    hold off

%     u0_plus = u0(:, j) + diff(u0(:, j));
%     u1_plus = u0(:, j) + diff(u0(:, j));
%     u0_minus = u0(:, j) - diff(u0(:, j));
%     u1_minus = u0(:, j) - diff(u0(:, j));
%     bc_error = max( ...
%         abs([u0_plus(-1),abs(u1_plus(-1)),u0_minus(1),u0_minus(1)]) )
    
    if mod(j, save_num) == 0
        u0_cgl(j-save_num+1:j, :) = u0(x_cgl, :)';
        u0_even(j-save_num+1:j, :) = u0(x_even, :)';
        u1_cgl(j-save_num+1:j, :) = u1(x_cgl, :)';
        u1_even(j-save_num+1:j, :) = u1(x_even, :)';
        U0_cgl = u0_cgl;
        U0_even = u0_even;
        U1_cgl = u1_cgl;
        U1_even = u1_even;
%         save('burgers_robin_v2-2048-22.mat', 'U0_cgl', 'U1_cgl', 'U0_even', 'U1_even', 'N', 'gamma', 'tau', 'sigma', 'Eps', 'j')
%         save('heat_robin0large3w.mat', 'u0_cgl', 'u1_cgl', 'u0_even', 'u1_even', 'N', 'gamma', 'tau', 'sigma', 'Eps', 'j', 'k')
    end
% %     axis([-1, 1, -1, 1])
end


% kk = 0;
% lnum = 4000;
% 
% u0_cgl(lnum*kk+1:lnum*kk+lnum, :) = U0_cgl;
% u1_cgl(lnum*kk+1:lnum*kk+lnum, :) = U1_cgl;
% u0_even(lnum*kk+1:lnum*kk+lnum, :) = U0_even;
% u1_even(lnum*kk+1:lnum*kk+lnum, :) = U1_even;

% u0_cgl(4901:10000, :) = U0_cgl(1:5100, :);
% u1_cgl(4901:10000, :) = U1_cgl(1:5100, :);
% u0_even(4901:10000, :) = U0_even(1:5100, :);
% u1_even(4901:10000, :) = U1_even(1:5100, :);

% kk = kk+1;

% save('burgers_robin2.mat', 'u0_cgl', 'u1_cgl', 'u0_even', 'u1_even', 'N', 'gamma', 'tau', 'sigma', 'Eps', 'j')
