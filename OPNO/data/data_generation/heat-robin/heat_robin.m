function u = heat_robin(init, tspan, k, Eps)

bc.left = @(u) u-diff(u);
bc.right = @(u) u+diff(u);
% f = @(t, x, u) k*diff(u, 2)+0.01*sin(x);
f = @(t, x, u) k*diff(u, 2);
opts = pdeset('Eps', Eps, 'Ylim', [0 20]);
u = pde15s(f, tspan, init, bc, opts);
% u = pde23t(f, tspan, init, bc, opts);
u = u(:, length(tspan));

