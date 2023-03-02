function u = burgers_cgl(init, tspan, N, visc, Eps, bc)

if bc.type == "dirichlet" && abs(bc.l-init(-1)) + (bc.r-init(1)) > 1e-12
    disp("inital condition error!")
end

f = @(t, x, u) -.5*diff(u.^2) + visc*diff(u, 2);
opts = pdeset('Eps', Eps, 'Ylim', [0 20]);
u = pde15s(f, tspan, init, bc.type, opts);
u = u(:, length(tspan));

