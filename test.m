clear;clc;close all

maxsteps = 100;

xt = 0;
yt = 0;
thetat = pi/2;
r = 0;
l = 0;

thetat = wrapToPi(thetat - pi/2)^2;

r1 = 10*((xt^2 + yt^2 + thetat^2) < 0.5);

r2 = -100*(abs(xt) >= 20 || abs(yt) >= 20);

r3 = -(.2*(r + l)^2 +.3*(r - l)^2 + .03*xt^2 + .03*yt^2 + .02*thetat^2);

rt = r1 + r2 + r3;

rt = rt*maxsteps

%%
thw = linspace(-pi,pi,41);
rwold = (thw - pi/2).^2;
rw = wrapToPi(thw - pi/2).^2;

h = figure;grid on;xlabel('Theta Wrapped');ylabel('Penalty Contribution')
h.WindowStyle = 'Docked';
hold on
plot(thw*180/pi, rwold, 'o-')
plot(thw*180/pi, rw, '*-')
legend('old', 'new')


%%
clear;clc;close all

A = [0 1; 0 0];
B = [0; .2]*4;
C = [1 0];
D = 0;

% kx = lqr(A, B, 1*eye(2), .01);
kx = place(A, B, [-.5; -1]*2)

Acl = A - B*kx
kr = -Acl(2,1)/B(2);

sys = ss(Acl,[0;-Acl(2,1)],C,D);

step(sys)

r = 180*pi/180;
x = [0; 0];
u = 0;
t =  0;
tend = 10; dt = .01; iend = round(tend/dt);
for i = 1:iend
    xs(:, i) = x;
    us(i) = u;
    ts(i) = t;

    u = -kx*x + kr*r;

    xdot = A*x + B*u;
    x = x + xdot*dt;
    t = t + dt;
end

h = figure;
h.WindowStyle = 'Docked';
plot(ts, xs(1, :)*180/pi)
grid on
xlabel('Time (s)')
ylabel('Angle (deg)')

h = figure;
h.WindowStyle = 'Docked';
plot(ts, us(1, :))
grid on
xlabel('Time (s)')
ylabel('Thrust (~)')

















