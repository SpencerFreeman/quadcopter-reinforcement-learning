clear;clc;close all

maxsteps = 75;

xt = 0;
yt = 0;
thetat = pi/2;
r = 0;
l = 0;


r1 = 10*((xt^2 + yt^2 + thetat^2) < 0.5);

r2 = -100*(abs(xt) >= 20 || abs(yt) >= 20);

r3 = -(.2*(r + l)^2 +.3*(r - l)^2 + .03*xt^2 + .03*yt^2 + .02*thetat^2);

rt = r1 + r2 + r3

rt = rt*maxsteps














