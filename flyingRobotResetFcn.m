function in = flyingRobotResetFcn(in)
% Randomize the position of the flying robot around a circle of radius R
% and the initial orientation of the robot.
R = 15;
t0 = 2*pi*rand();
t1 = 2*pi*rand();
x0 = cos(t1)*R;
y0 = sin(t1)*R;
in = setVariable(in, 'xmax', 100);
in = setVariable(in, 'ymax', 100);

in = setVariable(in,'theta0',t0);
in = setVariable(in,'x0',x0);
in = setVariable(in,'y0',y0);

% in = setVariable(in,'theta0',-90*pi/180);
% in = setVariable(in,'x0',10);
% in = setVariable(in,'y0',10);
